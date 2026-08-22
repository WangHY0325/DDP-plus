# finetune_mask_ddp_plus.py
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter  # 【新增】引入 Tensorboard
from transformers import get_cosine_schedule_with_warmup
import random

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from utils import print_args, initialize, print_rank, get_tokenizer, get_model
from mask_pruner_ddp_plus import DDPPlusMaskPruner

t_sp = float(os.environ.get("TARGET_SPARSITY_FLOAT", 0.50))


def setup_model_and_pruner(args, device):
    print_rank("Loading base model...")
    model = get_model(args, device)
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 🚨 显存优化 1：显式确认梯度检查点已开启
    if getattr(args, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print_rank("Gradient checkpointing is ENABLED.")

    print_rank("Initializing DDP+ Mask Pruner...")
    pruner = DDPPlusMaskPruner(
        model=model,
        target_sparsity=getattr(args, "target_sparsity", t_sp),
        mu_0=0.50,
        mu_T=0.05
    )
    return model, pruner


def prepare_dataset(args, tokenizer):
    data = {}
    rng = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(
            args, tokenizer, args.data_dir, "train",
            args.train_num, args.train_ratio, rng,
        )
    return data


def _build_per_group_scheduler(optimizer, warmup_steps, total_steps):
    """
    改动 2：为每个参数组构建独立的 lr schedule
    - Group 0 (z):       cosine with warmup
    - Group 1 (λ1, λ3):  恒定 lr（不 warmup、不衰减）
    - Group 2 (λ2):      恒定 lr（不 warmup、不衰减）
    """
    num_groups = len(optimizer.param_groups)

    def lr_lambda_z(current_step):
        """Group 0: cosine with warmup"""
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def lr_lambda_const(current_step):
        """Group 1, 2: 恒定"""
        return 1.0

    lambdas = []
    for i in range(num_groups):
        if i == 0:
            lambdas.append(lr_lambda_z)
        else:
            lambdas.append(lr_lambda_const)

    return LambdaLR(optimizer, lr_lambda=lambdas)


def _compute_shifted_ce_kd_loss(
    student_logits,
    teacher_logits,
    input_ids,
    attn_mask,
    kd_T,
    chunk_tokens,
):
    """Compute the same shifted CE/KD loss in token chunks to reduce peak memory."""
    shift_labels = input_ids[..., 1:]
    if attn_mask is not None:
        shift_mask = attn_mask[..., 1:]
        shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

    vocab_size = student_logits.shape[-1]
    seq_len = shift_labels.shape[-1]
    ce_sum = student_logits.new_zeros((), dtype=torch.float32)
    kd_sum = student_logits.new_zeros((), dtype=torch.float32)
    valid_count = 0

    chunk_tokens = max(1, int(chunk_tokens))
    for start in range(0, seq_len, chunk_tokens):
        end = min(seq_len, start + chunk_tokens)
        labels_chunk = shift_labels[..., start:end].reshape(-1)
        valid = labels_chunk != -100
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue

        s_chunk = student_logits[..., start:end, :].reshape(-1, vocab_size)[valid].float()
        t_chunk = teacher_logits[..., start:end, :].reshape(-1, vocab_size)[valid].float()
        labels_valid = labels_chunk[valid]

        ce_sum = ce_sum + F.cross_entropy(s_chunk, labels_valid, reduction="sum")
        log_ps = F.log_softmax(s_chunk / kd_T, dim=-1)
        p_t = F.softmax(t_chunk / kd_T, dim=-1)
        kd_sum = kd_sum + F.kl_div(log_ps, p_t, reduction="sum") * (kd_T ** 2)
        valid_count += n_valid

    if valid_count == 0:
        return None, None, 0
    denom = float(valid_count)
    return ce_sum / denom, kd_sum / denom, valid_count


def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)

    # 【新增】初始化 writer 为 None
    writer = None
    if dist.get_rank() == 0:
        os.makedirs(args.save, exist_ok=True)
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

        # 【新增】在 rank 0 初始化 Tensorboard Writer
        tb_dir = os.path.join(args.save, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    device = torch.cuda.current_device()
    args.dtype = "torch.bfloat16"
    args.deepspeed_config = None

    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)

    dp_world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_rank = dist.get_rank() if dist.is_initialized() else 0

    if args.do_train:
        args.train_iters_per_epoch = int(
            len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps)
        )
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

    model, pruner = setup_model_and_pruner(args, device)
    if args.dtype == "torch.bfloat16":
        model = model.bfloat16()

    param_groups = pruner.get_optimizer_param_groups()
    optimizer = AdamW(param_groups)
    if dist.is_initialized() and dist.get_rank() != 0:
        pass
    else:
        for i, pg in enumerate(optimizer.param_groups):
            print(f"[optimizer] group {i}: lr={pg['lr']}, betas={pg.get('betas', 'default')}, "
                  f"params={len(pg['params'])}")

    # 改动 2：分组 scheduler — z warmup+cosine, λ 恒定
    scheduler = _build_per_group_scheduler(
        optimizer,
        warmup_steps=args.warmup_iters,
        total_steps=args.total_iters
    )

    if not args.do_train:
        return

    train_sampler = DistributedSampler(
        dataset["train"], shuffle=True, drop_last=True,
        rank=dp_rank, num_replicas=dp_world_size,
    )
    train_loader = DataLoader(
        dataset["train"], sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset["train"].collate,
    )

    loss_fn = nn.CrossEntropyLoss()
    amp_dtype = torch.bfloat16 if args.dtype == "torch.bfloat16" else torch.float32
    kd_T = getattr(args, "kd_temp", 2.0)
    kd_eta = getattr(args, "kd_weight", 2.0)
    loss_chunk_tokens = int(os.environ.get("LOSS_CHUNK_TOKENS", "256"))

    step, global_step = 1, 1

    print_rank(f"Start DDP+ Training. Total Iters: {args.total_iters}")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_loader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            model_batch.pop("labels", None)

            # --- Teacher Forward ---
            pruner.is_active = False
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs_t = model(**model_batch, use_cache=False)
                teacher_logits = outputs_t.logits.detach()

            # --- Student Forward ---
            pruner.is_active = True
            model.train()
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs_s = model(**model_batch, use_cache=False)
                student_logits = outputs_s.logits

            # --- Loss 计算 ---
            input_ids = model_batch["input_ids"]
            attn_mask = model_batch.get("attention_mask", None)

            loss_ce, loss_kd, valid_count = _compute_shifted_ce_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                input_ids=input_ids,
                attn_mask=attn_mask,
                kd_T=kd_T,
                chunk_tokens=loss_chunk_tokens,
            )
            if valid_count == 0:
                step += 1
                continue

            loss_task = loss_ce + kd_eta * loss_kd

            # --- 收集状态并反传 ---
            L_sp, L_bin, info = pruner.compute_mask_losses(step=global_step, total_steps=args.total_iters)
            info["ce_loss"] = loss_ce.item()
            info["kd_loss"] = loss_kd.item()

            loss = loss_task + L_sp + L_bin

            # Do not backpropagate a corrupted batch into the mask variables.
            # The frozen base model cannot recover once z/lambda become NaN.
            if not torch.isfinite(loss).all():
                print_rank(f"[DDP+] non-finite loss at step {global_step}; skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue

            # --- DEBUG: gradient decomposition + counterfactual (Y v1.4b) ---
            is_boundary = step % args.gradient_accumulation_steps == 0
            _do_grad = os.environ.get("Y_GRAD_DEBUG") == "1"
            _do_cf = os.environ.get("Y_CF_DEBUG") == "1"
            _do_cf_bal = os.environ.get("Y_CF_BAL_DEBUG") == "1"
            if _do_grad or _do_cf or _do_cf_bal:
                if is_boundary:
                    debug_steps_grad = [int(s.strip()) for s in os.environ.get("Y_GRAD_DEBUG_STEPS", "").split(",") if s.strip()]
                    debug_steps_cf = [int(s.strip()) for s in os.environ.get("Y_CF_DEBUG_STEPS", "").split(",") if s.strip()]
                    debug_steps_cf_bal = [int(s.strip()) for s in os.environ.get("Y_CF_BAL_DEBUG_STEPS", "").split(",") if s.strip()]
                    if _do_grad and global_step in debug_steps_grad and hasattr(pruner, "grad_debug"):
                        pruner.grad_debug(loss_task, L_sp, L_bin, global_step)
                    if _do_cf and global_step in debug_steps_cf and hasattr(pruner, "counterfactual_debug"):
                        pruner.counterfactual_debug(global_step)
                    if _do_cf_bal and global_step in debug_steps_cf_bal and hasattr(pruner, "counterfactual_balanced_debug"):
                        pruner.counterfactual_balanced_debug(global_step)

            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            # 🚨 显存优化 2：立刻删除巨大张量，防止在累加期间造成显存溢出 (OOM)
            del teacher_logits, student_logits, outputs_t, outputs_s

            if is_boundary:
                pruner.sync_mask_gradients()

                # Keep non-finite mask gradients from corrupting z/lambda updates.
                for _p in pruner.get_mask_param_list():
                    if _p.grad is not None:
                        _p.grad.data = torch.nan_to_num(
                            _p.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        _p.grad.data.clamp_(-10.0, 10.0)
                clip_val = pruner.compute_adaptive_clip_value()
                torch.nn.utils.clip_grad_norm_(pruner.get_mask_param_list(), clip_val)

                pruner.zero_dual_grads_if_satisfied()
                pruner.negate_dual_grads()

                optimizer.step()

                pruner.clamp_dual_variables()
                pruner.clamp_z_values()
                # Final finite guard after Adam and before the next forward.
                for _p in pruner.get_mask_param_list():
                    _p.data = torch.nan_to_num(_p.data, nan=1.0, posinf=10.0, neginf=-1.0)
                    _p.data.clamp_(-1.0, 10.0)

                scheduler.step()
                optimizer.zero_grad()

                # === 中间保存 (两阶段): step450 checkpoint → 独立 finalize ===
                if os.environ.get("SAVE_STEP450", "0") == "1":
                    _ramp = int(os.environ.get("Y_TG_RAMP_STEPS", "450"))
                    if global_step == _ramp and dp_rank == 0:
                        mid_dir = os.path.join(args.save, "step450_ckpt")
                        os.makedirs(mid_dir, exist_ok=True)
                        _rw = model.module if hasattr(model, "module") else model
                        print_rank(f"[DDP+Y] Saving midpoint ckpt at step {global_step} -> {mid_dir}")
                        _rw.save_pretrained(mid_dir, safe_serialization=True,
                                            max_shard_size="5GB")
                        z_pt = os.path.join(mid_dir, "z_params.pt")
                        torch.save({
                            "z_mlp": {str(k): v.detach().cpu().clone() for k, v in pruner.mask_params["mlp"].items()},
                            "z_attn": {str(k): v.detach().cpu().clone() for k, v in pruner.mask_params["attn"].items()},
                            "layer_attn_configs": {str(k): v for k, v in pruner.layer_attn_configs.items()},
                            "step": global_step,
                        }, z_pt)
                        print_rank(f"[DDP+Y] Midpoint ckpt saved to {mid_dir}")

                # ==================== 【新增代码开始】 ====================
                if dp_rank == 0 and writer is not None:
                    writer.add_scalar("train/train_loss", info.get("ce_loss", 0), global_step)
                    writer.add_scalar("train/distill_loss", info.get("kd_loss", 0), global_step)
                    writer.add_scalar("train/lag_loss", L_sp.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                    # 分别记录 MLP 和 Attn 的稀疏度和约束误差，这是证明算法有效的核心图表！
                    writer.add_scalar("sparsity/mlp", info.get("sp_mlp", 0), global_step)
                    writer.add_scalar("sparsity/attn", info.get("sp_attn", 0), global_step)

                    writer.add_scalar("constraint/err_mlp", info.get("constraint_mlp", 0), global_step)
                    writer.add_scalar("constraint/err_attn", info.get("constraint_attn", 0), global_step)

                    writer.add_scalar("hyperparams/lam1_mlp", info.get("lam1_mlp", 0), global_step)
                    writer.add_scalar("hyperparams/lam1_attn", info.get("lam1_attn", 0), global_step)
                # ==================== 【新增代码结束】 ====================

                if global_step % getattr(args, "log_interval", 10) == 0:
                    pruner.print_status(global_step, args.total_iters, info)
                    pruner.print_z_diagnostics(global_step, args.total_iters, info["mu"])
                global_step += 1

            step += 1
            if global_step > args.total_iters:
                break
        if global_step > args.total_iters:
            break

    print_rank("\nTraining Completed. Finalizing physical pruning...")
    raw_model = model.module if hasattr(model, "module") else model
    layer_info = pruner.finalize_pruning()

    if dp_rank == 0:
        save_path = os.path.join(args.save, "pruned_final")
        os.makedirs(save_path, exist_ok=True)

        intermediate_sizes = [layer_info[i]["mlp"]["kept"] if "mlp" in layer_info[i] else -1 for i in
                              sorted(layer_info.keys())]
        num_heads_list = []
        num_kv_heads_list = []
        for i in sorted(layer_info.keys()):
            if "attn" in layer_info[i]:
                kept_nkv = layer_info[i]["attn"]["kept"]
                gs = pruner.layer_attn_configs[i][3]
                num_heads_list.append(kept_nkv * gs)
                num_kv_heads_list.append(kept_nkv)
            else:
                num_heads_list.append(-1)
                num_kv_heads_list.append(-1)

        raw_model.config.intermediate_sizes = intermediate_sizes
        raw_model.config.num_attention_heads_per_layer = num_heads_list
        raw_model.config.num_key_value_heads_per_layer = num_kv_heads_list
        raw_model.config.exact_pruning_mask_file = "exact_pruning_masks.json"
        raw_model.config.to_dict().pop("auto_map", None)

        tokenizer.save_pretrained(save_path)
        raw_model.save_pretrained(save_path, safe_serialization=True)
        sorted_layers = sorted(layer_info.keys())
        attn_head_mask_per_layer = []
        mlp_zero_counts_per_layer = []
        for i in sorted_layers:
            attn_info = layer_info[i].get("attn", {})
            num_heads = int(attn_info.get("num_heads_original", 0))
            kept_heads = set(int(h) for h in attn_info.get("kept_head_indices", []))
            attn_head_mask_per_layer.append([1 if h in kept_heads else 0 for h in range(num_heads)])

            mlp_info = layer_info[i].get("mlp", {})
            mlp_zero_counts_per_layer.append(len(mlp_info.get("pruned_indices", [])))

        pruner_metadata = {}
        if hasattr(pruner, "get_exact_mask_metadata"):
            pruner_metadata = pruner.get_exact_mask_metadata() or {}
        exact_mask_source = pruner_metadata.pop("source", "mask_pruner_ddp_plus.finalize_pruning")

        exact_mask_info = {
            "format_version": 1,
            "source": exact_mask_source,
            "model_path": getattr(args, "model_path", None),
            "ckpt_name": getattr(args, "ckpt_name", None),
            "save_path": save_path,
            "target_sparsity": getattr(args, "target_sparsity", None),
            "layers": {str(i): layer_info[i] for i in sorted_layers},
            "attn_head_mask_per_layer": attn_head_mask_per_layer,
            "mlp_zero_counts_per_layer": mlp_zero_counts_per_layer,
        }
        exact_mask_info.update(pruner_metadata)
        exact_mask_path = os.path.join(save_path, "exact_pruning_masks.json")
        with open(exact_mask_path, "w", encoding="utf-8") as f:
            json.dump(exact_mask_info, f, indent=2)
        print_rank(f"Saved to: {save_path}")
        print_rank(f"Saved exact pruning metadata to: {exact_mask_path}")

        # 【新增】训练结束时安全关闭 writer
        if writer is not None:
            writer.close()

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
