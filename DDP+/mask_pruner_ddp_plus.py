# DDP+: standalone Taylor-guided dynamic pruning mask optimizer.
import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist

_C0 = 2.4
_L = -0.1
_R = 1.1

t_sp = float(os.environ.get("TARGET_SPARSITY_FLOAT", 0.50))

def _surrogate(z: torch.Tensor, mu: float) -> torch.Tensor:
    v = torch.sigmoid((z - mu) * (_C0 / mu))
    v_bar = v * (_R - _L) + _L
    s = v_bar + (torch.clamp(v_bar, 0.0, 1.0) - v_bar).detach()
    return s


def _normalize_importance_per_group(I_list, device):
    if not I_list:
        return torch.tensor([], device=device, dtype=torch.float32)
    flat_I = torch.cat(I_list)
    I_min, I_max = flat_I.min(), flat_I.max()
    denom = (I_max - I_min).clamp(min=1e-8)
    return ((flat_I - I_min) / denom).detach()


class DDPPlusMaskPruner:
    def __init__(
            self,
            model,
            target_sparsity: float = t_sp,
            mu_0: float = 0.50,
            mu_T: float = 0.05,
    ):
        self.model = model
        self.target_sparsity = target_sparsity
        self.rho_global = 1.0 - target_sparsity
        self.mu_0 = mu_0
        self.mu_T = mu_T

        self.grad_norm_beta = 0.90

        self.mask_params = {"mlp": {}, "attn": {}}
        self.grad_acc = {"mlp": {}, "attn": {}}
        self.grad_counters = {"mlp": {}, "attn": {}}
        self.ema_importance = {"mlp": {}, "attn": {}}

        self.layer_attn_configs = {}
        self.mask_hooks = []

        self.current_constraint_error = 1.0
        self.ema_mask_grad_norm = None
        self.finalize_topk = os.environ.get("DDP_PLUS_FINALIZE_TOPK", "0").lower() in (
            "1", "true", "yes", "on"
        )

        self._freeze_all_weights()
        self._init_masks_and_lambdas()
        self.is_active = True
        self._register_hooks()

    def _get_layers(self):
        base = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers
        elif hasattr(base, "layers"):
            return base.layers
        raise ValueError("无法定位 transformer layers")

    def _freeze_all_weights(self):
        base = self.model.module if hasattr(self.model, "module") else self.model
        for p in base.parameters():
            p.requires_grad = False

    def _get_attn_config(self, attn):
        q_w, k_w = attn.q_proj.weight, attn.k_proj.weight
        q_out, k_out = q_w.shape[0], k_w.shape[0]
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            for hd in [128, 96, 80, 64, 48, 32]:
                if q_out % hd == 0 and k_out % hd == 0:
                    head_dim = hd
                    break
        if head_dim is None:
            gcd = math.gcd(q_out, k_out)
            head_dim = gcd if gcd >= 32 else (q_out // 64)
        num_heads = q_out // head_dim
        num_kv_heads = k_out // head_dim
        group_size = num_heads // num_kv_heads
        return num_heads, num_kv_heads, head_dim, group_size

    def _init_masks_and_lambdas(self):
        layers = self._get_layers()
        dev = next(self.model.parameters()).device

        # 【修改点 1】: 将拉格朗日乘子分为 mlp 和 attn 两套
        self.lambda1_mlp = nn.Parameter(torch.zeros(1, device=dev, dtype=torch.float32))
        self.lambda2_mlp = nn.Parameter(torch.zeros(1, device=dev, dtype=torch.float32))
        self.lambda3_mlp = nn.Parameter(torch.zeros(1, device=dev, dtype=torch.float32))

        self.lambda1_attn = nn.Parameter(torch.zeros(1, device=dev, dtype=torch.float32))
        self.lambda2_attn = nn.Parameter(torch.zeros(1, device=dev, dtype=torch.float32))
        self.lambda3_attn = nn.Parameter(torch.zeros(1, device=dev, dtype=torch.float32))

        # 新增分别记录误差的变量
        self.current_constraint_error_mlp = 1.0
        self.current_constraint_error_attn = 1.0

        for i, layer in enumerate(layers):
            # ...(保留原有的 Mask 初始化代码不变)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
                C = layer.mlp.gate_proj.weight.shape[0]
                self.mask_params["mlp"][i] = nn.Parameter(torch.ones(C, device=dev, dtype=torch.float32))

            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
                nh, nkv, hd, gs = self._get_attn_config(layer.self_attn)
                self.layer_attn_configs[i] = (nh, nkv, hd, gs)
                self.mask_params["attn"][i] = nn.Parameter(torch.ones(nkv, device=dev, dtype=torch.float32))

    def _make_mask_hook(self, layer_idx, module_type):
        def hook(module, args):
            x = args[0]
            if not getattr(self, "is_active", True):
                return args

            z = self.mask_params[module_type][layer_idx]
            m = torch.relu(z).to(x.dtype)

            if m.requires_grad:
                n_val = max(math.prod(x.shape[:-1]), 1)
                def backward_hook(grad):
                    with torch.no_grad():

                        g_abs = (grad.float().abs() / n_val).detach()
                        g_abs = torch.nan_to_num(g_abs, nan=0.0)

                        if layer_idx not in self.grad_acc[module_type]:
                            self.grad_acc[module_type][layer_idx] = g_abs
                            self.grad_counters[module_type][layer_idx] = 1
                        else:
                            self.grad_acc[module_type][layer_idx] += g_abs
                            self.grad_counters[module_type][layer_idx] += 1

                m.register_hook(backward_hook)

            if module_type == "attn":
                nh, nkv, hd, gs = self.layer_attn_configs[layer_idx]
                x_reshaped = x.view(*x.shape[:-1], nkv, gs * hd)
                m_reshaped = m.view(*([1] * (x.dim() - 1)), nkv, 1)
                return ((x_reshaped * m_reshaped).view_as(x),) + tuple(args[1:])
            else:
                m_reshaped = m.view(*([1] * (x.dim() - 1)), -1)
                return (x * m_reshaped,) + tuple(args[1:])

        return hook

    def _register_hooks(self):
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            if i in self.mask_params["mlp"]:
                target_proj = layer.mlp.down_proj
                h = target_proj.register_forward_pre_hook(self._make_mask_hook(i, "mlp"), with_kwargs=False)
                self.mask_hooks.append(h)
            if i in self.mask_params["attn"]:
                h = layer.self_attn.o_proj.register_forward_pre_hook(self._make_mask_hook(i, "attn"), with_kwargs=False)
                self.mask_hooks.append(h)

    @torch.no_grad()
    def _sync_and_update_importance(self):
        ws = dist.get_world_size() if dist.is_initialized() else 1

        for m_type in ["mlp", "attn"]:
            for i in list(self.grad_acc[m_type].keys()):
                g = self.grad_acc[m_type][i]
                cnt = max(self.grad_counters[m_type][i], 1)
                if ws > 1 and dist.is_initialized():
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)
                g = g / (cnt * ws)

                if i not in self.ema_importance[m_type]:
                    self.ema_importance[m_type][i] = g.clone()
                else:
                    self.ema_importance[m_type][i].mul_(self.grad_norm_beta).add_(g, alpha=1 - self.grad_norm_beta)

                del self.grad_acc[m_type][i]
                self.grad_counters[m_type][i] = 0

    def compute_mask_losses(self, step: int, total_steps: int):
        self._sync_and_update_importance()

        progress = min(1.0, step / max(total_steps, 1))
        mu_t = self.mu_0 - (self.mu_0 - self.mu_T) * math.sqrt(progress)
        mu_t = max(mu_t, self.mu_T)

        dev = next(iter(self.mask_params["mlp"].values())).device

        flat_z_mlp_list, flat_I_mlp_list = [], []
        flat_z_attn_list, flat_I_attn_list = [], []

        # 收集 MLP 和 Attn 的 z 与 重要性 (保持原有收集逻辑)
        for i in sorted(self.mask_params["mlp"].keys()):
            z = self.mask_params["mlp"][i]
            flat_z_mlp_list.append(z.flatten())
            flat_I_mlp_list.append(
                self.ema_importance["mlp"][i].flatten() if i in self.ema_importance["mlp"] else torch.full_like(
                    z.flatten(), 0.5))

        for i in sorted(self.mask_params["attn"].keys()):
            z = self.mask_params["attn"][i]
            flat_z_attn_list.append(z.flatten())
            flat_I_attn_list.append(
                self.ema_importance["attn"][i].flatten() if i in self.ema_importance["attn"] else torch.full_like(
                    z.flatten(), 0.5))

        # 归一化重要性
        norm_I_mlp = _normalize_importance_per_group(flat_I_mlp_list, dev)
        norm_I_attn = _normalize_importance_per_group(flat_I_attn_list, dev)

        # 【修改点 2】：分别为 MLP 和 Attn 计算 Surrogate 和 Constraint
        # --- MLP 部分 ---
        flat_z_mlp = torch.cat(flat_z_mlp_list)
        s_flat_mlp = _surrogate(flat_z_mlp, mu_t)
        constraint_mlp = s_flat_mlp.mean() - self.rho_global
        self.current_constraint_error_mlp = constraint_mlp.item()

        L_sp_mlp = self.lambda1_mlp * constraint_mlp + self.lambda2_mlp * constraint_mlp.pow(2)

        if os.environ.get("DDP_NO_IMPORTANCE", "0") == "1":
            c_k_mlp = torch.full_like(norm_I_mlp, 0.5)
        else:
            c_k_mlp = (1.0 - norm_I_mlp).clamp(0.05, 0.95)
        mask_le_mlp = (s_flat_mlp <= c_k_mlp)
        loss_le_mlp = 1.0 - ((c_k_mlp - s_flat_mlp) / c_k_mlp).pow(2)
        loss_gt_mlp = 1.0 - ((s_flat_mlp - c_k_mlp) / (1.0 - c_k_mlp)).pow(2)
        spline_loss_mlp = torch.where(mask_le_mlp, loss_le_mlp, loss_gt_mlp).mean()
        L_bin_mlp = self.lambda3_mlp * spline_loss_mlp

        # --- Attention 部分 ---
        flat_z_attn = torch.cat(flat_z_attn_list)
        s_flat_attn = _surrogate(flat_z_attn, mu_t)
        constraint_attn = s_flat_attn.mean() - self.rho_global
        self.current_constraint_error_attn = constraint_attn.item()

        L_sp_attn = self.lambda1_attn * constraint_attn + self.lambda2_attn * constraint_attn.pow(2)

        if os.environ.get("DDP_NO_IMPORTANCE", "0") == "1":
            c_k_attn = torch.full_like(norm_I_attn, 0.5)
        else:
            c_k_attn = (1.0 - norm_I_attn).clamp(0.05, 0.95)
        mask_le_attn = (s_flat_attn <= c_k_attn)
        loss_le_attn = 1.0 - ((c_k_attn - s_flat_attn) / c_k_attn).pow(2)
        loss_gt_attn = 1.0 - ((s_flat_attn - c_k_attn) / (1.0 - c_k_attn)).pow(2)
        spline_loss_attn = torch.where(mask_le_attn, loss_le_attn, loss_gt_attn).mean()
        L_bin_attn = self.lambda3_attn * spline_loss_attn

        # --- 总和 ---
        L_sp = L_sp_mlp + L_sp_attn
        L_bin = L_bin_mlp + L_bin_attn

        info = {
            "mu": mu_t,
            "progress": progress,
            "constraint_mlp": self.current_constraint_error_mlp,
            "constraint_attn": self.current_constraint_error_attn,
            "lam1_mlp": self.lambda1_mlp.item(),
            "lam1_attn": self.lambda1_attn.item(),
            "sp_total": self.get_current_sparsity("global"),
            "sp_mlp": self.get_current_sparsity("mlp"),
            "sp_attn": self.get_current_sparsity("attn"),
        }
        return L_sp, L_bin, info

    @torch.no_grad()
    def sync_mask_gradients(self):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        for m_type in ["mlp", "attn"]:
            for z in self.mask_params[m_type].values():
                if z.grad is not None:
                    dist.all_reduce(z.grad.data, op=dist.ReduceOp.AVG)

    # 创新点4: 自适应梯度裁剪
    @torch.no_grad()
    def compute_adaptive_clip_value(self):
        mask_params = self.get_mask_param_list()
        grads = [p.grad.data for p in mask_params if p.grad is not None]
        if not grads:
            return 1.0

        total_norm = torch.norm(
            torch.stack([g.norm(2) for g in grads]), 2
        ).item()

        if self.ema_mask_grad_norm is None:
            self.ema_mask_grad_norm = total_norm
        else:
            self.ema_mask_grad_norm = 0.95 * self.ema_mask_grad_norm + 0.05 * total_norm

        return max(self.ema_mask_grad_norm * 3.0, 1.0)

    # 创新点3: ε-KKT 死区
    @torch.no_grad()
    def zero_dual_grads_if_satisfied(self, tolerance: float = 0.005):
        # 【修改点 3】：按各自的误差情况归零死区梯度
        if abs(self.current_constraint_error_mlp) <= tolerance:
            for p in [self.lambda1_mlp, self.lambda2_mlp, self.lambda3_mlp]:
                if p.grad is not None: p.grad.zero_()

        if abs(self.current_constraint_error_attn) <= tolerance:
            for p in [self.lambda1_attn, self.lambda2_attn, self.lambda3_attn]:
                if p.grad is not None: p.grad.zero_()

    @torch.no_grad()
    def negate_dual_grads(self):
        for p in [self.lambda1_mlp, self.lambda2_mlp, self.lambda3_mlp,
                  self.lambda1_attn, self.lambda2_attn, self.lambda3_attn]:
            if p.grad is not None:
                p.grad.data.neg_()

    @torch.no_grad()
    def clamp_dual_variables(self):
        self.lambda2_mlp.data.clamp_(min=0.0)
        self.lambda3_mlp.data.clamp_(min=0.0)
        self.lambda2_attn.data.clamp_(min=0.0)
        self.lambda3_attn.data.clamp_(min=0.0)

    # 🚨 【修复报错】把丢失的 clamp_z_values 补回来
    @torch.no_grad()
    def clamp_z_values(self):
        for m_type in ["mlp", "attn"]:
            for z in self.mask_params[m_type].values():
                z.data.clamp_(min=-1.0, max=10.0)

    def get_optimizer_param_groups(self):
        z_params = list(self.mask_params["mlp"].values()) + list(self.mask_params["attn"].values())
        lambda2_lr = 1.6 if self.target_sparsity <= 0.25 else 0.8

        # 保持 3 个 group，以便 finetune_mask_ddp_plus.py 里的 scheduler 不报错
        return [
            {"params": z_params, "lr": 2e-2, "weight_decay": 0.0},
            {"params": [self.lambda1_mlp, self.lambda3_mlp, self.lambda1_attn, self.lambda3_attn],
             "lr": 2e-2, "weight_decay": 0.0},
            {"params": [self.lambda2_mlp, self.lambda2_attn],
             "lr": lambda2_lr, "weight_decay": 0.0},
        ]

    def get_mask_param_list(self):
        return list(self.mask_params["mlp"].values()) + list(self.mask_params["attn"].values())

    def get_current_sparsity(self, m_type="global") -> float:
        dead, total = 0, 0
        target_types = ["mlp", "attn"] if m_type == "global" else [m_type]
        for mt in target_types:
            for z in self.mask_params[mt].values():
                dead += (z.detach() <= 0).sum().item()
                total += z.numel()
        return dead / max(total, 1)

    @torch.no_grad()
    def print_z_diagnostics(self, step: int, total_steps: int, mu: float):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        all_z_mlp = []
        all_z_attn = []
        for i in sorted(self.mask_params["mlp"].keys()):
            all_z_mlp.append(self.mask_params["mlp"][i].detach().cpu().float())
        for i in sorted(self.mask_params["attn"].keys()):
            all_z_attn.append(self.mask_params["attn"][i].detach().cpu().float())

        z_mlp = torch.cat(all_z_mlp) if all_z_mlp else torch.tensor([])
        z_attn = torch.cat(all_z_attn) if all_z_attn else torch.tensor([])
        z_all = torch.cat([z_mlp, z_attn])

        def compute_ds_dz(z_vals, mu_val):
            u = (z_vals - mu_val) * (_C0 / mu_val)
            sig = torch.sigmoid(u)
            sig_prime = sig * (1.0 - sig)
            ds_dz = sig_prime * (_C0 / mu_val) * (_R - _L)
            return ds_dz

        ds_dz_all = compute_ds_dz(z_all, mu)
        s_all = _surrogate(z_all, mu)

        pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        z_percentiles = torch.quantile(z_all, torch.tensor([p / 100.0 for p in pcts])) if len(z_all) > 0 else []

        print(f"\n{'─' * 75}")
        print(f"  [Z-DIAG] Step {step}/{total_steps} | μ={mu:.4f} | 2μ={2 * mu:.4f}")
        print(f"{'─' * 75}")

        print(f"  z ALL  (N={z_all.numel():,}): "
              f"mean={z_all.mean():.4f}, std={z_all.std():.4f}, "
              f"min={z_all.min():.4f}, max={z_all.max():.4f}")
        if len(z_percentiles) > 0:
            pct_str = ", ".join([f"p{p}={v:.4f}" for p, v in zip(pcts, z_percentiles)])
            print(f"         percentiles: {pct_str}")

        if len(z_mlp) > 0:
            print(f"  z MLP  (N={z_mlp.numel():,}): "
                  f"mean={z_mlp.mean():.4f}, std={z_mlp.std():.4f}, "
                  f"min={z_mlp.min():.4f}, max={z_mlp.max():.4f}")

        if len(z_attn) > 0:
            print(f"  z ATTN (N={z_attn.numel():,}): "
                  f"mean={z_attn.mean():.4f}, std={z_attn.std():.4f}, "
                  f"min={z_attn.min():.4f}, max={z_attn.max():.4f}")

        bins = [
            ("z < -0.5", z_all < -0.5),
            ("-0.5 ≤ z < 0", (z_all >= -0.5) & (z_all < 0)),
            ("0 ≤ z < 0.3", (z_all >= 0) & (z_all < 0.3)),
            ("0.3 ≤ z < 0.7", (z_all >= 0.3) & (z_all < 0.7)),
            ("0.7 ≤ z < 1.0", (z_all >= 0.7) & (z_all < 1.0)),
            ("1.0 ≤ z < 1.3", (z_all >= 1.0) & (z_all < 1.3)),
            ("z ≥ 1.3", z_all >= 1.3),
        ]
        bin_strs = []
        for label, mask in bins:
            count = mask.sum().item()
            pct = count / max(z_all.numel(), 1) * 100
            bin_strs.append(f"{label}: {count:,} ({pct:.1f}%)")
        print(f"  z bins: {' | '.join(bin_strs[:4])}")
        print(f"          {' | '.join(bin_strs[4:])}")

        print(f"  s ALL: mean={s_all.mean():.4f}, "
              f"s<0.1: {(s_all < 0.1).sum().item():,}, "
              f"s>0.9: {(s_all > 0.9).sum().item():,}, "
              f"0.1≤s≤0.9: {((s_all >= 0.1) & (s_all <= 0.9)).sum().item():,}")

        print(f"  ds/dz ALL: mean={ds_dz_all.mean():.6f}, "
              f"median={ds_dz_all.median():.6f}, "
              f"max={ds_dz_all.max():.6f}")
        print(f"  ds/dz > 0.01: {(ds_dz_all > 0.01).sum().item():,} / {ds_dz_all.numel():,} "
              f"({(ds_dz_all > 0.01).float().mean():.2%})")
        print(f"  ds/dz > 0.1:  {(ds_dz_all > 0.1).sum().item():,} / {ds_dz_all.numel():,} "
              f"({(ds_dz_all > 0.1).float().mean():.2%})")
        print(f"  ds/dz > 1.0:  {(ds_dz_all > 1.0).sum().item():,} / {ds_dz_all.numel():,} "
              f"({(ds_dz_all > 1.0).float().mean():.2%})")

        grad_vals = []
        for mt in ["mlp", "attn"]:
            for z in self.mask_params[mt].values():
                if z.grad is not None:
                    grad_vals.append(z.grad.detach().cpu().float().flatten())

        if grad_vals:
            all_grads = torch.cat(grad_vals)
            print(f"  z.grad: mean={all_grads.mean():.6f}, std={all_grads.std():.6f}, "
                  f"abs_mean={all_grads.abs().mean():.6f}, "
                  f"abs_max={all_grads.abs().max():.6f}")
            print(f"  z.grad > 0 (keep): {(all_grads > 0).sum().item():,} | "
                  f"z.grad < 0 (prune): {(all_grads < 0).sum().item():,}")
        else:
            print(f"  z.grad: not available (already zeroed or not computed)")

        print(f"{'─' * 75}\n")

    def print_status(self, step: int, total_steps: int, info: dict):
        if not (dist.is_initialized() and dist.get_rank() == 0):
            return
        print(f"\n{'=' * 75}")
        print(f"[DDP+] Step {step:5d}/{total_steps} | Prog={info.get('progress', 0):.2%}")
        print(f"  Losses   | CE: {info.get('ce_loss', 0):.4f} | KD: {info.get('kd_loss', 0):.4f}")
        print(f"  Sparsity | Total: {info.get('sp_total', 0):.2%} "
              f"(MLP: {info.get('sp_mlp', 0):.2%} | Attn: {info.get('sp_attn', 0):.2%})")
        print(f"  Err(MLP) | {info.get('constraint_mlp', 0):.4f} | λ1_mlp: {info.get('lam1_mlp', 0):.4f}")
        print(f"  Err(Attn)| {info.get('constraint_attn', 0):.4f} | λ1_attn: {info.get('lam1_attn', 0):.4f}")
        print(f"{'=' * 75}\n")

    @torch.no_grad()
    def _build_exact_topk_keep_indices(self, m_type: str):
        """Build a global top-k active set for final physical pruning."""
        if not self.finalize_topk:
            return None
        if m_type not in self.mask_params or not self.mask_params[m_type]:
            return None

        items = [(i, self.mask_params[m_type][i].detach().float()) for i in sorted(self.mask_params[m_type].keys())]
        total = sum(z.numel() for _, z in items)
        if total <= 0:
            return None

        min_keep = len(items)
        target_keep = int(round(self.rho_global * total))
        target_keep = max(min_keep, min(total, target_keep))

        selected = {}
        candidates = []
        for layer_idx, z in items:
            flat = z.flatten()
            best = int(torch.argmax(flat).item())
            selected[layer_idx] = {best}
            for local_idx, score in enumerate(flat):
                if local_idx != best:
                    candidates.append((float(score.item()), int(layer_idx), int(local_idx)))

        remaining = target_keep - min_keep
        if remaining > 0:
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, layer_idx, local_idx in candidates[:remaining]:
                selected[layer_idx].add(local_idx)

        keep_indices = {}
        for layer_idx, z in items:
            keep_indices[layer_idx] = torch.tensor(
                sorted(selected[layer_idx]), dtype=torch.long, device=z.device
            )

        if not dist.is_initialized() or dist.get_rank() == 0:
            realized = sum(v.numel() for v in keep_indices.values())
            print(
                f"[DDP+] Finalize top-k enabled for {m_type}: "
                f"keep={realized}/{total} sparsity={1.0 - realized / max(total, 1):.4%} "
                f"target={self.target_sparsity:.4%}"
            )
        return keep_indices

    def finalize_pruning(self) -> dict:
        for h in self.mask_hooks:
            h.remove()
        self.mask_hooks.clear()

        layers = self._get_layers()
        layer_keep_info = {}
        final_keep_indices = {
            "mlp": self._build_exact_topk_keep_indices("mlp"),
            "attn": self._build_exact_topk_keep_indices("attn"),
        }

        for i, layer in enumerate(layers):
            info = {}
            if i in self.mask_params["mlp"]:
                z = self.mask_params["mlp"][i].detach()
                m = torch.relu(z)
                if final_keep_indices["mlp"] is not None and i in final_keep_indices["mlp"]:
                    keep_idx = final_keep_indices["mlp"][i]
                else:
                    keep_idx = torch.where(m > 0.01)[0]
                if len(keep_idx) == 0:
                    keep_idx = torch.tensor([z.argmax().item()], device=z.device)

                mask_vals = m[keep_idx]
                C_orig, C_keep = z.numel(), keep_idx.numel()

                mlp = layer.mlp
                dev, dt = mlp.up_proj.weight.device, mlp.up_proj.weight.dtype
                hidden_size = mlp.up_proj.weight.shape[1]

                new_up = torch.index_select(mlp.up_proj.weight.data, 0, keep_idx).clone().contiguous()
                new_up = new_up * mask_vals.unsqueeze(1).to(dt)
                has_up_bias = mlp.up_proj.bias is not None
                up_bias = mlp.up_proj.bias.data[keep_idx].clone() * mask_vals.to(dt) if has_up_bias else None

                new_down = torch.index_select(mlp.down_proj.weight.data, 1, keep_idx).clone().contiguous()
                has_down_bias = mlp.down_proj.bias is not None
                down_bias = mlp.down_proj.bias.data.clone() if has_down_bias else None

                mlp.up_proj = nn.Linear(hidden_size, C_keep, bias=has_up_bias, device=dev, dtype=dt)
                mlp.up_proj.weight.data.copy_(new_up)
                if has_up_bias:
                    mlp.up_proj.bias.data.copy_(up_bias)

                mlp.down_proj = nn.Linear(C_keep, hidden_size, bias=has_down_bias, device=dev, dtype=dt)
                mlp.down_proj.weight.data.copy_(new_down)
                if has_down_bias:
                    mlp.down_proj.bias.data.copy_(down_bias)

                if hasattr(mlp, 'gate_proj'):
                    new_gate = torch.index_select(mlp.gate_proj.weight.data, 0, keep_idx).clone().contiguous()
                    has_gate_bias = mlp.gate_proj.bias is not None
                    gate_bias = mlp.gate_proj.bias.data[keep_idx].clone() if has_gate_bias else None
                    mlp.gate_proj = nn.Linear(hidden_size, C_keep, bias=has_gate_bias, device=dev, dtype=dt)
                    mlp.gate_proj.weight.data.copy_(new_gate)
                    if has_gate_bias:
                        mlp.gate_proj.bias.data.copy_(gate_bias)

                keep_mask = torch.zeros(C_orig, dtype=torch.bool, device=z.device)
                keep_mask[keep_idx] = True
                pruned_idx = torch.arange(C_orig, device=z.device)[~keep_mask]

                info["mlp"] = {
                    "original": int(C_orig),
                    "kept": int(C_keep),
                    "sparsity": float(1.0 - C_keep / C_orig),
                    "kept_indices": keep_idx.detach().cpu().long().tolist(),
                    "pruned_indices": pruned_idx.detach().cpu().long().tolist(),
                }

            if i in self.mask_params["attn"]:
                z = self.mask_params["attn"][i].detach()
                m = torch.relu(z)
                if final_keep_indices["attn"] is not None and i in final_keep_indices["attn"]:
                    keep_groups = final_keep_indices["attn"][i]
                else:
                    keep_groups = torch.where(m > 0.01)[0]
                if len(keep_groups) == 0:
                    keep_groups = torch.tensor([z.argmax().item()], device=z.device)

                mask_vals = m[keep_groups]
                C_orig, C_keep = z.numel(), keep_groups.numel()

                attn = layer.self_attn
                dev, dt = attn.q_proj.weight.device, attn.q_proj.weight.dtype
                nh, nkv, hd, gs = self.layer_attn_configs[i]

                new_num_kv_heads, new_num_heads = C_keep, C_keep * gs
                group_q_dim = gs * hd

                q_w = attn.q_proj.weight.data
                k_w = attn.k_proj.weight.data
                v_w = attn.v_proj.weight.data
                o_w = attn.o_proj.weight.data
                q_out, hidden_in = q_w.shape[0], q_w.shape[1]

                new_q = torch.index_select(
                    q_w.view(nkv, group_q_dim, hidden_in), 0, keep_groups
                ).contiguous().view(new_num_heads * hd, hidden_in)
                has_q_bias = attn.q_proj.bias is not None
                q_bias = torch.index_select(
                    attn.q_proj.bias.data.view(nkv, group_q_dim), 0, keep_groups
                ).contiguous().view(-1) if has_q_bias else None

                new_k = torch.index_select(
                    k_w.view(nkv, hd, hidden_in), 0, keep_groups
                ).contiguous().view(new_num_kv_heads * hd, hidden_in)
                has_k_bias = attn.k_proj.bias is not None
                k_bias = torch.index_select(
                    attn.k_proj.bias.data.view(nkv, hd), 0, keep_groups
                ).contiguous().view(-1) if has_k_bias else None

                new_v = torch.index_select(
                    v_w.view(nkv, hd, hidden_in), 0, keep_groups
                ).contiguous().view(new_num_kv_heads * hd, hidden_in)
                has_v_bias = attn.v_proj.bias is not None
                v_bias = torch.index_select(
                    attn.v_proj.bias.data.view(nkv, hd), 0, keep_groups
                ).contiguous().view(-1) if has_v_bias else None

                has_o_bias = attn.o_proj.bias is not None
                o_bias = attn.o_proj.bias.data.clone() if has_o_bias else None

                mask_expanded = mask_vals.view(-1, 1).repeat(1, group_q_dim).view(-1).to(dt)
                if o_w.shape[1] == q_out:
                    hidden_out = o_w.shape[0]
                    new_o = torch.index_select(
                        o_w.view(hidden_out, nkv, group_q_dim), 1, keep_groups
                    ).contiguous().view(hidden_out, new_num_heads * hd)
                    new_o = new_o * mask_expanded.unsqueeze(0)
                    o_in, o_out = new_num_heads * hd, hidden_out
                else:
                    hidden_out = o_w.shape[1]
                    new_o = torch.index_select(
                        o_w.view(nkv, group_q_dim, hidden_out), 0, keep_groups
                    ).contiguous().view(new_num_heads * hd, hidden_out)
                    new_o = new_o * mask_expanded.unsqueeze(1)
                    o_in, o_out = hidden_out, new_num_heads * hd

                attn.q_proj = nn.Linear(hidden_in, new_num_heads * hd, bias=has_q_bias, device=dev, dtype=dt)
                attn.q_proj.weight.data.copy_(new_q)
                if has_q_bias:
                    attn.q_proj.bias.data.copy_(q_bias)

                attn.k_proj = nn.Linear(hidden_in, new_num_kv_heads * hd, bias=has_k_bias, device=dev, dtype=dt)
                attn.k_proj.weight.data.copy_(new_k)
                if has_k_bias:
                    attn.k_proj.bias.data.copy_(k_bias)

                attn.v_proj = nn.Linear(hidden_in, new_num_kv_heads * hd, bias=has_v_bias, device=dev, dtype=dt)
                attn.v_proj.weight.data.copy_(new_v)
                if has_v_bias:
                    attn.v_proj.bias.data.copy_(v_bias)

                attn.o_proj = nn.Linear(o_in, o_out, bias=has_o_bias, device=dev, dtype=dt)
                attn.o_proj.weight.data.copy_(new_o)
                if has_o_bias:
                    attn.o_proj.bias.data.copy_(o_bias)

                if hasattr(attn, 'num_heads'):
                    attn.num_heads = new_num_heads
                if hasattr(attn, 'num_key_value_heads'):
                    attn.num_key_value_heads = new_num_kv_heads
                if hasattr(attn, 'num_key_value_groups'):
                    attn.num_key_value_groups = gs
                if hasattr(attn, 'hidden_size'):
                    attn.hidden_size = new_num_heads * hd

                keep_group_mask = torch.zeros(C_orig, dtype=torch.bool, device=z.device)
                keep_group_mask[keep_groups] = True
                pruned_groups = torch.arange(C_orig, device=z.device)[~keep_group_mask]

                kept_group_indices = keep_groups.detach().cpu().long().tolist()
                pruned_group_indices = pruned_groups.detach().cpu().long().tolist()
                kept_head_indices = [
                    int(head_idx)
                    for group_idx in kept_group_indices
                    for head_idx in range(int(group_idx) * int(gs), (int(group_idx) + 1) * int(gs))
                ]
                pruned_head_indices = [
                    int(head_idx)
                    for group_idx in pruned_group_indices
                    for head_idx in range(int(group_idx) * int(gs), (int(group_idx) + 1) * int(gs))
                ]

                info["attn"] = {
                    "original": int(C_orig),
                    "kept": int(C_keep),
                    "sparsity": float(1.0 - C_keep / C_orig),
                    "group_size": int(gs),
                    "num_heads_original": int(nh),
                    "num_key_value_heads_original": int(nkv),
                    "num_heads_kept": int(new_num_heads),
                    "num_key_value_heads_kept": int(new_num_kv_heads),
                    "kept_group_indices": kept_group_indices,
                    "pruned_group_indices": pruned_group_indices,
                    "kept_head_indices": kept_head_indices,
                    "pruned_head_indices": pruned_head_indices,
                }
            layer_keep_info[i] = info
        return layer_keep_info


# ---------------------------------------------------------------------------
# DDP+ extensions
# ---------------------------------------------------------------------------
# This file intentionally contains the complete pruner implementation.  The
# code above is the standalone DDP+ mask/loss/finalize engine; the methods
# The integrated implementation below contains the complete DDP+ behavior.

DDPPlusMaskPruner.method_version = "ddp_plus"
DDPPlusMaskPruner.importance_score_type = "mean_abs_activation_times_gradient"
DDPPlusMaskPruner.importance_reduction = (
    "mean over batch/tokens of sum(abs(activation * grad_output))"
)

_standalone_init = DDPPlusMaskPruner.__init__
_standalone_compute_mask_losses = DDPPlusMaskPruner.compute_mask_losses
_standalone_finalize_pruning = DDPPlusMaskPruner.finalize_pruning


def _v3_init(self, *args, **kwargs):
    _standalone_init(self, *args, **kwargs)
    self.last_importance_stats = {}
    self._current_global_step = 0
    self._current_total_steps = 1


def _v3_current_mu_t(self):
    progress = min(
        1.0,
        max(int(getattr(self, "_current_global_step", 0)), 0)
        / max(int(getattr(self, "_current_total_steps", 1)), 1),
    )
    mu = self.mu_0 - (self.mu_0 - self.mu_T) * math.sqrt(progress)
    return float(max(mu, self.mu_T))


@torch.no_grad()
def _v3_accumulate_taylor_importance(self, layer_idx, module_type, score):
    score = torch.nan_to_num(score.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(score).all():
        raise RuntimeError(
            f"DDP+ non-finite Taylor importance: type={module_type}, layer={layer_idx}"
        )
    # Taylor scores are used only for within-group ranking.  Normalize each
    # observation by a robust scale so a transient activation/gradient spike
    # cannot overflow the EMA and subsequently corrupt z.
    positive = score[score > 0]
    if positive.numel():
        scale = torch.quantile(positive, 0.99).clamp_min(1e-8)
        score = (score / scale).clamp_(0.0, 10.0)
    if layer_idx not in self.grad_acc[module_type]:
        self.grad_acc[module_type][layer_idx] = score
        self.grad_counters[module_type][layer_idx] = 1
    else:
        self.grad_acc[module_type][layer_idx] += score
        self.grad_counters[module_type][layer_idx] += 1


def _v3_make_mask_hook(self, layer_idx, module_type):
    def hook(module, args):
        x = args[0]
        if not getattr(self, "is_active", True):
            return args

        z = self.mask_params[module_type][layer_idx]
        m = _surrogate(z, _v3_current_mu_t(self)).to(x.dtype)
        n_val = max(math.prod(x.shape[:-1]), 1)

        if module_type == "attn":
            _, nkv, hd, group_size = self.layer_attn_configs[layer_idx]
            x_group = x.reshape(*x.shape[:-1], nkv, group_size * hd)
            m_group = m.view(*([1] * (x.dim() - 1)), nkv, 1)
            masked = (x_group * m_group).reshape_as(x)

            if masked.requires_grad:
                def backward_hook(grad):
                    with torch.no_grad():
                        xg = x.detach().float().reshape(*x.shape[:-1], nkv, group_size * hd)
                        gg = grad.detach().float().reshape(*grad.shape[:-1], nkv, group_size * hd)
                        score = (xg * gg).abs().sum(dim=-1)
                        reduce_dims = tuple(range(score.dim() - 1))
                        score = score.sum(dim=reduce_dims) / n_val
                        _v3_accumulate_taylor_importance(self, layer_idx, module_type, score)

                masked.register_hook(backward_hook)
            return (masked,) + tuple(args[1:])

        masked = x * m.view(*([1] * (x.dim() - 1)), -1)
        if masked.requires_grad:
            def backward_hook(grad):
                with torch.no_grad():
                    score = (x.detach().float() * grad.detach().float()).abs()
                    reduce_dims = tuple(range(score.dim() - 1))
                    score = score.sum(dim=reduce_dims) / n_val
                    _v3_accumulate_taylor_importance(self, layer_idx, module_type, score)

            masked.register_hook(backward_hook)
        return (masked,) + tuple(args[1:])

    return hook


@torch.no_grad()
def _v3_collect_importance_stats(self):
    stats = {"finite": True}
    for module_type in ("mlp", "attn"):
        values = [
            value.detach().float().flatten()
            for value in self.ema_importance[module_type].values()
            if value is not None and value.numel() > 0
        ]
        if not values:
            stats[f"{module_type}_count"] = 0
            continue
        flat = torch.cat(values)
        finite = torch.isfinite(flat)
        stats["finite"] = bool(stats["finite"] and finite.all().item())
        flat = flat[finite]
        stats[f"{module_type}_count"] = int(flat.numel())
        if flat.numel():
            stats[f"{module_type}_mean"] = float(flat.mean().item())
            stats[f"{module_type}_median"] = float(flat.median().item())
            stats[f"{module_type}_q95"] = float(torch.quantile(flat, 0.95).item())
            stats[f"{module_type}_max"] = float(flat.max().item())
    self.last_importance_stats = stats
    return stats


def _v3_compute_mask_losses(self, step: int, total_steps: int):
    self._current_global_step = int(step)
    self._current_total_steps = int(total_steps)
    losses = _standalone_compute_mask_losses(self, step=step, total_steps=total_steps)
    stats = _v3_collect_importance_stats(self)
    if not stats.get("finite", True):
        raise RuntimeError("DDP+ non-finite EMA Taylor importance detected")
    l_sp, l_bin, info = losses
    info.update({f"v3_{key}": value for key, value in stats.items()})
    info["v3_mu_t_used_in_forward"] = _v3_current_mu_t(self)
    return l_sp, l_bin, info


def _v3_finalize_pruning(self):
    self._current_global_step = self._current_total_steps
    original_relu = torch.relu

    def phi_as_relu(z):
        return _surrogate(z, self.mu_T)

    torch.relu = phi_as_relu
    try:
        return _standalone_finalize_pruning(self)
    finally:
        torch.relu = original_relu


def _v3_print_status(self, step: int, total_steps: int, info: dict):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    stats = self.last_importance_stats or {}
    print(
        "[DDP+] Taylor importance "
        f"finite={stats.get('finite', True)} | "
        f"MLP count={stats.get('mlp_count', 0)} mean={stats.get('mlp_mean', 0.0):.6e} | "
        f"Attn count={stats.get('attn_count', 0)} mean={stats.get('attn_mean', 0.0):.6e}"
    )
    print(f"[DDP+] mu_t in forward = {_v3_current_mu_t(self):.4f}")


def _v3_metadata(self):
    return {
        "source": "mask_pruner_ddp_plus.standalone",
        "method_version": self.method_version,
        "importance_score_type": self.importance_score_type,
        "importance_reduction": self.importance_reduction,
        "importance_scope": {
            "mlp": "input channels to each decoder MLP down_proj",
            "attn": "input groups to each decoder attention o_proj, grouped by KV head",
        },
        "ema_beta": self.grad_norm_beta,
        "importance_finite_check": True,
        "last_importance_stats": self.last_importance_stats,
        "mask_function_forward": "phi(z; mu_t), asymmetric spline",
        "mask_function_finalize": "phi(z; mu_T), asymmetric spline",
    }


DDPPlusMaskPruner.__init__ = _v3_init
DDPPlusMaskPruner._current_mu_t = _v3_current_mu_t
DDPPlusMaskPruner._make_mask_hook = _v3_make_mask_hook
DDPPlusMaskPruner.compute_mask_losses = _v3_compute_mask_losses
DDPPlusMaskPruner.finalize_pruning = _v3_finalize_pruning
DDPPlusMaskPruner.print_status = _v3_print_status
DDPPlusMaskPruner.get_exact_mask_metadata = _v3_metadata


__all__ = ["DDPPlusMaskPruner", "_surrogate"]
