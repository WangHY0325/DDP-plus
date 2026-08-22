#!/usr/bin/env python3
import os
import sys
import json
import gc
import time
import argparse
import dataclasses
import math
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file as safe_load_file

# 强制离线环境
HF_HOME_DIR = os.environ.get("HF_HOME_DIR", "/root/autodl-tmp/AAAI-prune/hf_cache")
os.environ["HF_HOME"] = HF_HOME_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME_DIR, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME_DIR, "hub")
_eval_offline = os.environ.get("EVAL_OFFLINE", "0")
os.environ["HF_DATASETS_OFFLINE"] = _eval_offline
os.environ["HF_HUB_OFFLINE"] = _eval_offline
os.environ["HF_EVALUATE_OFFLINE"] = _eval_offline
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "10"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "10"

# 引入 lm-eval 核心库
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table


def make_json_safe(obj):
    if dataclasses.is_dataclass(obj):
        return make_json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if callable(obj):
        return None
    return obj


def assert_finite_tree(obj, path="results"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_finite_tree(value, f"{path}.{key}")
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            assert_finite_tree(value, f"{path}[{idx}]")
        return
    if isinstance(obj, float) and not math.isfinite(obj):
        raise FloatingPointError(f"Non-finite value at {path}: {obj}")
    if isinstance(obj, (np.floating,)) and not math.isfinite(float(obj)):
        raise FloatingPointError(f"Non-finite value at {path}: {obj}")


def dtype_from_name(name):
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported eval dtype: {name}")


@torch.no_grad()
def run_finite_logits_smoke(model, tokenizer, samples=16, max_length=64):
    if samples <= 0:
        return {"enabled": False, "samples": 0}
    device = next(model.parameters()).device
    prompts = [
        "The capital of France is",
        "A reliable pruning result should report",
        "Question: Is water wet? Answer:",
        "The quick brown fox",
    ]
    checked = 0
    max_abs = 0.0
    model.eval()
    for idx in range(samples):
        batch = tokenizer(prompts[idx % len(prompts)], return_tensors="pt", truncation=True, max_length=max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch, use_cache=False)
        logits = outputs.logits
        if not torch.isfinite(logits).all():
            bad = torch.isfinite(logits).logical_not().sum().item()
            raise FloatingPointError(f"Finite logits smoke failed at sample {idx}: {bad} non-finite logits")
        max_abs = max(max_abs, float(logits.detach().abs().max().float().cpu()))
        checked += int(batch["input_ids"].numel())
    print(f"[Smoke] finite logits passed: samples={samples} checked_tokens={checked} max_abs_logit={max_abs:.6f}")
    return {"enabled": True, "samples": samples, "checked_tokens": checked, "max_abs_logit": max_abs}


def measure_efficiency(model, tokenizer):
    """测量模型的静态显存、峰值显存和基础生成吞吐量"""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return {"error": "效率测评仅在 CUDA 环境下有效"}

    stats = {}

    # ==================== [新增] 统计真实物理参数量 ====================
    total_params = sum(p.numel() for p in model.parameters())
    stats["total_parameters"] = total_params
    print(f"  [效率测试] 模型真实物理参数量: {total_params:,} (约 {total_params / 1e9:.4f} B)")
    # ==================================================================

    torch.cuda.synchronize()
    # 静态显存 (仅包含模型权重)
    static_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
    stats["static_gpu_memory_GB"] = round(static_mem, 3)

    print("  [效率测试] 正在进行吞吐量预热...")
    input_len = 128
    output_len = 128
    test_steps = 5

    dummy_input = torch.randint(10, 10000, (1, input_len), device=device)

    # 预热 1 次
    with torch.no_grad():
        model.generate(dummy_input, max_new_tokens=output_len, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    print(f"  [效率测试] 正在测试生成速度 ({test_steps} 次)...")
    start_time = time.perf_counter()
    total_generated_tokens = 0

    # 正式计时生成
    for _ in range(test_steps):
        with torch.no_grad():
            out = model.generate(dummy_input, max_new_tokens=output_len, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            total_generated_tokens += (out.shape[1] - input_len)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 峰值显存 (含生成时的 KV Cache 和计算图开销)
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    stats["peak_gpu_memory_GB"] = round(peak_mem, 3)

    duration = end_time - start_time
    throughput = total_generated_tokens / duration
    stats["throughput_tokens_per_sec"] = round(throughput, 2)
    stats["avg_latency_per_token_ms"] = round((duration / total_generated_tokens) * 1000, 2)

    print(f"  📊 静态显存: {stats['static_gpu_memory_GB']} GB | 峰值显存: {stats['peak_gpu_memory_GB']} GB")
    print(
        f"  ⚡ 吞吐量: {stats['throughput_tokens_per_sec']} tokens/s | 单 Token 延迟: {stats['avg_latency_per_token_ms']} ms")

    # 清理测试残余，重置显存峰值记录器
    del dummy_input, out
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return stats


def load_model_for_eval(model_path, dtype=torch.bfloat16):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    intermediate_sizes = cfg.get("intermediate_sizes")
    heads_per_layer = cfg.get("num_attention_heads_per_layer")
    kv_heads_per_layer = cfg.get("num_key_value_heads_per_layer")

    is_nonuniform = (intermediate_sizes is not None) or (heads_per_layer is not None)

    if not is_nonuniform:
        print("  [加载] 标准模型, 直接 from_pretrained")
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        ).eval()

    print("  [加载] 非均匀剪枝模型, 两阶段加载")
    has_mlp_prune = intermediate_sizes is not None
    has_attn_prune = heads_per_layer is not None

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cpu",
        trust_remote_code=True, ignore_mismatched_sizes=True
    )

    sf_single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(sf_single):
        state_dict = safe_load_file(sf_single)
    else:
        sf_files = sorted(f for f in os.listdir(model_path) if f.startswith("model") and f.endswith(".safetensors"))
        state_dict = {}
        for sf in sf_files:
            state_dict.update(safe_load_file(os.path.join(model_path, sf)))

    text_cfg = cfg.get("text_config", cfg)
    hidden_size = text_cfg.get("hidden_size", cfg.get("hidden_size"))

    import re as _re
    prefix_candidates = [k for k in state_dict if _re.search(r'\.\d+\.(mlp|self_attn)\.', k)]
    m = _re.match(r'(.+)\.\d+\.(mlp|self_attn)\.', prefix_candidates[0])
    base = m.group(1)

    if has_attn_prune:
        first_valid_idx = next(i for i, h in enumerate(heads_per_layer) if h > 0)
        first_nh = heads_per_layer[first_valid_idx]
        q_w_key = f"{base}.{first_valid_idx}.self_attn.q_proj.weight"
        if q_w_key in state_dict:
            q_out_features = state_dict[q_w_key].shape[0]
            head_dim = q_out_features // first_nh
        else:
            orig_num_heads = text_cfg.get("num_attention_heads", cfg.get("num_attention_heads", 32))
            head_dim = hidden_size // orig_num_heads
    else:
        orig_num_heads = text_cfg.get("num_attention_heads", cfg.get("num_attention_heads", 32))
        head_dim = hidden_size // orig_num_heads

    def _inject_linear(module, attr_name, state_dict, key_prefix, dtype):
        w_key = f"{key_prefix}.weight"
        b_key = f"{key_prefix}.bias"
        if w_key not in state_dict: return False
        w = state_dict[w_key]
        has_bias = b_key in state_dict
        new_linear = nn.Linear(w.shape[1], w.shape[0], bias=has_bias, dtype=dtype)
        new_linear.weight.data.copy_(w)
        if has_bias: new_linear.bias.data.copy_(state_dict[b_key])
        setattr(module, attr_name, new_linear)
        return True

    for layer_idx, layer in enumerate(model.model.layers):
        if has_mlp_prune and intermediate_sizes[layer_idx] != -1:
            mlp_prefix = f"{base}.{layer_idx}.mlp"
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                _inject_linear(layer.mlp, proj, state_dict, f"{mlp_prefix}.{proj}", dtype)

        if has_attn_prune and layer_idx < len(heads_per_layer) and heads_per_layer[layer_idx] != -1:
            new_nh = heads_per_layer[layer_idx]
            new_nkv = kv_heads_per_layer[layer_idx]
            attn = layer.self_attn
            attn_prefix = f"{base}.{layer_idx}.self_attn"
            injected_any = False
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if _inject_linear(attn, proj, state_dict, f"{attn_prefix}.{proj}", dtype):
                    injected_any = True
            if injected_any:
                if hasattr(attn, 'num_heads'): attn.num_heads = new_nh
                if hasattr(attn, 'num_key_value_heads'): attn.num_key_value_heads = new_nkv
                if hasattr(attn, 'num_key_value_groups'): attn.num_key_value_groups = new_nh // max(new_nkv, 1)
                if hasattr(attn, 'head_dim'): attn.head_dim = head_dim

    del state_dict
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH"),
        help="Path to the checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        default=os.environ.get(
            "EVAL_TASKS",
            "wikitext,boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
        ),
        help="Comma-separated lm-eval task list.",
    )
    parser.add_argument("--num-fewshot", type=int, default=int(os.environ.get("NUM_FEWSHOT", "0")))
    parser.add_argument("--batch-size", default=os.environ.get("EVAL_BATCH_SIZE", "auto"))
    parser.add_argument("--output-name", default=os.environ.get("EVAL_OUTPUT_NAME", "ddp_comparison_results.json"))
    parser.add_argument("--eval-dtype", default=os.environ.get("EVAL_DTYPE", "bf16"))
    parser.add_argument("--smoke-samples", type=int, default=int(os.environ.get("SMOKE_SAMPLES", "16")))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--allow-nonfinite-results", action="store_true")
    args = parser.parse_args()
    if not args.model_path:
        raise ValueError("Please pass --model-path or set MODEL_PATH.")
    return args


def main():
    args = parse_args()
    model_path = args.model_path
    dtype = dtype_from_name(args.eval_dtype)

    print(f"[1] 正在加载 Tokenizer与模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model_for_eval(model_path, dtype=dtype)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    smoke_stats = run_finite_logits_smoke(
        model,
        tokenizer,
        samples=0 if args.skip_smoke else args.smoke_samples,
    )

    print("[2] 执行物理效率测评 (内存与吞吐量)...")
    efficiency_stats = measure_efficiency(model, tokenizer)

    print("[3] 初始化 lm-evaluation-harness 包装器...")
    # 改为 auto 动态捕获最大可用显存限制，彻底规避 OOM
    lm_model = HFLM(pretrained=model, backend="causal", tokenizer=tokenizer, batch_size=args.batch_size)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    print(f"[4] 开始执行 Zero-shot 测评任务: {tasks}")

    # 批处理也设置为 auto，启动探测机制
    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        confirm_run_unsafe_code=True
    )

    # 注入刚才测得的效率指标
    if "efficiency_stats" not in results:
        results["efficiency_stats"] = efficiency_stats
    results["eval_manifest"] = {
        "method_type": args.method_type,
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "prune_model_path": args.prune_model_path,
        "peft_path": args.peft_path,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "eval_dtype": args.eval_dtype,
        "output_name": args.output_name,
        "smoke": smoke_stats,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }

    print("\n" + "=" * 80)
    print("📊 lm-evaluation-harness 测评结果")
    print("=" * 80)
    print(make_table(results))

    out_dir = os.path.join(model_path, "lm_eval_results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, args.output_name)
    safe_results = make_json_safe(results)
    if not args.allow_nonfinite_results:
        assert_finite_tree(safe_results)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(safe_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细评测结果与效率指标已保存至: {out_file}")


if __name__ == "__main__":
    main()
