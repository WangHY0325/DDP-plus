# DDP+

Standalone implementation of DDP+ for structured pruning of LLaMA-family causal language models.

## Components

- `finetune_ddp_plus.py`: distributed training entry point.
- `mask_pruner_ddp_plus.py`: integrated DDP+ mask optimizer. It contains the Taylor EMA importance path, asymmetric surrogate binarization, sparsity constraints, and final mask materialization. It does not import or call DDP+ V2.
- `arguments.py`, `utils.py`: model, tokenizer, distributed initialization, and argument handling.
- `data_utils/`: FineWeb/indexed dataset readers.
- `scripts/llama2/sft/train_ddp_plus.sh`: direct `torchrun` launcher.
- `run_lm_eval.py`: local/offline lm-evaluation harness entry point.

## Example

```bash
bash scripts/llama2/sft/train_ddp_plus.sh \
  /path/to/DDP+ 29500 4 50% 1000
```

The launcher defaults to per-GPU batch size 2, gradient accumulation 4, sequence length 1024, learning rate 0.02, mask learning rate 0.02, warmup 50 steps, and 1000 iterations. Set `MODEL_PATH`, `DATA_DIR`, and `SAVE_TAG_SUFFIX` as needed.

## Requirements

PyTorch, Transformers, DeepSpeed, Accelerate, PEFT, NumPy, TensorBoard, safetensors, datasets, and lm-evaluation-harness.

No model weights, datasets, caches, checkpoints, or experiment logs are included.
