# DDP+

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DDP+ is a standalone implementation of Taylor-guided dynamic structured pruning for LLaMA-family causal language models. The method learns structured masks over MLP channels and attention heads while keeping pretrained weights frozen.

## Overview

DDP+ combines two components:

1. Taylor importance estimation: an exponential moving average of activation-gradient scores prioritizes influential channels and attention heads.
2. Asymmetric spline binarization: a differentiable surrogate drives masks toward exact binary decisions while preserving the forward/finalize consistency required for structured pruning.

The public implementation is self-contained. `ddp_plus/pruner.py` contains the complete DDP+ algorithm and does not import, inherit from, or invoke a V2 implementation.

## Repository Layout

```text
ddp_plus/                  # DDP+ algorithm, model utilities, and dataset readers
  pruner.py                # Taylor importance, spline masks, and exact finalization
  data_utils/              # Indexed-language-model dataset readers
train.py                    # Distributed DDP+ training entry point
evaluate.py                 # lm-evaluation-harness evaluation entry point
scripts/train_llama2.sh     # LLaMA2 torchrun launcher
requirements.txt            # Runtime dependencies
```

## Installation

```bash
conda create -n ddpplus python=3.10 -y
conda activate ddpplus
pip install -r requirements.txt
```

## Data Preparation

The training entry point expects an indexed language-model dataset directory containing the `train` split used by `LMTrainDataset`. Model weights and datasets are intentionally excluded from this repository.

## Training LLaMA2

Prepare the pretrained model and indexed FineWeb data locally, then run:

```bash
export MODEL_PATH=/path/to/Llama-2-7b-chat-hf
export DATA_DIR=/path/to/processed_data/fineweb/full
bash scripts/train_llama2.sh /path/to/DDP-plus 29500 4 50% 1000
```

Default LLaMA2 configuration: sequence length 1024, per-GPU batch size 2, gradient accumulation 4, learning rate 0.02, mask learning rate 0.02, 50 warmup steps, and 1000 optimization steps.

The launcher accepts the following positional arguments:

```text
bash scripts/train_llama2.sh BASE_PATH MASTER_PORT GPUS_PER_NODE SPARSITY TOTAL_ITERS
```

For example, the command above launches 50% structured pruning for 1000 steps on four GPUs. Use `20%` to run the 20% setting. The output checkpoint includes the physically pruned model and exact mask metadata.

## Evaluation

`evaluate.py` evaluates a saved `pruned_final` checkpoint with lm-evaluation-harness. A local Hugging Face dataset cache can be used for offline execution.

```bash
python evaluate.py \
  --model-path /path/to/pruned_final \
  --batch-size 1 \
  --tasks wikitext,boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
```

## Reproducibility Defaults

| Setting | Value |
| --- | ---: |
| Sequence length | 1024 |
| Per-GPU batch size | 2 |
| Gradient accumulation | 4 |
| Learning rate | 0.02 |
| Mask learning rate | 0.02 |
| Warmup steps | 50 |
| Training steps | 1000 |

## License

This project is distributed under the [MIT License](LICENSE). It includes and modifies components originally released under the MIT License by Microsoft LMOps.

No pretrained weights, datasets, checkpoints, caches, logs, or experiment outputs are distributed with this repository.
