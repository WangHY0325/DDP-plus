# DDP+

DDP+ is a standalone implementation of Taylor-guided dynamic structured pruning for LLaMA-family causal language models. It learns masks over MLP channels and attention heads while keeping the base weights frozen.

## Method

DDP+ combines two components:

1. Taylor importance estimation: an exponential moving average of activation-gradient scores prioritizes influential channels and attention heads.
2. Asymmetric spline binarization: a differentiable surrogate drives masks toward exact binary decisions while preserving the forward/finalize consistency required for structured pruning.

The public implementation is self-contained. `ddp_plus/pruner.py` includes the complete DDP+ algorithm and does not import, inherit from, or invoke a V2 implementation.

## Repository layout

```text
ddp_plus/             # Algorithm, distributed utilities, and dataset readers
train.py               # DDP+ training entry point
evaluate.py            # lm-evaluation-harness entry point
scripts/train_llama2.sh
requirements.txt
```

## Installation

```bash
conda create -n ddpplus python=3.10 -y
conda activate ddpplus
pip install -r requirements.txt
```

## LLaMA2 training

Prepare the pretrained model and indexed FineWeb data locally, then run:

```bash
export MODEL_PATH=/path/to/Llama-2-7b-chat-hf
export DATA_DIR=/path/to/processed_data/fineweb/full
bash scripts/train_llama2.sh /path/to/DDP-plus 29500 4 50% 1000
```

Default LLaMA2 configuration: sequence length 1024, per-GPU batch size 2, gradient accumulation 4, learning rate 0.02, mask learning rate 0.02, 50 warmup steps, and 1000 optimization steps.

## Evaluation

`evaluate.py` supports local lm-evaluation-harness runs. Install the evaluation datasets or provide a Hugging Face cache before launching evaluation.

No pretrained weights, datasets, checkpoints, caches, logs, or experiment outputs are distributed with this repository.
