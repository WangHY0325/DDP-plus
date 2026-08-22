#!/bin/bash
set -euo pipefail

BASE_PATH=${1:?BASE_PATH is required}
MASTER_PORT=${2:?MASTER_PORT is required}
GPUS_PER_NODE=${3:?GPUS_PER_NODE is required}

if [[ -n "${4:-}" ]]; then
  SPARSITY_STR="$4"
else
  SPARSITY_STR="${SPARSITY_STR:-50%}"
fi

TARGET_SPARSITY_FLOAT="${TARGET_SPARSITY_FLOAT:-$(echo "${SPARSITY_STR}" | sed 's/%//' | awk '{printf "%.6f", $1 / 100}')}"
SAVE_TAG_SUFFIX="${SAVE_TAG_SUFFIX:-}"

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                  --nnodes ${NNODES} \
                  --node_rank ${NODE_RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

CKPT="${MODEL_PATH:-/root/autodl-tmp/AAAI-prune/models/Llama-2-7b-chat-hf}"
CKPT_NAME="llama2_7B_model"
DATA_DIR="${DATA_DIR:-${BASE_PATH}/processed_data/fineweb/full/}"
SAVE_PATH="${BASE_PATH}/results/llama2/pruned/fineweb/ddp+_${SPARSITY_STR}${SAVE_TAG_SUFFIX}"

BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACC=${GRAD_ACC:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-1024}
LR=${LR:-0.02}
MASK_LR=${MASK_LR:-0.02}
WARMUP_ITERS=${WARMUP_ITERS:-50}
TOTAL_ITERS=${TOTAL_ITERS:-1000}
KD_WEIGHT=${KD_WEIGHT:-2.0}
KD_TEMP=${KD_TEMP:-1.0}
SEED=${SEED:-10}
SEED_ORDER=${SEED_ORDER:-10}

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
OPTS+=" --gradient-checkpointing"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 800"
OPTS+=" --lr ${LR}"
OPTS+=" --mask-lr ${MASK_LR}"
OPTS+=" --kd-weight ${KD_WEIGHT}"
OPTS+=" --kd-temp ${KD_TEMP}"
OPTS+=" --target-sparsity ${TARGET_SPARSITY_FLOAT}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters ${WARMUP_ITERS}"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.0"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 1"
OPTS+=" --total-iters ${TOTAL_ITERS}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num 1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
OPTS+=" --type lm"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export TORCH_COMPILE_DISABLE=1
export PYTHONPATH=${BASE_PATH}
export TARGET_SPARSITY_FLOAT

echo "DDP+ LLaMA2"
echo "SPARSITY_STR=${SPARSITY_STR}"
echo "TARGET_SPARSITY_FLOAT=${TARGET_SPARSITY_FLOAT}"
echo "TOTAL_ITERS=${TOTAL_ITERS}"
echo "SAVE_PATH=${SAVE_PATH}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS}"
echo "${CMD}"
mkdir -p "${SAVE_PATH}"
${CMD}
