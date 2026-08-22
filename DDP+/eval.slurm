#!/bin/bash
#SBATCH --job-name=eval_pruned_lm
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/gpool/home/wanghongyang/WangHY/minillm/Minillm
#SBATCH --output=/gpool/home/wanghongyang/WangHY/minillm/Minillm/duibi_Method/DDP+Y/logs/eval/eval_%x_%j.out
#SBATCH --error=/gpool/home/wanghongyang/WangHY/minillm/Minillm/duibi_Method/DDP+Y/logs/eval/eval_%x_%j.err

set -eEuo pipefail
trap 'echo "ERROR: line $LINENO failed with exit code $?" >&2' ERR

PROJECT_ROOT="/gpool/home/wanghongyang/WangHY/minillm/Minillm"
cd "$PROJECT_ROOT"
mkdir -p logs

export CUDA_HOME=/opt/cuda/12.4
export PATH=/gpool/home/wanghongyang/.conda/envs/minillm/bin:/opt/cuda/12.4/bin:$PATH
export MINILLM_PREFER_CONDA_SITE="${MINILLM_PREFER_CONDA_SITE:-1}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# 始终使用单 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  CLEAN_LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr ':' '\n' | awk 'index($0, "/stubs") == 0' | paste -sd ':' -)"
  export LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH}"
else
  unset LD_LIBRARY_PATH
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH must point to a checkpoint directory." >&2
  exit 2
fi

echo "=========================================================="
echo "Job Start Time: $(date)"
echo "Assigned Nodes: ${SLURM_JOB_NODELIST:-}"
echo "Assigned GPUs : ${CUDA_VISIBLE_DEVICES:-}"
echo "Model Path    : ${MODEL_PATH}"
echo "Eval Tasks    : ${EVAL_TASKS:-default}"
echo "Eval Dtype    : ${EVAL_DTYPE:-bf16}"
echo "Smoke Samples : ${SMOKE_SAMPLES:-16}"
echo "Conda Site First: ${MINILLM_PREFER_CONDA_SITE}"
echo "PYTHONPATH    : ${PYTHONPATH}"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv || true

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data/my_full_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false
echo "HF_HOME       : ${HF_HOME}"
echo "HF_DATASETS_CACHE : ${HF_DATASETS_CACHE}"
echo "HF_HUB_CACHE  : ${HF_HUB_CACHE}"

python - <<'PY'
import os
import sys
import torch

print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("CUDA is unavailable; refusing to run lm-eval on CPU.", file=sys.stderr)
    sys.exit(13)
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
PY

python -u run_lm_eval.py \
  --model-path "$MODEL_PATH" \
  --tasks "${EVAL_TASKS:-wikitext,boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa}" \
  --num-fewshot "${NUM_FEWSHOT:-0}" \
  --batch-size "${EVAL_BATCH_SIZE:-1}" \
  --output-name "${EVAL_OUTPUT_NAME:-ddp_comparison_results.json}" \
  --eval-dtype "${EVAL_DTYPE:-bf16}" \
  --smoke-samples "${SMOKE_SAMPLES:-16}"

echo "=========================================================="
echo "Evaluation finished at $(date)."
echo "=========================================================="
