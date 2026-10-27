#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-score}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" != "score" && "$mode" != "label" ]]; then
    echo "Usage: bash scripts/run_asei.sh {score|label} [TAB CLI options]" >&2
    exit 2
fi

seed="${ASEI_SEED:-0}"
hyper_params="${ASEI_HYPER_PARAMS:-}"
if [[ -z "$hyper_params" ]]; then
    hyper_params='{"multi_gpu":true,"alpha":0.01,"checkpoint_dir":"result/asei_checkpoints"}'
fi

# CUDA_VISIBLE_DEVICES is set by TAB before CUDA initialization:
# logical cuda:0 -> physical GPU1 (primary), logical cuda:1 -> physical GPU0.
exec "${ASEI_PYTHON:-python}" -B scripts/run_benchmark.py \
    --config-path "asei_detect_${mode}_multi_config.json" \
    --model-name self_impl.ASEI \
    --model-hyper-params "$hyper_params" \
    --gpus 1 0 --eval-backend sequential --num-workers 1 \
    --seed "$seed" --report-method csv \
    --save-path "${mode}/ASEI/seed${seed}" "$@"
