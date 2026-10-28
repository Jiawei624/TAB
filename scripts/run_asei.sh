#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-score}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" != "score" && "$mode" != "label" ]]; then
    echo "Usage: bash scripts/run_asei.sh {score|label} [TAB CLI options]" >&2
    exit 2
fi

seed="${ASEI_SEED:-2026}"
gpu="${ASEI_GPU:-1}"
profile="${ASEI_TRAINING_PROFILE:-original}"
if [[ "$profile" != "original" && "$profile" != "extended" ]]; then
    echo "ASEI_TRAINING_PROFILE must be original or extended" >&2
    exit 2
fi
hyper_params="${ASEI_HYPER_PARAMS:-}"
if [[ -z "$hyper_params" ]]; then
    hyper_params="{\"device\":\"auto\",\"seed\":${seed},\"training_profile\":\"${profile}\",\"win_size\":63,\"alpha\":0.01,\"frozen_evidence_backend\":\"canonical\",\"checkpoint_dir\":\"result/asei_checkpoints/v34/${profile}\"}"
fi

# CUDA_VISIBLE_DEVICES is set by TAB before CUDA initialization:
# logical cuda:0 -> the single physical GPU selected by ASEI_GPU.
exec "${ASEI_PYTHON:-python}" -B scripts/run_benchmark.py \
    --config-path "unfixed_detect_${mode}_multi_config.json" \
    --model-name self_impl.ASEI \
    --model-hyper-params "$hyper_params" \
    --gpus "$gpu" --eval-backend sequential --num-workers 1 \
    --seed "$seed" --report-method csv \
    --save-path "${mode}/ASEI/v34/${profile}/seed${seed}" "$@"
