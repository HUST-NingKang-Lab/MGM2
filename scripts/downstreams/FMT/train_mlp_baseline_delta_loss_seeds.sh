#!/usr/bin/env bash
set -euo pipefail

# Train repeated exploratory MLP FMT community baselines with post + delta loss.
#
# Run from the repository root:
#   DEVICE="cuda:0" SEEDS="11 22 33 44 55" bash scripts/downstreams/FMT/train_mlp_baseline_delta_loss_seeds.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/FMT/mlp_baseline_exploratory/delta_loss}"
SEEDS=(${SEEDS:-11 22 33 44 55})
DELTA_LOSS_WEIGHT="${DELTA_LOSS_WEIGHT:-0.5}"

mkdir -p "${OUTPUT_ROOT}"

for seed in "${SEEDS[@]}"; do
    run_dir="${OUTPUT_ROOT}/seed_${seed}"
    mkdir -p "${run_dir}"
    echo "[$(date '+%F %T')] Running exploratory FMT MLP delta-loss seed ${seed}"
    "${PYTHON_BIN}" -u scripts/downstreams/FMT/train_mlp_baseline_delta_loss.py \
        --output_dir "${run_dir}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        --delta_loss_weight "${DELTA_LOSS_WEIGHT}" \
        "$@"
done

echo "[$(date '+%F %T')] Finished exploratory FMT MLP delta-loss seed runs"
