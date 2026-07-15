#!/usr/bin/env bash
set -euo pipefail

# Train repeated fully connected FMT community baselines.
#
# Run from the repository root:
#   DEVICE="cuda:0" SEEDS="11 22 33 44 55" bash scripts/downstreams/FMT/train_mlp_baseline_seeds.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/FMT/mlp_baseline}"
SEEDS=(${SEEDS:-11 22 33 44 55})

mkdir -p "${OUTPUT_ROOT}"

for seed in "${SEEDS[@]}"; do
    run_dir="${OUTPUT_ROOT}/seed_${seed}"
    mkdir -p "${run_dir}"
    echo "[$(date '+%F %T')] Running FMT MLP baseline seed ${seed}"
    "${PYTHON_BIN}" -u scripts/downstreams/FMT/train_mlp_baseline.py \
        --output_dir "${run_dir}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        "$@"
done

echo "[$(date '+%F %T')] Finished FMT MLP baseline seed runs"
