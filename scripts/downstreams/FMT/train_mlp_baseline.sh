#!/usr/bin/env bash
set -euo pipefail

# Train the fully connected FMT baseline on the prepared FMT community dataset.
#
# Run from the repository root:
#   DEVICE="cuda:0" bash scripts/downstreams/FMT/train_mlp_baseline.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/FMT/mlp_baseline}"
SEED="${SEED:-42}"

"${PYTHON_BIN}" -u scripts/downstreams/FMT/train_mlp_baseline.py \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "$@"
