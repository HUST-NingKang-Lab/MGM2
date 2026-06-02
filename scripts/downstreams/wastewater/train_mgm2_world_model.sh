#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALE="${SCALE:-scale_small}"
MODE="${MODE:-mae}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE="${CACHE:-data/downstreams/wastewater/cache/mgm2_${SCALE}_top200_tokens.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater/mgm2_world_model/${MODE}_${SCALE}_top200}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python scripts/downstreams/wastewater/train_mgm2_world_model.py \
  --dataset "${DATASET}" \
  --cache "${CACHE}" \
  --mode "${MODE}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  "$@"
