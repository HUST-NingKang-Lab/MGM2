#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
DATA_ROOT="${DATA_ROOT:-/home/zhanghaohong/data6/databases/wastewater/MC-prediction/data_organized}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"
SCALE="${SCALE:-scale_small}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT="${OUTPUT:-data/downstreams/wastewater/cache/mgm2_${SCALE}_top200_tokens.pkl}"

python scripts/downstreams/wastewater/extract_mgm2_token_cache.py \
  --dataset "${DATASET}" \
  --data_root "${DATA_ROOT}" \
  --stage2_root "${STAGE2_ROOT}" \
  --scale "${SCALE}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --output "${OUTPUT}" \
  "$@"
