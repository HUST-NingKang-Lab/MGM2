#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_ctx10_h10.pkl}"
DATA_ROOT="${DATA_ROOT:-/data9/zhanghaohong/databases/wastewater/MC-prediction/data_organized}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"
SCALE="${SCALE:-scale_small}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT="${OUTPUT:-data/downstreams/wastewater/cache/mgm2_${SCALE}_cls.pkl}"

python scripts/downstreams/wastewater/extract_mgm2_cls_cache.py \
  --dataset "${DATASET}" \
  --data_root "${DATA_ROOT}" \
  --stage2_root "${STAGE2_ROOT}" \
  --scale "${SCALE}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --output "${OUTPUT}" \
  "$@"
