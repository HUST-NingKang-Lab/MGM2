#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALE="${SCALE:-scale_small}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_ctx10_h10.pkl}"
CACHE="${CACHE:-data/downstreams/wastewater/cache/mgm2_${SCALE}_cls.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater/mgm2_temporal/${SCALE}}"
DEVICE="${DEVICE:-cuda:0}"

python scripts/downstreams/wastewater/train_mgm2_temporal_forecaster.py \
  --dataset "${DATASET}" \
  --cache "${CACHE}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
