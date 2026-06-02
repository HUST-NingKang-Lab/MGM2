#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data9/zhanghaohong/databases/wastewater/MC-prediction/data_organized}"
OUTPUT="${OUTPUT:-data/downstreams/wastewater/wastewater_windows_ctx10_h10.pkl}"
CONTEXT_LEN="${CONTEXT_LEN:-10}"
HORIZON="${HORIZON:-10}"
TOP_K_ASVS="${TOP_K_ASVS:-0}"

python scripts/downstreams/wastewater/build_wastewater_windows.py \
  --data_root "${DATA_ROOT}" \
  --output "${OUTPUT}" \
  --context_len "${CONTEXT_LEN}" \
  --horizon "${HORIZON}" \
  --top_k_asvs "${TOP_K_ASVS}" \
  "$@"
