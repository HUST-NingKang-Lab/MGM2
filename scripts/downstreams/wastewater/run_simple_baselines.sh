#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_ctx10_h10.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater/baselines}"

python scripts/downstreams/wastewater/run_simple_baselines.py \
  --dataset "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
