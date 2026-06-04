#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater_global_forecast/simple_baselines_top200}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

args=(
  --dataset "${DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --splits test
)

if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  args+=(--save_predictions)
fi

python scripts/downstreams/wastewater/run_simple_baselines.py "${args[@]}"
