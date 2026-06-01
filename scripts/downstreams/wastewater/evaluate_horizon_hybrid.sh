#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

LEARNED_NAME="${LEARNED_NAME:-vector_tcn}"
BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS:-outputs_downstreams/wastewater/baseline/baseline_top200/predictions.csv.gz}"
LEARNED_PREDICTIONS="${LEARNED_PREDICTIONS:-outputs_downstreams/wastewater/vector_temporal/tcn_top200/predictions.csv.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater/hybrid/${LEARNED_NAME}_persistence_top200}"

python scripts/downstreams/wastewater/evaluate_horizon_hybrid.py \
  --baseline_predictions "${BASELINE_PREDICTIONS}" \
  --learned_predictions "${LEARNED_PREDICTIONS}" \
  --learned_model "${LEARNED_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
