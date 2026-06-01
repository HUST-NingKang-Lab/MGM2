#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
MODEL_TYPE="${MODEL_TYPE:-gru}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater/vector_temporal/${MODEL_TYPE}_top200}"
DEVICE="${DEVICE:-cuda:0}"

python scripts/downstreams/wastewater/train_vector_temporal_baseline.py \
  --dataset "${DATASET}" \
  --model_type "${MODEL_TYPE}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
