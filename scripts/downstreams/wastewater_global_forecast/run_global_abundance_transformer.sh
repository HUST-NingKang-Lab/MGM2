#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater_global_forecast/abundance_transformer_top200}"
DEVICE="${DEVICE:-cuda:7}"
MSE_WEIGHT="${MSE_WEIGHT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-120}"
PATIENCE="${PATIENCE:-15}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

args=(
  --dataset "${DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --max_epochs "${MAX_EPOCHS}"
  --patience "${PATIENCE}"
  --hidden_size "${HIDDEN_SIZE}"
  --num_layers "${NUM_LAYERS}"
  --num_heads "${NUM_HEADS}"
  --loss_mse_weight "${MSE_WEIGHT}"
  --splits test
)

if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  args+=(--save_predictions)
fi

python scripts/downstreams/wastewater_global_forecast/train_global_abundance_transformer.py "${args[@]}"
