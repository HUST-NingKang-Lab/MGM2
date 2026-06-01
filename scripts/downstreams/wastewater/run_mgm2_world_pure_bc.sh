#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALE="${SCALE:-scale_xlarge}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE="${CACHE:-data/downstreams/wastewater/cache/mgm2_${SCALE}_top200_tokens.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/wastewater/mgm2_world_model/mae_pure_bc_${SCALE}_top200}"
DEVICE="${DEVICE:-cuda:7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
FORCE="${FORCE:-0}"

if [[ "${FORCE}" != "1" && -s "${OUTPUT_DIR}/summary.csv" ]]; then
  echo "[skip] pure BC world model exists: ${OUTPUT_DIR}/summary.csv"
  exit 0
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "[error] missing dataset: ${DATASET}" >&2
  exit 1
fi
if [[ ! -f "${CACHE}" ]]; then
  echo "[error] missing MGM2 token cache: ${CACHE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

save_predictions_args=()
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  save_predictions_args=(--save_predictions)
fi

bash scripts/downstreams/wastewater/train_mgm2_world_model.sh \
  --dataset "${DATASET}" \
  --cache "${CACHE}" \
  --mode mae \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --loss_mse_weight 0 \
  "${save_predictions_args[@]}" \
  "$@" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
