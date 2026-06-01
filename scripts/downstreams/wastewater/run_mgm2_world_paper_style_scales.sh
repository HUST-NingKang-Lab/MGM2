#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE_DIR="${CACHE_DIR:-data/downstreams/wastewater/cache}"
WORLD_ROOT="${WORLD_ROOT:-outputs_downstreams/wastewater/mgm2_world_model_paper_style}"
DEVICE="${DEVICE:-cuda:7}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
JEPA_PROBE_EPOCHS="${JEPA_PROBE_EPOCHS:-30}"
JEPA_PROBE_PATIENCE="${JEPA_PROBE_PATIENCE:-6}"

QUERY_MODE="${QUERY_MODE:-temporal_attention}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.0}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
FORCE="${FORCE:-1}"

save_predictions_args=()
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  save_predictions_args=(--save_predictions)
fi

run_or_skip() {
  local done_file="$1"
  local description="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -s "${done_file}" ]]; then
    echo "[skip] ${description}: ${done_file} exists"
    return 0
  fi
  echo "[run] ${description}"
  "$@"
}

for scale in ${SCALES}; do
  cache="${CACHE_DIR}/mgm2_${scale}_top200_tokens.pkl"
  mae_dir="${WORLD_ROOT}/mae_${scale}_top200"
  jepa_dir="${WORLD_ROOT}/jepa_${scale}_top200"

  if [[ ! -f "${DATASET}" ]]; then
    echo "[error] missing dataset: ${DATASET}" >&2
    exit 1
  fi
  if [[ ! -f "${cache}" ]]; then
    echo "[error] missing MGM2 token cache for ${scale}: ${cache}" >&2
    exit 1
  fi

  mkdir -p "${mae_dir}" "${jepa_dir}"

  run_or_skip "${mae_dir}/summary.csv" "${scale} paper-style MAE world model" \
    bash scripts/downstreams/wastewater/train_mgm2_world_model.sh \
      --dataset "${DATASET}" \
      --cache "${cache}" \
      --mode mae \
      --output_dir "${mae_dir}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE}" \
      --max_epochs "${MAX_EPOCHS}" \
      --patience "${PATIENCE}" \
      --abundance_head paper_relu \
      --query_mode "${QUERY_MODE}" \
      --bc_loss_mode raw \
      --loss_mse_weight "${LOSS_MSE_WEIGHT}" \
      "${save_predictions_args[@]}" \
      2>&1 | tee "${mae_dir}/train.log"

  run_or_skip "${jepa_dir}/summary.csv" "${scale} paper-style JEPA world model" \
    bash scripts/downstreams/wastewater/train_mgm2_world_model.sh \
      --dataset "${DATASET}" \
      --cache "${cache}" \
      --mode jepa \
      --output_dir "${jepa_dir}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE}" \
      --max_epochs "${MAX_EPOCHS}" \
      --patience "${PATIENCE}" \
      --probe_epochs "${JEPA_PROBE_EPOCHS}" \
      --probe_patience "${JEPA_PROBE_PATIENCE}" \
      --abundance_head paper_relu \
      --query_mode "${QUERY_MODE}" \
      --bc_loss_mode raw \
      --loss_mse_weight "${LOSS_MSE_WEIGHT}" \
      "${save_predictions_args[@]}" \
      2>&1 | tee "${jepa_dir}/train.log"
done

echo "Done. Results:"
find "${WORLD_ROOT}" -maxdepth 2 -name summary.csv | sort
