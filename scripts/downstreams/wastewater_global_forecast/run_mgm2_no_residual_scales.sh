#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
MODES="${MODES:-mae jepa}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE_DIR="${CACHE_DIR:-data/downstreams/wastewater/cache_ntv3_asv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/wastewater_global_forecast/mgm2_no_residual_ntv3_asv}"
DEVICE="${DEVICE:-cuda:7}"
MSE_WEIGHT="${MSE_WEIGHT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
JEPA_PROBE_EPOCHS="${JEPA_PROBE_EPOCHS:-30}"
JEPA_PROBE_PATIENCE="${JEPA_PROBE_PATIENCE:-6}"
FORCE="${FORCE:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

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

mkdir -p "${OUTPUT_ROOT}"

for scale in ${SCALES}; do
  cache="${CACHE_DIR}/mgm2_${scale}_top200_tokens.pkl"
  if [[ ! -f "${cache}" ]]; then
    echo "[error] missing token cache for ${scale}: ${cache}" >&2
    exit 1
  fi

  for mode in ${MODES}; do
    output_dir="${OUTPUT_ROOT}/${mode}_${scale}_top200"
    mkdir -p "${output_dir}"
    cmd=(
      bash scripts/downstreams/wastewater/train_mgm2_world_model.sh
        --dataset "${DATASET}"
        --cache "${cache}"
        --mode "${mode}"
        --output_dir "${output_dir}"
        --device "${DEVICE}"
        --batch_size "${BATCH_SIZE}"
        --max_epochs "${MAX_EPOCHS}"
        --patience "${PATIENCE}"
        --no_residual_persistence
        --abundance_head softmax
        --query_mode temporal_attention
        --loss_mse_weight "${MSE_WEIGHT}"
        --bc_loss_mode normalized
        --splits test
    )
    if [[ "${mode}" == "jepa" ]]; then
      cmd+=(--probe_epochs "${JEPA_PROBE_EPOCHS}" --probe_patience "${JEPA_PROBE_PATIENCE}")
    fi
    if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
      cmd+=(--save_predictions)
    fi

    run_or_skip "${output_dir}/summary.csv" "${scale} ${mode} no-residual MGM2 world model" \
      "${cmd[@]}" \
      2>&1 | tee "${output_dir}/train.log"
  done
done

echo "Done. Results:"
find "${OUTPUT_ROOT}" -maxdepth 2 -name summary.csv | sort
