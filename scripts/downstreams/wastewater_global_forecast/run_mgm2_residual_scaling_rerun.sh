#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
SEEDS="${SEEDS:-42}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE_DIR="${CACHE_DIR:-data/downstreams/wastewater/cache_ntv3_asv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/wastewater_global_forecast/mgm2_residual_ntv3_asv_rerun}"
DEVICE="${DEVICE:-cuda:7}"
MSE_WEIGHT="${MSE_WEIGHT:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
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

for seed in ${SEEDS}; do
  for scale in ${SCALES}; do
    cache="${CACHE_DIR}/mgm2_${scale}_top200_tokens.pkl"
    if [[ ! -f "${cache}" ]]; then
      echo "[error] missing token cache for ${scale}: ${cache}" >&2
      exit 1
    fi

    output_dir="${OUTPUT_ROOT}/seed_${seed}/mae_${scale}_top200"
    mkdir -p "${output_dir}"
    cmd=(
      bash scripts/downstreams/wastewater/train_mgm2_world_model.sh
        --dataset "${DATASET}"
        --cache "${cache}"
        --mode mae
        --output_dir "${output_dir}"
        --device "${DEVICE}"
        --batch_size "${BATCH_SIZE}"
        --max_epochs "${MAX_EPOCHS}"
        --patience "${PATIENCE}"
        --seed "${seed}"
        --abundance_head logratio_residual
        --query_mode temporal_attention
        --residual_scale 1.0
        --loss_mse_weight "${MSE_WEIGHT}"
        --bc_loss_mode normalized
        --splits test
    )
    if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
      cmd+=(--save_predictions)
    fi

    run_or_skip "${output_dir}/summary.csv" "seed=${seed} ${scale} residual MAE scaling rerun" \
      "${cmd[@]}" \
      2>&1 | tee "${output_dir}/train.log"
  done
done

python scripts/downstreams/wastewater_global_forecast/summarize_mgm2_residual_scaling_rerun.py \
  --root "${OUTPUT_ROOT}"

echo "Done. Results:"
find "${OUTPUT_ROOT}" -maxdepth 3 -name summary.csv | sort
