#!/usr/bin/env bash
set -euo pipefail

# Stage-1 fixed-architecture grid search for MGM2 pretraining.
# Edit the fixed settings in this header block before launching the script.
# Cross-run ranking policy:
#   1. val/bin_accuracy descending
#   2. val/regression_mse ascending
#
# Example:
#   bash scripts/grid_search_pretrain_stage1.sh

DATA_FILE="data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.768.pkl"
NTV3_EMBEDDING_PATH="ntv3_embeddings/otus_99_embeddings.h5"
QWEN3_EMBEDDING_PATH="qwen3_embeddings/microbeatlas_keywords_emb.h5"
BASE_CONFIG="src/configs/mgm2_default.json"
OUTPUT_ROOT="outputs_grid/stage1"
RUN_GROUP="$(date +%Y%m%d_%H%M%S)"
MAX_EPOCHS=100
BATCH_SIZE=1024
NUM_WORKERS=8
DEVICES=(2 3 4 5)
EXPERIMENT_PREFIX="stage1_pretrain"

LEARNING_RATES=(1e-4 5e-5)
WEIGHT_DECAYS=(0.01)
MASK_PROBS=(0.15 0.30)
LAMBDA_REGRESSIONS=(0.3)
QWEN3_LOSS_WEIGHTS=(0.1 0.2)

for required in "$DATA_FILE" "$NTV3_EMBEDDING_PATH" "$BASE_CONFIG"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

STAGE_ROOT="${OUTPUT_ROOT}/${RUN_GROUP}"
SUMMARY_CSV="${STAGE_ROOT}/summary.csv"
mkdir -p "${STAGE_ROOT}"

QWEN3_CONFIG_OVERRIDES=(--set qwen3_alignment.enabled=false --set qwen3_alignment.loss_weight=0.0)
QWEN3_PRETRAIN_ARGS=()
if [[ -n "${QWEN3_EMBEDDING_PATH}" ]]; then
  if [[ ! -f "${QWEN3_EMBEDDING_PATH}" ]]; then
    echo "Qwen3 embedding file not found: ${QWEN3_EMBEDDING_PATH}" >&2
    exit 1
  fi
  QWEN3_CONFIG_OVERRIDES=()
  QWEN3_PRETRAIN_ARGS=(--enable_qwen3_alignment --qwen3_embedding_path "${QWEN3_EMBEDDING_PATH}")
fi

run_index=0
for learning_rate in "${LEARNING_RATES[@]}"; do
  for weight_decay in "${WEIGHT_DECAYS[@]}"; do
    for mask_prob in "${MASK_PROBS[@]}"; do
      for lambda_regression in "${LAMBDA_REGRESSIONS[@]}"; do
        for qwen3_loss_weight in "${QWEN3_LOSS_WEIGHTS[@]}"; do
          ((run_index+=1))
          run_name=$(printf 'run_%03d_lr%s_wd%s_mask%s_reg%s_qw%s' "${run_index}" "${learning_rate}" "${weight_decay}" "${mask_prob}" "${lambda_regression}" "${qwen3_loss_weight}")
          run_dir="${STAGE_ROOT}/${run_name}"
          config_path="${run_dir}/config.json"
          metrics_csv="${run_dir}/logs/${run_name}/version_0/metrics.csv"

          mkdir -p "${run_dir}"
          build_config_cmd=(
            python scripts/build_grid_config.py
            --base "${BASE_CONFIG}"
            --output "${config_path}"
            --set training.learning_rate="${learning_rate}"
            --set training.weight_decay="${weight_decay}"
            --set training.mask_prob="${mask_prob}"
            --set loss.lambda_regression="${lambda_regression}"
            --set qwen3_alignment.loss_weight="${qwen3_loss_weight}"
          )
          if (( ${#QWEN3_CONFIG_OVERRIDES[@]} )); then
            build_config_cmd+=("${QWEN3_CONFIG_OVERRIDES[@]}")
          fi
          "${build_config_cmd[@]}"

          pretrain_cmd=(
            python -u src/training/Pretrain.py
            --data_file "${DATA_FILE}"
            --ntv3_embedding_path "${NTV3_EMBEDDING_PATH}"
            --config "${config_path}"
            --max_epochs "${MAX_EPOCHS}"
            --batch_size "${BATCH_SIZE}"
            --num_workers "${NUM_WORKERS}"
            --output_dir "${run_dir}"
            --experiment_name "${run_name}"
            --devices "${DEVICES[@]}"
          )
          if (( ${#QWEN3_PRETRAIN_ARGS[@]} )); then
            pretrain_cmd+=("${QWEN3_PRETRAIN_ARGS[@]}")
          fi
          "${pretrain_cmd[@]}" 2>&1 | tee "${run_dir}/train.log"

          python scripts/summarize_grid_results.py             --metrics_csv "${metrics_csv}"             --summary_csv "${SUMMARY_CSV}"             --run_name "${run_name}"             --output_dir "${run_dir}"             --config_json "${config_path}"             --checkpoint_dir "${run_dir}/checkpoints"
        done
      done
    done
  done
done

echo "Stage-1 summary written to ${SUMMARY_CSV}"
