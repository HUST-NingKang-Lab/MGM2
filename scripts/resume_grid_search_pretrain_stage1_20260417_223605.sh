#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted stage-1 grid search at outputs_grid/stage1/20260417_223605.
# Completed runs 001-005 are skipped. Run 006+ are continued in the same run group.
# Example:
#   bash scripts/resume_grid_search_pretrain_stage1_20260417_223605.sh

DATA_FILE="data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.768.pkl"
NTV3_EMBEDDING_PATH="ntv3_embeddings/otus_99_embeddings.h5"
QWEN3_EMBEDDING_PATH="qwen3_embeddings/microbeatlas_keywords_emb.h5"
BASE_CONFIG="src/configs/mgm2_default.json"
STAGE_ROOT="/data9/zhanghaohong/projects/MWorld/MGM2/outputs_grid/stage1/20260426_230440"
SUMMARY_CSV="${STAGE_ROOT}/summary.csv"
MAX_EPOCHS=100
BATCH_SIZE=1024
NUM_WORKERS=8
DEVICES=(1 2 3 4)
EXPERIMENT_PREFIX="stage1_pretrain"
RESUME_FROM_RUN_INDEX=6

LEARNING_RATES=(1e-4 5e-5)
WEIGHT_DECAYS=(0.01)
MASK_PROBS=(0.15 0.30)
LAMBDA_REGRESSIONS=(0.3)
QWEN3_LOSS_WEIGHTS=(0.1 0.2)

for required in "${DATA_FILE}" "${NTV3_EMBEDDING_PATH}" "${BASE_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file not found: ${required}" >&2
    exit 1
  fi
done

if [[ ! -d "${STAGE_ROOT}" ]]; then
  echo "Stage root not found: ${STAGE_ROOT}" >&2
  exit 1
fi

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

summary_has_run() {
  local run_name="$1"
  [[ -f "${SUMMARY_CSV}" ]] && awk -F, -v run="${run_name}" 'NR > 1 && $1 == run {found=1} END {exit found ? 0 : 1}' "${SUMMARY_CSV}"
}

summarize_run() {
  local run_name="$1"
  local run_dir="$2"
  local config_path="$3"
  local metrics_csv="${run_dir}/logs/${run_name}/version_0/metrics.csv"

  python scripts/summarize_grid_results.py \
    --metrics_csv "${metrics_csv}" \
    --summary_csv "${SUMMARY_CSV}" \
    --run_name "${run_name}" \
    --output_dir "${run_dir}" \
    --config_json "${config_path}" \
    --checkpoint_dir "${run_dir}/checkpoints"
}

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
          resume_ckpt="${run_dir}/checkpoints/last.ckpt"

          if (( run_index < RESUME_FROM_RUN_INDEX )); then
            echo "Skipping ${run_name}: marked completed before resume point"
            continue
          fi
          if summary_has_run "${run_name}"; then
            echo "Skipping ${run_name}: already present in ${SUMMARY_CSV}"
            continue
          fi

          mkdir -p "${run_dir}"
          if [[ ! -f "${config_path}" ]]; then
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
          fi

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
          if [[ -f "${resume_ckpt}" ]]; then
            pretrain_cmd+=(--resume "${resume_ckpt}")
            echo "Resuming ${run_name} from ${resume_ckpt}"
          else
            echo "Starting ${run_name} from scratch"
          fi

          "${pretrain_cmd[@]}" 2>&1 | tee -a "${run_dir}/train.log"
          summarize_run "${run_name}" "${run_dir}" "${config_path}"
        done
      done
    done
  done
done

echo "Stage-1 resume complete. Summary written to ${SUMMARY_CSV}"
