#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ACTIVATIONS="${ACTIVATIONS:-data/sae/pretrain_xlarge_sample50k/mgm2_scale_xlarge_last_hidden_activations.h5}"
TOKEN_TYPE="${TOKEN_TYPE:-cls}"
DICT_SIZE_GRID="${DICT_SIZE_GRID:-1024 2048 4096}"
TOP_K_GRID="${TOP_K_GRID:-8 16 32}"
AUX_PENALTY_GRID="${AUX_PENALTY_GRID:-0.01 0.03}"
AUX_K="${AUX_K:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
STEPS="${STEPS:-200000}"
DEVICE="${DEVICE:-cuda:6}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EARLY_STOPPING="${EARLY_STOPPING:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-5}"
EARLY_STOPPING_MIN_STEPS="${EARLY_STOPPING_MIN_STEPS:-10000}"
PRELOAD_TO_MEMORY="${PRELOAD_TO_MEMORY:-1}"
PRELOAD_DTYPE="${PRELOAD_DTYPE:-float16}"
PRELOAD_CHUNK_ROWS="${PRELOAD_CHUNK_ROWS:-262144}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_sae/mgm2_xlarge_${TOKEN_TYPE}_batchtopk_grid}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${OUTPUT_ROOT}"

for dict_size in ${DICT_SIZE_GRID}; do
  for top_k in ${TOP_K_GRID}; do
    if (( top_k >= dict_size )); then
      echo "Skipping invalid run: dict_size=${dict_size} top_k=${top_k}"
      continue
    fi
    for aux_penalty in ${AUX_PENALTY_GRID}; do
      aux_label="${aux_penalty//./p}"
      output_dir="${OUTPUT_ROOT}/d${dict_size}_k${top_k}_aux${aux_label}"
      log_path="${output_dir}/run.log"

      if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/metrics.csv" ]]; then
        echo "Skipping existing run: ${output_dir}"
        continue
      fi

      mkdir -p "${output_dir}"
      echo "Running CLS BatchTopK SAE: dict_size=${dict_size} top_k=${top_k} aux_penalty=${aux_penalty} output=${output_dir}"
      (
        export ACTIVATIONS
        export TOKEN_TYPE
        export DICT_SIZE="${dict_size}"
        export TOP_K="${top_k}"
        export AUX_K
        export AUX_PENALTY="${aux_penalty}"
        export BATCH_SIZE
        export STEPS
        export DEVICE
        export LEARNING_RATE
        export WEIGHT_DECAY
        export EVAL_EVERY
        export EVAL_BATCHES
        export SAVE_EVERY
        export EARLY_STOPPING
        export EARLY_STOPPING_PATIENCE
        export EARLY_STOPPING_MIN_DELTA
        export EARLY_STOPPING_MIN_STEPS
        export PRELOAD_TO_MEMORY
        export PRELOAD_DTYPE
        export PRELOAD_CHUNK_ROWS
        export OUTPUT_DIR="${output_dir}"
        bash scripts/sae/train_mgm2_xlarge_otu_batchtopk_sae.sh
      ) 2>&1 | tee "${log_path}"
    done
  done
done
