#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/downstreams/FMT_5folds}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/FMT_5folds/mlp_community_prediction}"
PYTHON="${PYTHON:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DEVICE="${DEVICE:-auto}"
SEEDS=(${SEEDS:-0})
FOLDS=(${FOLDS:-0 1 2 3 4})
NUM_FOLDS="${NUM_FOLDS:-5}"
LABEL_COLUMN="${LABEL_COLUMN:-indication}"

DATA_FILE="${DATA_FILE:-${DATA_DIR}/community_all.pkl}"
METADATA_FILE="${METADATA_FILE:-${DATA_DIR}/post_all_meta.csv}"
NTV3_EMBEDDING="${NTV3_EMBEDDING:-${DATA_DIR}/community_all.motu_embeddings.h5}"

for required in "${DATA_FILE}" "${METADATA_FILE}" "${NTV3_EMBEDDING}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    out_dir="${OUTPUT_ROOT}/seed_${seed}/fold_${fold}"
    echo "Training FMT MLP community seed=${seed} fold=${fold}/${NUM_FOLDS}"
    "${PYTHON}" scripts/downstreams/FMT/train_mlp_community_5folds.py \
      --data_file "${DATA_FILE}" \
      --metadata_file "${METADATA_FILE}" \
      --ntv3_embedding_path "${NTV3_EMBEDDING}" \
      --output_dir "${out_dir}" \
      --fold_index "${fold}" \
      --num_folds "${NUM_FOLDS}" \
      --label_column "${LABEL_COLUMN}" \
      --seed "${seed}" \
      --device "${DEVICE}"
  done
done
