#!/usr/bin/env bash
set -euo pipefail

# Train microbiome-only MOZAIC reproduction on the same Cell Reports
# donor-similarity random splits used by MGM2.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DATA_DIR="${DATA_DIR:-data/downstreams/FMT_cellreports}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_downstreams/FMT_cellreports/mozaic_similarity_random}"
SPLITS=(${SPLITS:-0 1 2 3 4 5 6 7 8 9})
DEVICE="${DEVICE:-cuda:0}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"

"${PYTHON_BIN}" -u scripts/downstreams/FMT_cellreports/train_mozaic_similarity_random.py \
  --data_file "${DATA_DIR}/community_all.pkl" \
  --ntv3_embedding_path "${DATA_DIR}/community_all.cellreports_embeddings.h5" \
  --split_root "${DATA_DIR}/similarity_random_splits" \
  --output_dir "${OUTPUT_DIR}" \
  --splits "${SPLITS[@]}" \
  --device "${DEVICE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}"
