#!/usr/bin/env bash
set -euo pipefail

# Build all-event MGM2 FMT dataset for standard internal 5-fold experiments.
# Existing data/downstreams/FMT files are not modified.
#
# Run from repository root:
#   bash scripts/downstreams/FMT/build_mgm2_fmt_5fold_dataset.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"

DATA_DIR="${DATA_DIR:-/home/zhanghaohong/data6/databases/fmt_strain/data}"
OUTPUT_DIR="${OUTPUT_DIR:-data/downstreams/FMT_5folds}"
SOURCE_MOTU_EMBEDDING="${SOURCE_MOTU_EMBEDDING:-ntv3_embeddings/fmt_motus_embeddings.h5}"
MOTU_EMBEDDING="${MOTU_EMBEDDING:-${OUTPUT_DIR}/community_all.motu_embeddings.h5}"
MAX_MOTUS="${MAX_MOTUS:-1024}"
TOP_K_OTUS="${TOP_K_OTUS:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "[$(date '+%F %T')] Preparing FMT all-event 5-fold tables"
"${PYTHON_BIN}" -u scripts/downstreams/FMT/prepare_fmt_5fold_tables.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_motus "${MAX_MOTUS}" \
    2>&1 | tee "${OUTPUT_DIR}/prepare_tables.log"

echo "[$(date '+%F %T')] Aggregating marker-level mOTU embeddings"
"${PYTHON_BIN}" -u scripts/downstreams/FMT/build_fmt_motu_embedding.py \
    --source_embedding "${SOURCE_MOTU_EMBEDDING}" \
    --selected_motus "${OUTPUT_DIR}/selected_motus.tsv" \
    --output "${MOTU_EMBEDDING}" \
    --summary "${OUTPUT_DIR}/community_all.motu_embeddings.summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/motu_embedding.build.log"

echo "[$(date '+%F %T')] Building community_all dataset"
"${PYTHON_BIN}" -u src/data/build_dataset.py \
    --otu_table "${OUTPUT_DIR}/community_all_otu_table.csv" \
    --format csv \
    --ntv3_embedding "${MOTU_EMBEDDING}" \
    --output "${OUTPUT_DIR}/community_all.pkl" \
    --top_k_otus "${TOP_K_OTUS}" \
    2>&1 | tee "${OUTPUT_DIR}/community_all.build.log"

echo "[$(date '+%F %T')] Finished FMT all-event 5-fold dataset conversion"
