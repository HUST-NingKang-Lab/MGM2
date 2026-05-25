#!/usr/bin/env bash
set -euo pipefail

# Build MGnify-style MGM2 datasets for FMT downstream experiments.
# Run from repository root:
#   bash scripts/downstreams/FMT/build_mgm2_fmt_dataset.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

DATA_DIR="${DATA_DIR:-/home/zhanghaohong/data6/databases/fmt_strain/data}"
OUTPUT_DIR="${OUTPUT_DIR:-data/downstreams/FMT}"
NTV3_EMBEDDING="${NTV3_EMBEDDING:-ntv3_embeddings/otus_99_embeddings.h5}"
FALLBACK_EMBEDDING="${FALLBACK_EMBEDDING:-ntv3_embeddings/otus_99_taxonomy_fallback.h5}"
MAX_MOTUS="${MAX_MOTUS:-1024}"
TEST_RATIO="${TEST_RATIO:-0.2}"
SEED="${SEED:-42}"
TOP_K_OTUS="${TOP_K_OTUS:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "[$(date '+%F %T')] Preparing FMT MGnify-style tables"
"${PYTHON_BIN}" -u scripts/downstreams/FMT/prepare_fmt_mgm2_tables.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_motus "${MAX_MOTUS}" \
    --test_ratio "${TEST_RATIO}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUTPUT_DIR}/prepare_tables.log"

build_split() {
    local split_name="$1"
    local input_file="$2"
    local output_file="$3"
    local resolved_embedding="$4"
    local log_file="${OUTPUT_DIR}/${split_name}.build.log"

    echo "[$(date '+%F %T')] Building ${split_name} dataset"
    echo "  input: ${input_file}"
    echo "  output: ${output_file}"
    echo "  resolved_embedding: ${resolved_embedding}"

    "${PYTHON_BIN}" -u src/data/build_dataset.py \
        --otu_table "${input_file}" \
        --format csv \
        --ntv3_embedding "${NTV3_EMBEDDING}" \
        --fallback_embedding "${FALLBACK_EMBEDDING}" \
        --resolved_embedding "${resolved_embedding}" \
        --output "${output_file}" \
        --top_k_otus "${TOP_K_OTUS}" \
        2>&1 | tee "${log_file}"
}

build_split "response_train" \
    "${OUTPUT_DIR}/response_train_otu_table.csv" \
    "${OUTPUT_DIR}/response_train.pkl" \
    "${OUTPUT_DIR}/response_train.resolved_embeddings.h5"

build_split "response_test" \
    "${OUTPUT_DIR}/response_test_otu_table.csv" \
    "${OUTPUT_DIR}/response_test.pkl" \
    "${OUTPUT_DIR}/response_test.resolved_embeddings.h5"

build_split "community_train" \
    "${OUTPUT_DIR}/community_train_otu_table.csv" \
    "${OUTPUT_DIR}/community_train.pkl" \
    "${OUTPUT_DIR}/community_train.resolved_embeddings.h5"

build_split "community_test" \
    "${OUTPUT_DIR}/community_test_otu_table.csv" \
    "${OUTPUT_DIR}/community_test.pkl" \
    "${OUTPUT_DIR}/community_test.resolved_embeddings.h5"

echo "[$(date '+%F %T')] Finished FMT MGM2 dataset conversion"
