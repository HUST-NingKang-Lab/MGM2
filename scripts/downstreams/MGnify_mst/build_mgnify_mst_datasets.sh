#!/usr/bin/env bash
set -euo pipefail

# Build MGM2-format train/test datasets for the MGnify MST downstream benchmark.
# Run from the repository root:
#   bash scripts/downstreams/MGnify_mst/build_mgnify_mst_datasets.sh
#
# Labels are intentionally not handled here. Downstream training should read
# train_meta/test_meta and align labels to these pkl files by sample_id.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="/data9/zhanghaohong/projects/MGM2Benchmarks/MGnify_mst/expert"
OUTPUT_DIR="data/downstreams/MGnify_mst"

NTV3_EMBEDDING="ntv3_embeddings/otus_99_embeddings.h5"
FALLBACK_EMBEDDING="ntv3_embeddings/otus_99_taxonomy_fallback.h5"
H5_KEY="genus"
TOP_K_OTUS=768
H5_LOAD_ALL=true

TRAIN_INPUT="${INPUT_DIR}/train_inputs.h5_raw.h5"
TEST_INPUT="${INPUT_DIR}/test_inputs.h5_raw.h5"
TRAIN_OUTPUT="${OUTPUT_DIR}/train.pkl"
TEST_OUTPUT="${OUTPUT_DIR}/test.pkl"
TRAIN_RESOLVED_EMBEDDING="${OUTPUT_DIR}/train.resolved_embeddings.h5"
TEST_RESOLVED_EMBEDDING="${OUTPUT_DIR}/test.resolved_embeddings.h5"

mkdir -p "${OUTPUT_DIR}"

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
    echo "  h5_load_all: ${H5_LOAD_ALL}"
    echo "  log: ${log_file}"

    local h5_load_all_args=()
    if [[ "${H5_LOAD_ALL}" == "true" ]]; then
        h5_load_all_args+=(--h5_load_all)
    fi

    python -u src/data/build_dataset.py \
        --otu_table "${input_file}" \
        --format h5 \
        --h5_key "${H5_KEY}" \
        "${h5_load_all_args[@]}" \
        --ntv3_embedding "${NTV3_EMBEDDING}" \
        --fallback_embedding "${FALLBACK_EMBEDDING}" \
        --resolved_embedding "${resolved_embedding}" \
        --output "${output_file}" \
        --top_k_otus "${TOP_K_OTUS}" \
        2>&1 | tee "${log_file}"
}

build_split "train" "${TRAIN_INPUT}" "${TRAIN_OUTPUT}" "${TRAIN_RESOLVED_EMBEDDING}"
build_split "test" "${TEST_INPUT}" "${TEST_OUTPUT}" "${TEST_RESOLVED_EMBEDDING}"

echo "[$(date '+%F %T')] Finished MGnify MST dataset conversion"
