#!/usr/bin/env bash
set -euo pipefail

# Build Cell Reports FMT all-feature MGM2 dataset.
# Uses bacteria + archaea + fungi by default; viral data are intentionally excluded.
#
# Run from repository root:
#   bash scripts/downstreams/FMT_cellreports/build_mgm2_fmt_cellreports_dataset.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DATA_DIR="${DATA_DIR:-/data9/zhanghaohong/databases/fmt_strain/cell_reports_data}"
OUTPUT_DIR="${OUTPUT_DIR:-data/downstreams/FMT_cellreports}"
NTV3_EMBEDDING="${NTV3_EMBEDDING:-ntv3_embeddings/otus_99_embeddings.h5}"
OTU_INFO="${OTU_INFO:-/data9/zhanghaohong/databases/microbeatals/otus.info.tsv}"
KINGDOMS=(${KINGDOMS:-bacteria archaea fungi})
H5_LOAD_ALL="${H5_LOAD_ALL:-1}"

mkdir -p "${OUTPUT_DIR}"

echo "[$(date '+%F %T')] Inspecting Cell Reports FMT data"
"${PYTHON_BIN}" -u scripts/downstreams/FMT_cellreports/inspect_cellreports_data.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --kingdoms "${KINGDOMS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/inspect.log"

echo "[$(date '+%F %T')] Preparing Cell Reports tables and HDF5 community matrix"
"${PYTHON_BIN}" -u scripts/downstreams/FMT_cellreports/prepare_cellreports_tables.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --kingdoms "${KINGDOMS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/prepare_tables.log"

echo "[$(date '+%F %T')] Building direct NTv3-derived Cell Reports feature embeddings"
"${PYTHON_BIN}" -u scripts/downstreams/FMT_cellreports/build_cellreports_embeddings.py \
  --features "${OUTPUT_DIR}/cellreports_features.tsv" \
  --ntv3_embedding "${NTV3_EMBEDDING}" \
  --otu_info "${OTU_INFO}" \
  --output "${OUTPUT_DIR}/community_all.cellreports_embeddings.h5" \
  --summary "${OUTPUT_DIR}/community_all.cellreports_embeddings.summary.tsv" \
  2>&1 | tee "${OUTPUT_DIR}/cellreports_embeddings.build.log"

echo "[$(date '+%F %T')] Building MGM2 preprocessed dataset"
build_args=(
  src/data/build_dataset.py
  --otu_table "${OUTPUT_DIR}/community_all.h5"
  --format h5
  --h5_key cellreports
  --ntv3_embedding "${OUTPUT_DIR}/community_all.cellreports_embeddings.h5"
  --output "${OUTPUT_DIR}/community_all.pkl"
  --top_k_otus 768
)
if [[ "${H5_LOAD_ALL}" == "1" ]]; then
  build_args+=(--h5_load_all)
fi
"${PYTHON_BIN}" -u "${build_args[@]}" 2>&1 | tee "${OUTPUT_DIR}/community_all.build.log"

echo "[$(date '+%F %T')] Preparing leave-one-cohort-out splits"
"${PYTHON_BIN}" -u scripts/downstreams/FMT_cellreports/prepare_loco_splits.py \
  --data_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/loco_splits" \
  2>&1 | tee "${OUTPUT_DIR}/loco_splits.build.log"

echo "[$(date '+%F %T')] Preparing donor-similarity random splits"
"${PYTHON_BIN}" -u scripts/downstreams/FMT_cellreports/prepare_similarity_random_splits.py \
  --data_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/similarity_random_splits" \
  2>&1 | tee "${OUTPUT_DIR}/similarity_random_splits.build.log"

echo "[$(date '+%F %T')] Finished Cell Reports FMT dataset build"
