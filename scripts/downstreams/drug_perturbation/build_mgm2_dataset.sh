#!/usr/bin/env bash
set -euo pipefail

# Build compact MGM2-format data for drug perturbation downstream training.
# Run from the repository root:
#   bash scripts/downstreams/drug_perturbation/build_mgm2_dataset.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

DATA_DIR="/home/zhanghaohong/data6/databases/MicrobeDrugPerturbation/data"
OUTPUT_DIR="data/downstreams/drug_perturbation"
BASELINE_DIR="outputs_downstreams/drug_perturbation/baseline"
NTV3_EMBEDDING_PATH="ntv3_embeddings/otus_99_embeddings.h5"
MAX_OTUS=1024
MIN_READ_SUM=100
MIN_CONTROL_READ_SUM=100
OUTPUT_FILE="${OUTPUT_DIR}/drug_perturbation_top${MAX_OTUS}.pkl"
LOG_FILE="${OUTPUT_DIR}/build_top${MAX_OTUS}.log"

mkdir -p "${OUTPUT_DIR}"

echo "[$(date '+%F %T')] Building drug perturbation compact dataset"
echo "  raw data: ${DATA_DIR}"
echo "  ntv3: ${NTV3_EMBEDDING_PATH}"
echo "  baseline cache: ${BASELINE_DIR}"
echo "  output: ${OUTPUT_FILE}"

"${PYTHON_BIN}" -u src/data/build_drug_perturbation_dataset.py \
    --data_dir "${DATA_DIR}" \
    --ntv3_embedding_path "${NTV3_EMBEDDING_PATH}" \
    --output "${OUTPUT_FILE}" \
    --baseline_dir "${BASELINE_DIR}" \
    --max_otus "${MAX_OTUS}" \
    --min_read_sum "${MIN_READ_SUM}" \
    --min_control_read_sum "${MIN_CONTROL_READ_SUM}" \
    2>&1 | tee "${LOG_FILE}"

echo "[$(date '+%F %T')] Finished drug perturbation compact dataset build"
