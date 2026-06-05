#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_FILE="${DATA_FILE:-data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.pkl}"
METADATA_FILE="${METADATA_FILE:-/data6/zhanghaohong/databases/microbeatals/samples.env.info.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/sae/pretrain_xlarge_sample200k}"
SEED="${SEED:-42}"
MAX_BUCKET_FRACTION="${MAX_BUCKET_FRACTION:-0.25}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python scripts/sae/build_pretrain_sae_sampling_manifest.py \
  --data_file "${DATA_FILE}" \
  --metadata_file "${METADATA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --animal_quota "${ANIMAL_QUOTA:-40000}" \
  --aquatic_quota "${AQUATIC_QUOTA:-40000}" \
  --soil_quota "${SOIL_QUOTA:-40000}" \
  --plant_quota "${PLANT_QUOTA:-40000}" \
  --missing_quota "${MISSING_QUOTA:-20000}" \
  --wgs_supplement "${WGS_SUPPLEMENT:-20000}" \
  --val_fraction "${VAL_FRACTION:-0.05}" \
  --test_fraction "${TEST_FRACTION:-0.05}" \
  --seed "${SEED}" \
  --max_bucket_fraction "${MAX_BUCKET_FRACTION}" \
  "$@"
