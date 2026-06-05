#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
MODES="${MODES:-mae jepa}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE_DIR="${CACHE_DIR:-data/downstreams/wastewater/cache_ntv3_asv}"
WORLD_ROOT="${WORLD_ROOT:-outputs_downstreams/wastewater/mgm2_world_model_ntv3_asv_by_plant}"
DEVICE="${DEVICE:-cuda:7}"

WORLD_BATCH_SIZE="${WORLD_BATCH_SIZE:-8}"
WORLD_MAX_EPOCHS="${WORLD_MAX_EPOCHS:-80}"
WORLD_PATIENCE="${WORLD_PATIENCE:-10}"
JEPA_PROBE_EPOCHS="${JEPA_PROBE_EPOCHS:-30}"
JEPA_PROBE_PATIENCE="${JEPA_PROBE_PATIENCE:-6}"
FORCE="${FORCE:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
PLANTS="${PLANTS:-}"

run_or_skip() {
  local done_file="$1"
  local description="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -s "${done_file}" ]]; then
    echo "[skip] ${description}: ${done_file} exists"
    return 0
  fi
  echo "[run] ${description}"
  "$@"
}

mkdir -p "${WORLD_ROOT}"

plant_table="$(mktemp)"
trap 'rm -f "${plant_table}"' EXIT

if [[ -n "${PLANTS}" ]]; then
  PLANTS="${PLANTS}" python - <<'PY' > "${plant_table}"
import os
import re
import unicodedata

def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return slug or "plant"

for plant in os.environ["PLANTS"].split("|"):
    plant = plant.strip()
    if plant:
        print(f"{plant}\t{slugify(plant)}")
PY
else
  DATASET="${DATASET}" python - <<'PY' > "${plant_table}"
import os
import pickle
import re
import unicodedata
from pathlib import Path

dataset = Path(os.environ["DATASET"])

def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return slug or "plant"

with dataset.open("rb") as handle:
    obj = pickle.load(handle)
for plant in sorted({window["plant"] for window in obj["windows"]}):
    print(f"{plant}\t{slugify(plant)}")
PY
fi

while IFS=$'\t' read -r plant plant_slug; do
  [[ -n "${plant}" ]] || continue
  for scale in ${SCALES}; do
    cache="${CACHE_DIR}/mgm2_${scale}_top200_tokens.pkl"
    if [[ ! -f "${cache}" ]]; then
      echo "[error] missing cache for ${scale}: ${cache}" >&2
      exit 1
    fi

    for mode in ${MODES}; do
      out_dir="${WORLD_ROOT}/${mode}_${scale}_top200/${plant_slug}"
      mkdir -p "${out_dir}"

      cmd=(
        bash scripts/downstreams/wastewater/train_mgm2_world_model.sh
          --dataset "${DATASET}"
          --cache "${cache}"
          --mode "${mode}"
          --plants "${plant}"
          --output_dir "${out_dir}"
          --device "${DEVICE}"
          --batch_size "${WORLD_BATCH_SIZE}"
          --max_epochs "${WORLD_MAX_EPOCHS}"
          --patience "${WORLD_PATIENCE}"
          --abundance_head logratio_residual
          --query_mode temporal_attention
          --residual_scale 1.0
          --loss_mse_weight 0.0
          --bc_loss_mode normalized
      )
      if [[ "${mode}" == "jepa" ]]; then
        cmd+=(--probe_epochs "${JEPA_PROBE_EPOCHS}" --probe_patience "${JEPA_PROBE_PATIENCE}")
      fi
      if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
        cmd+=(--save_predictions)
      fi

      run_or_skip "${out_dir}/summary.csv" "${plant} ${scale} ${mode} world model" \
        "${cmd[@]}" \
          2>&1 | tee "${out_dir}/train.log"
    done
  done
done < "${plant_table}"

echo "Done. Results:"
find "${WORLD_ROOT}" -maxdepth 3 -name summary.csv | sort
