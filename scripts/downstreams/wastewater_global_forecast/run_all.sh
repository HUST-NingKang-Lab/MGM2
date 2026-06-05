#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_MGM2="${RUN_MGM2:-1}"

echo "[run] simple persistence/context-mean baselines"
bash scripts/downstreams/wastewater_global_forecast/run_simple_baselines.sh

echo "[run] global abundance-only transformer baseline"
bash scripts/downstreams/wastewater_global_forecast/run_global_abundance_transformer.sh

if [[ "${RUN_MGM2}" == "1" ]]; then
  echo "[run] MGM2 no-residual global models"
  bash scripts/downstreams/wastewater_global_forecast/run_mgm2_no_residual_scales.sh
fi

echo "[run] summarize"
python scripts/downstreams/wastewater_global_forecast/summarize_global_forecast.py
