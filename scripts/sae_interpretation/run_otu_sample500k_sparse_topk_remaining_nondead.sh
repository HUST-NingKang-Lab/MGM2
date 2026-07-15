#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json}
SOURCE_ROOT=${SOURCE_ROOT:-outputs_sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_remaining_nondead}
FEATURE_SCORES=${FEATURE_SCORES:-${SOURCE_ROOT}/feature_scores.csv}
EXISTING_OUTPUTS_DIR=${EXISTING_OUTPUTS_DIR:-${SOURCE_ROOT}/codex_outputs}
EXCLUDE_FEATURE_IDS_FILE=${EXCLUDE_FEATURE_IDS_FILE:-${SOURCE_ROOT}/selected_interpretable_feature_ids.json}
BACKGROUND_JSON=${BACKGROUND_JSON:-${SOURCE_ROOT}/background_metadata.json}
DEVICE=${DEVICE:-cpu}
NUM_THREADS=${NUM_THREADS:-16}
MIN_POSITIVE_COUNT=${MIN_POSITIVE_COUNT:-1}

FEATURE_IDS=${FEATURE_IDS:-${OUTPUT_ROOT}/remaining_nondead_feature_ids.json}
FEATURE_TABLE=${FEATURE_TABLE:-${OUTPUT_ROOT}/remaining_nondead_features.csv}
LOOKUP_DIR=${LOOKUP_DIR:-${OUTPUT_ROOT}/feature_lookup}
EVIDENCE_DIR=${EVIDENCE_DIR:-${OUTPUT_ROOT}/evidence}
TASK_DIR=${TASK_DIR:-${OUTPUT_ROOT}/codex_tasks}
CODEX_OUTPUTS_DIR=${CODEX_OUTPUTS_DIR:-${OUTPUT_ROOT}/codex_outputs}

mkdir -p "${OUTPUT_ROOT}" "${LOOKUP_DIR}" "${EVIDENCE_DIR}" "${TASK_DIR}" "${CODEX_OUTPUTS_DIR}"

select_cmd=(
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python
  scripts/sae_interpretation/select_remaining_nondead_features.py
  --config "${CONFIG}"
  --feature-scores "${FEATURE_SCORES}"
  --existing-outputs-dir "${EXISTING_OUTPUTS_DIR}"
  --exclude-feature-ids-file "${EXCLUDE_FEATURE_IDS_FILE}"
  --output-json "${FEATURE_IDS}"
  --output-csv "${FEATURE_TABLE}"
  --min-positive-count "${MIN_POSITIVE_COUNT}"
)

if [[ -n "${MAX_FEATURES:-}" ]]; then
  select_cmd+=(--max-features "${MAX_FEATURES}")
fi
if [[ -n "${SORT_BY:-}" ]]; then
  select_cmd+=(--sort-by "${SORT_BY}")
fi
if [[ "${DESCENDING:-0}" == "1" ]]; then
  select_cmd+=(--descending)
fi

"${select_cmd[@]}"

lookup_cmd=(
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python
  scripts/sae_interpretation/build_feature_lookup.py
  --config "${CONFIG}"
  --feature-ids-file "${FEATURE_IDS}"
  --feature-scores "${FEATURE_SCORES}"
  --output-dir "${LOOKUP_DIR}"
  --device "${DEVICE}"
  --num-threads "${NUM_THREADS}"
)

if [[ -n "${CHUNK_SIZE:-}" ]]; then
  lookup_cmd+=(--chunk-size "${CHUNK_SIZE}")
fi
if [[ -n "${FEATURE_LOOKUP_BATCH_SIZE:-}" ]]; then
  lookup_cmd+=(--feature-batch-size "${FEATURE_LOOKUP_BATCH_SIZE}")
fi
if [[ -n "${MAX_EVENTS_PER_FEATURE:-}" ]]; then
  lookup_cmd+=(--max-events-per-feature "${MAX_EVENTS_PER_FEATURE}")
fi
if [[ -n "${MAX_CHUNKS:-}" ]]; then
  lookup_cmd+=(--max-chunks "${MAX_CHUNKS}")
fi

"${lookup_cmd[@]}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python \
  scripts/sae_interpretation/build_evidence_packages.py \
  --config "${CONFIG}" \
  --feature-ids-file "${FEATURE_IDS}" \
  --lookup-dir "${LOOKUP_DIR}" \
  --output-dir "${EVIDENCE_DIR}" \
  --background-json "${BACKGROUND_JSON}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python \
  scripts/sae_interpretation/prepare_codex_interpretation_tasks.py \
  --config "${CONFIG}" \
  --feature-ids-file "${FEATURE_IDS}" \
  --evidence-dir "${EVIDENCE_DIR}" \
  --output-dir "${TASK_DIR}"

cat > "${OUTPUT_ROOT}/README.md" <<EOF
# Remaining non-dead OTU SAE feature interpretation tasks

This directory is separate from:

\`${SOURCE_ROOT}\`

It contains lookup tables, evidence packages, and Codex task packets for features
that satisfy:

- present in \`${FEATURE_SCORES}\`
- \`positive_count >= ${MIN_POSITIVE_COUNT}\`
- not already present in \`${EXISTING_OUTPUTS_DIR}\`
- not listed in \`${EXCLUDE_FEATURE_IDS_FILE}\`

Outputs:

- \`${FEATURE_IDS}\`: selected remaining non-dead feature ids
- \`${FEATURE_TABLE}\`: selected feature score rows
- \`${LOOKUP_DIR}\`: top activation lookup tables
- \`${EVIDENCE_DIR}\`: deterministic evidence packages
- \`${TASK_DIR}\`: Codex task packets for manual/agent interpretation
- \`${CODEX_OUTPUTS_DIR}\`: empty target directory for future interpretation JSON outputs
EOF

printf 'Remaining non-dead feature interpretation assets saved to %s\n' "${OUTPUT_ROOT}"
