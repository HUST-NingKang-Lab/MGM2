# SAE Feature Interpretation

This workflow converts trained OTU-token SAE features into reproducible evidence
packages and structured feature annotations. It combines activation statistics,
sample metadata, abundance context, and OTU taxonomy while preserving the
source records used for each description.

## Inputs and configuration

A JSON configuration under `configs/sae_interpretation/` defines the SAE
checkpoint, activation HDF5 file, token manifest, sample metadata, taxonomy
sources, selected splits, execution device, and output directory. Start from an
existing configuration and replace all artifact paths with locations available
in the current installation.

The principal inputs are:

```text
SAE best.pt checkpoint
MGM2 activation HDF5
tokens.csv.gz
samples.csv
OTU taxonomy metadata
```

## Build feature evidence

Run the orchestrated pipeline for explicitly selected features:

```bash
python scripts/sae_interpretation/run_otu_feature_interpretation_pipeline.py \
  --config configs/sae_interpretation/otu_xlarge_k64_aux0p03.json \
  --feature-ids 12,84,215 \
  --device cpu \
  --num-threads 16
```

Selection can instead use `--feature-ids-file`, `--top-n-features N`, or
`--all-features`; exactly one mode is required. For global ranking, the pipeline
first computes deterministic activation statistics with
`score_sae_features.py`.

The orchestrator then creates:

```text
background_metadata.json
feature_scores.csv
feature_lookup/feature_*.events.csv.gz
feature_lookup/feature_*.lookup_summary.json
evidence/feature_*.evidence.json
codex_tasks/feature_*/
```

The lookup tables retain top activating events and attach taxonomy only for OTUs
needed by the selected features. Evidence packages summarize activation,
environment, technology, abundance-rank, project, and taxonomic distributions.

## Global catalog

Build lookup and evidence records for the complete SAE dictionary:

```bash
CONFIG=configs/sae_interpretation/otu_xlarge_k64_aux0p03_global.json \
DEVICE=cpu \
NUM_THREADS=16 \
bash scripts/sae_interpretation/run_otu_global_feature_catalog.sh
```

To continue a catalog without revisiting completed or inactive features, first
score all features and select the remaining non-dead feature IDs:

```bash
python scripts/sae_interpretation/score_sae_features.py \
  --config configs/sae_interpretation/otu_xlarge_k64_aux0p03.json \
  --device cpu

python scripts/sae_interpretation/select_remaining_nondead_features.py \
  --config configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json \
  --feature-scores outputs_sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global/feature_scores.csv \
  --existing-outputs-dir outputs_sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global/codex_outputs \
  --exclude-feature-ids-file outputs_sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global/selected_interpretable_feature_ids.json \
  --output-json outputs_sae_interpretation/remaining_nondead_feature_ids.json \
  --output-csv outputs_sae_interpretation/remaining_nondead_features.csv \
  --min-positive-count 1
```

Pass the generated feature-ID JSON to the orchestrated pipeline with
`--feature-ids-file`.

## Validate and summarize annotations

Structured interpretation outputs are expected under the configured
`codex_outputs/feature_<id>/` directories. Validate schema and content before
aggregation:

```bash
python scripts/sae_interpretation/validate_codex_outputs.py \
  --config configs/sae_interpretation/otu_xlarge_k64_aux0p03.json \
  --strict-quality

python scripts/sae_interpretation/summarize_feature_interpretations.py \
  --config configs/sae_interpretation/otu_xlarge_k64_aux0p03.json
```

The summary step writes a machine-readable JSON report and a tabular CSV with
feature names, descriptions, confidence labels, assigned classes, and
acceptance status. Plotting scripts in this directory consume these validated
tables and evidence packages to generate feature-level and catalog-level
figures.
