# Cell Reports FMT Benchmark

This directory adapts the Cell Reports FMT cohort collection to MGM2. It
supports post-FMT community prediction, clinical response classification, and
classification of whether the recipient community becomes more or less similar
to the donor after FMT.

## Inputs

The raw-data directory contains `FMT_metadata_summary.xlsx` and kingdom-level
feature tables. The default build combines bacterial, archaeal, and fungal
profiles. Feature embeddings are constructed from an NTv3 embedding store and
an OTU metadata table.

## Build the dataset and splits

```bash
DATA_DIR=/path/to/cell_reports_data \
OUTPUT_DIR=data/downstreams/FMT_cellreports \
NTV3_EMBEDDING=/path/to/otus_99_embeddings.h5 \
OTU_INFO=/path/to/otus.info.tsv \
KINGDOMS="bacteria archaea fungi" \
bash scripts/downstreams/FMT_cellreports/build_mgm2_fmt_cellreports_dataset.sh
```

The build performs four steps: metadata and coverage inspection, multi-kingdom
community-table construction, feature-embedding construction, and split
generation. Principal outputs include:

```text
community_all.pkl
community_all.h5
community_all.cellreports_embeddings.h5
cellreports_metadata_normalized.csv
loco_splits/
similarity_random_splits/
```

`loco_splits/` contains leave-one-cohort-out train/test metadata for community
and response tasks. `similarity_random_splits/` contains repeated train,
validation, and test partitions for donor-similarity classification.

## Leave-one-cohort-out evaluation

Post-FMT community prediction:

```bash
DATA_DIR=data/downstreams/FMT_cellreports \
STAGE2_ROOT=/path/to/stage2 \
STAGE2_SCALES="scale_small scale_medium scale_large scale_xlarge" \
DEVICES="0" \
bash scripts/downstreams/FMT_cellreports/train_cellreports_loco_community_all_scales.sh
```

Clinical response classification:

```bash
DATA_DIR=data/downstreams/FMT_cellreports \
STAGE2_ROOT=/path/to/stage2 \
STAGE2_SCALES="scale_small scale_medium scale_large scale_xlarge" \
DEVICES="0" \
bash scripts/downstreams/FMT_cellreports/train_cellreports_loco_response_all_scales.sh
```

Set `COHORTS="cohort_a cohort_b"` to run a selected subset. Runs are stored by
scale, seed, and held-out cohort below `outputs_downstreams/FMT_cellreports/`.

## Donor-similarity evaluation

Run repeated random splits for MGM2:

```bash
DATA_DIR=data/downstreams/FMT_cellreports \
STAGE2_ROOT=/path/to/stage2 \
STAGE2_SCALES="scale_small scale_medium scale_large scale_xlarge" \
SPLITS="0 1 2 3 4 5 6 7 8 9" \
DEVICES="0" \
bash scripts/downstreams/FMT_cellreports/train_cellreports_similarity_random_all_scales.sh
```

Run the microbiome-only MOZAIC-style baseline on the identical split files:

```bash
DATA_DIR=data/downstreams/FMT_cellreports \
SPLITS="0 1 2 3 4 5 6 7 8 9" \
DEVICE=cuda:0 \
bash scripts/downstreams/FMT_cellreports/train_mozaic_similarity_random.sh
```

Aggregate metrics and generate comparison figures with:

```bash
python scripts/downstreams/FMT_cellreports/plot_cellreports_similarity_results.py \
  --results_dir outputs_downstreams/FMT_cellreports/community_similarity_random \
  --mozaic_results_dir outputs_downstreams/FMT_cellreports/mozaic_similarity_random \
  --output_dir figures/FMT_cellreports
```

## Checkpoint convention

All scale launchers load the stage-2 backbone from
`${STAGE2_ROOT}/${SCALE}/checkpoints/last.ckpt`. Downstream prediction defaults
to `EVAL_CHECKPOINT=best`, selected within the training partition. Model outputs
include prediction tables, metric JSON files, training logs, and run
configuration artifacts.
