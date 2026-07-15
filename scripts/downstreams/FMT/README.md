# FMT Prediction

This workflow evaluates MGM2 on fecal microbiota transplantation (FMT). A model
receives paired donor and recipient pre-FMT communities and predicts either the
recipient post-FMT community or a clinical response label.

## Inputs

The preparation scripts expect the FMT metadata, sample metadata, mOTU
taxonomy, and abundance tables in one raw-data directory. The main files are:

```text
fmt_metadata.csv
sample_metadata.csv
mOTU_taxonomy.csv
mOTU_rel_abundance.csv
mOTU_count_abundance.csv
phenotype.csv
```

The standard cross-validation workflow uses an all-event community dataset,
event-level metadata, and an mOTU embedding HDF5 file under
`data/downstreams/FMT_5folds/`.

## Dataset summary

Generate descriptive tables and figures directly from a configurable raw-data
location:

```bash
python scripts/downstreams/FMT/analyze_fmt_dataset.py \
  --data_dir /path/to/fmt_data \
  --output_dir outputs_downstreams/FMT/dataset_eval
```

## Build the five-fold dataset

The build selects mOTUs, prepares donor-recipient event tables, constructs a
shared mOTU embedding file, and serializes the MGM2 community dataset.

```bash
DATA_DIR=/path/to/fmt_data \
SOURCE_MOTU_EMBEDDING=/path/to/fmt_motus_embeddings.h5 \
OUTPUT_DIR=data/downstreams/FMT_5folds \
MAX_MOTUS=1024 \
bash scripts/downstreams/FMT/build_mgm2_fmt_5fold_dataset.sh
```

Key outputs are `community_all.pkl`, `community_all.motu_embeddings.h5`,
`response_all_meta.csv`, `post_all_meta.csv`, and the preparation summary JSON.
The fold utilities group events by `fmt_id` and support stratification by an
event metadata column such as `indication`.

## Train MGM2

Run post-FMT community prediction across model scales:

```bash
STAGE2_ROOT=/path/to/stage2 \
STAGE2_SCALES="scale_small scale_medium scale_large scale_xlarge" \
DEVICES="0 1" \
SEEDS="0" \
FOLDS="0 1 2 3 4" \
bash scripts/downstreams/FMT/train_fmt_community_5folds_all_scales.sh
```

Run response classification on the same outer folds:

```bash
STAGE2_ROOT=/path/to/stage2 \
STAGE2_SCALES="scale_small scale_medium scale_large scale_xlarge" \
DEVICES="0 1" \
SEEDS="0" \
FOLDS="0 1 2 3 4" \
bash scripts/downstreams/FMT/train_fmt_classification_5folds_all_scales.sh
```

Both launchers initialize the frozen backbone from each scale's stage-2
`checkpoints/last.ckpt`. `EVAL_CHECKPOINT=best` selects the best downstream
checkpoint for fold predictions; set it explicitly when reproducing a run.

## Baselines

The abundance MLP and response-classification MLP use the same prepared data
and grouped folds:

```bash
DATA_DIR=data/downstreams/FMT_5folds \
SEEDS="0" FOLDS="0 1 2 3 4" DEVICE=cuda:0 \
bash scripts/downstreams/FMT/train_mlp_community_5folds.sh

DATA_DIR=data/downstreams/FMT_5folds \
SEEDS="0" FOLDS="0 1 2 3 4" DEVICE=cuda:0 \
bash scripts/downstreams/FMT/train_mlp_classification_5folds.sh
```

A microbiome-only MOZAIC-style response baseline is available through:

```bash
DATA_DIR=data/downstreams/FMT_5folds \
SEEDS="0" FOLDS="0 1 2 3 4" DEVICE=cuda:0 \
bash scripts/downstreams/FMT/train_mozaic_classification_5folds.sh
```

## Evaluation outputs

MGM2 fold outputs are organized by scale, seed, and fold. Community prediction
writes abundance predictions and per-sample distance metrics; classification
writes class probabilities and classification metrics. Baseline outputs follow
the same seed/fold organization below `outputs_downstreams/FMT_5folds/`.

Create the aggregate tables and figures after all requested folds finish:

```bash
python scripts/downstreams/FMT/plot_fmt_5fold_results.py
```

Use `python scripts/downstreams/FMT/plot_fmt_5fold_results.py --help` to override
result roots, metadata paths, output directories, and plotting options.
