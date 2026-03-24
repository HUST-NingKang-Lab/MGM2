# MGM2

MGM2 is a microbiome pretraining project built around OTU abundance modeling. The current codepath lives in [`src/`](./src) and focuses on masked abundance prediction rather than token identity reconstruction.

## What It Does

The current model uses NTv3 OTU embeddings with a Transformer backbone and supports:

- masked abundance-bin prediction
- masked log-abundance regression
- optional CLS-to-Qwen3 contrastive alignment at the sample level

Recent design choices in the active codepath:

- no legacy `presence` task
- masking corrupts only the abundance channel and preserves OTU identity
- `quantile` binning is performed independently within each sample
- regression supervision is configurable with `regression_top_k_bins`

## Repository Layout

```text
MGM2/
├── src/                         # Main implementation
│   ├── configs/                 # Strict config definitions
│   ├── data/                    # Dataset building, preprocessing, dataloaders
│   ├── losses/                  # Multi-task loss definitions
│   ├── models/                  # Backbone and prediction heads
│   ├── training/                # Lightning module and training entrypoint
│   ├── README.md
│   └── README_ANALYSIS.md
├── tests/                       # Regression tests for the active codepath
├── docs/repository_guide_zh.md  # Detailed Chinese repository guide
├── qwen3_keywords_embedding.py  # Build sample-level Qwen3 target embeddings
├── ntv3_embedding.py            # Build / inspect NTv3-related embeddings
└── scripts/validate_dataset.py  # Dataset validation utility
```

## What Is Not Included

This public repository intentionally excludes large local artifacts and non-essential local files, including:

- raw data and processed datasets
- NTv3 and Qwen3 embedding files
- checkpoints and training outputs
- legacy experimental code
- local planning notes and debug-only scripts

You need to prepare or mount these resources separately in your own environment.

## Quick Start

### 1. Build a preprocessed dataset

From a CSV OTU table:

```bash
python src/data/build_dataset.py   --otu_table data/otu_table.csv   --format csv   --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5   --output data/pretrain_dataset.pkl
```

From a BIOM OTU table:

```bash
python src/data/build_dataset.py   --otu_table data/otu_table.biom   --format biom   --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5   --output data/pretrain_dataset.pkl
```

### 2. Train the default dual-task model

```bash
python src/training/Train.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --config src/configs/mgm2_default.json
```

### 3. Enable optional Qwen3 alignment

```bash
python src/training/Train.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --enable_qwen3_alignment   --qwen3_embedding_path qwen3_embeddings/microbeatlas_keywords_emb.h5
```

## Core Configuration

The main config entry is [`src/configs/mgm2_default.json`](./src/configs/mgm2_default.json).

Important abundance-related settings:

- `binning_method`: `quantile` or `fixed_threshold`
- `regression_top_k_bins`: how many highest bins participate in regression supervision
- `num_bins`: number of abundance bins
- `min_abundance`, `output_min`, `output_max`: log-abundance preprocessing controls

Important optional alignment settings:

- `qwen3_alignment.enabled`
- `qwen3_alignment.embedding_path`
- `qwen3_alignment.projection_dim`
- `qwen3_alignment.temperature`
- `qwen3_alignment.loss_weight`

## Notes on Data Semantics

- dataset building stores raw relative abundance
- training-time preprocessing applies log transformation and clipping
- regression targets are preprocessed log-abundance values, not raw abundance
- Qwen3-alignment samples without matching `sample_id` are kept, but excluded from alignment loss

## Documentation

- [`src/README.md`](./src/README.md): package-level usage notes
- [`src/README_ANALYSIS.md`](./src/README_ANALYSIS.md): implementation notes and design analysis
- [`docs/repository_guide_zh.md`](./docs/repository_guide_zh.md): detailed Chinese guide to the full repository

## Verification

The active codepath is covered by regression tests in [`tests/test_mgm2_regressions.py`](./tests/test_mgm2_regressions.py).
