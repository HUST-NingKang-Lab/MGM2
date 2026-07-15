# Sparse Autoencoder Analysis

These scripts train BatchTopK sparse autoencoders (SAEs) on hidden states from
the MGM2 community encoder. The workflow supports CLS and OTU-token
representations; the large-scale workflows focus on OTU tokens from
MGM2-xlarge.

## Workflow

The pipeline has four stages:

1. Select pretraining samples and create stable train/validation/test manifests.
2. Extract MGM2 hidden states for the selected tokens.
3. Train or grid-search BatchTopK SAEs.
4. Analyze feature activation patterns and generate summary figures.

Run all commands from the repository root in the `mwm` environment.

## 1. Build a sampling manifest

The Python entry point supports both compact studies and configurable
large-sample extraction. For example:

```bash
python scripts/sae/build_pretrain_sae_sampling_manifest.py \
  --data_file /path/to/preprocessed_microbiome.pkl \
  --metadata_file /path/to/sample_metadata.tsv \
  --output_dir data/sae/pretrain_xlarge_sample500k \
  --animal_quota 100000 \
  --aquatic_quota 100000 \
  --soil_quota 100000 \
  --plant_quota 100000 \
  --missing_quota 60000 \
  --wgs_supplement 40000 \
  --val_fraction 0.05 \
  --test_fraction 0.05 \
  --seed 42
```

The manifest builder performs environment-aware sampling and writes
`samples.csv`, compressed token metadata, split assignments, and summary files.
The quota values above illustrate a large-sample run and can be adjusted to the
available corpus. Use `--quota_scale` for proportional scaling or set individual
group quotas and split fractions explicitly.

## 2. Extract hidden states

```bash
python scripts/sae/extract_mgm2_sae_activations.py \
  --data_file /path/to/preprocessed_microbiome.pkl \
  --manifest_dir data/sae/pretrain_xlarge_sample500k \
  --stage2_root /path/to/stage2 \
  --scale scale_xlarge \
  --ntv3_embedding /path/to/otus_99_embeddings.h5 \
  --token_type otu \
  --output data/sae/pretrain_xlarge_sample500k/mgm2_scale_xlarge_last_hidden_otu_activations.h5 \
  --device cuda:0 \
  --batch_size 4 \
  --max_batch_tokens 4096 \
  --dtype float16
```

`--token_type` accepts `all`, `otu`, or `cls`. By default the extractor loads
the selected scale's `config.json` and stage-2 `checkpoints/last.ckpt`; explicit
`--config` and `--checkpoint` paths can be supplied when using a different
artifact layout. For large manifests, tune `--batch_size` and
`--max_batch_tokens` together, and use `--resume` to continue an existing HDF5
extraction.

Activation HDF5 files include token type, split, sample row, OTU index,
abundance, rank, and hidden-state metadata.

## 3. Train BatchTopK SAEs

Run one configuration:

```bash
ACTIVATIONS=data/sae/pretrain_xlarge_sample500k/mgm2_scale_xlarge_last_hidden_otu_activations.h5 \
TOKEN_TYPE=otu \
DICT_SIZE=4096 \
TOP_K=32 \
DEVICE=cuda:0 \
OUTPUT_DIR=outputs_sae/mgm2_xlarge_otu_k32 \
bash scripts/sae/train_mgm2_xlarge_otu_batchtopk_sae.sh
```

Run the OTU-token grid:

```bash
ACTIVATIONS=data/sae/pretrain_xlarge_sample500k/mgm2_scale_xlarge_last_hidden_otu_activations.h5 \
TOP_K_GRID="8 16 32 64 128" \
DICT_SIZE=4096 \
DEVICE=cuda:0 \
OUTPUT_ROOT=outputs_sae/mgm2_xlarge_otu_sample500k_batchtopk_grid_d4096 \
bash scripts/sae/grid_mgm2_xlarge_otu_sample500k_batchtopk_sae.sh
```

Each run stores `best.pt`, periodic checkpoints, `metrics.csv`, a configuration
summary, and a training log. Validation loss selects `best.pt`.

## 4. Summarize and analyze features

Plot grid-search reconstruction and feature-usage metrics:

```bash
python scripts/sae/plot_mgm2_xlarge_otu_sae_grid.py \
  --grid-dir outputs_sae/mgm2_xlarge_otu_sample500k_batchtopk_grid_d4096 \
  --output-dir figures/sae/otu_grid
```

Feature statistics can be computed in shards and merged deterministically:

```bash
SHARD_COUNT=4 DEVICE=cuda:0 \
bash scripts/sae/run_mgm2_xlarge_otu_sample500k_sae_feature_shards.sh
```

The merged analysis writes feature firing statistics, environment and
technology summaries, embeddings, and figures below `outputs_sae_analysis/`
and `figures/sae/`. See [SAE interpretation](../sae_interpretation/README.md)
for evidence packaging and biological annotation.
