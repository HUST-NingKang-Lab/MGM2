# MGM2: A Foundation Model for Microbial Communities

MGM2 encodes a microbial community into contextualized microbial-token representations and a single community-level embedding. It combines sequence-derived NTv3 representations of microbial identity with relative-abundance signals in an abundance-aware Transformer.

![MGM2 model overview](pipeline.png)

## Model

Each nonzero microbial feature is represented by:

- a 1,536-dimensional NTv3 embedding describing microbial identity;
- its relative abundance in the community; and
- contextual information learned from the other members of the same community.

A learnable `[CLS]` token aggregates the complete community into one fixed-size vector. MGM2 is trained with masked abundance-bin classification and masked log-abundance regression. Only abundance values are masked; microbial identity remains visible so that the model learns community context and abundance structure.

The exported community representations are:

- `cls`: the `[CLS]` hidden state. This is the recommended general-purpose MGM2 embedding.
- `mean_pool`: the attention-mask-aware mean of contextualized microbial tokens, excluding `[CLS]`.
- `qwen3_proj`: the projected `[CLS]` representation from models released with a Qwen3 alignment head.

## Model Sizes

All MGM2 sizes share the same input representation and output interface.

| Model | Hidden size | Transformer layers | Attention heads | Feed-forward size |
| --- | ---: | ---: | ---: | ---: |
| MGM2-small | 224 | 6 | 4 | 896 |
| MGM2-medium | 320 | 6 | 5 | 1,280 |
| MGM2-large | 384 | 8 | 6 | 1,536 |
| MGM2-xlarge | 512 | 10 | 8 | 2,048 |

The dimension of a `cls` or `mean_pool` embedding is the hidden size of the selected model.

## Installation

MGM2 requires Python, PyTorch, PyTorch Lightning, NumPy, pandas, SciPy, h5py, tqdm, Transformers, and Biopython. A minimal environment can be created with:

```bash
conda create -n mgm2 python=3.10
conda activate mgm2
pip install torch pytorch-lightning numpy pandas scipy h5py tqdm transformers biopython huggingface_hub
```

Run the commands below from the repository root so that the local `src` package is importable.

## Download a Model

MGM2 weights and their matching model resources will be published on Hugging Face. Replace the placeholder with the repository ID from the selected model card:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="<HUGGINGFACE_REPO_ID>",
    local_dir="models/MGM2-small",
)
```

The commands below assume this model bundle layout:

```text
models/MGM2-small/
├── config.json
├── model.ckpt
├── reference_ntv3_embeddings.h5
└── taxonomy_fallback.h5
```

`last.ckpt` and `checkpoints/last.ckpt` are also recognized automatically in place of `model.ckpt`.

## Community Input

MGM2 accepts a feature-by-sample CSV or TSV table. The first column contains feature identifiers; each remaining column is one community. Values may be counts or abundances because the input builder converts each sample to relative abundance.

```csv
feature_id,sample_A,sample_B
OTU_001,120,0
OTU_002,35,82
OTU_003,0,17
```

There are two supported ways to assign microbial identity embeddings.

## Option A: OTU or ASV Sequences

Use this route when representative nucleotide sequences are available. Every FASTA record ID must exactly match the corresponding row identifier in the community table.

```fasta
>OTU_001
ACGT...
>OTU_002
TGCA...
```

First encode the sequences with NTv3:

```bash
python ntv3_embedding.py \
  --fasta_path communities.fasta \
  --model_path /path/to/NTv3_650M_pre \
  --output_path community_ntv3_embeddings.h5 \
  --gpu_ids 0
```

Then organize the abundance table as MGM2 input:

```bash
python src/data/build_dataset.py \
  --otu_table community_table.csv \
  --format csv \
  --ntv3_embedding community_ntv3_embeddings.h5 \
  --top_k_otus 768 \
  --output community_input.pkl
```

`--top_k_otus 768` keeps the 768 most abundant nonzero features in each community. Use `--top_k_otus 0` to keep all nonzero features.

## Option B: Species Names

Use this route when representative sequences are unavailable and the table contains taxonomic names. Bare or rank-prefixed species names are accepted:

```csv
feature_id,sample_A,sample_B
Bacteroides fragilis,0.21,0.08
s__Faecalibacterium prausnitzii,0.09,0.17
g__Bifidobacterium,0.03,0.11
```

Resolve those names with the taxonomy fallback resource distributed with the selected MGM2 model:

```bash
python src/data/build_dataset.py \
  --otu_table community_species.csv \
  --format csv \
  --ntv3_embedding models/MGM2-small/reference_ntv3_embeddings.h5 \
  --fallback_embedding models/MGM2-small/taxonomy_fallback.h5 \
  --resolved_embedding community_resolved_embeddings.h5 \
  --top_k_otus 768 \
  --output community_input.pkl
```

Name-only matching is rank aware and accepts only an unambiguous name within a taxonomic rank. A full rank-prefixed path is preferable when available:

```text
d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides;s__Bacteroides fragilis
```

`community_resolved_embeddings.h5` records `embedding_source` and `matched_taxon_key` for every resolved feature. Exact NTv3 identifiers are marked `ntv3`; name matches are marked, for example, `fallback:species_name`. The global fallback is used only when no exact identifier, taxonomy path, or unambiguous taxon name can be resolved.

## Extract Community Embeddings

Both input routes produce the same `community_input.pkl` interface. Extract unlabeled MGM2 embeddings with:

```bash
python scripts/extract_community_embeddings.py \
  --model_dir models/MGM2-small \
  --data_file community_input.pkl \
  --output community_embeddings.h5 \
  --embedding_strategy cls \
  --batch_size 64 \
  --device cuda:0
```

The embedding resource path stored in `community_input.pkl` is used automatically. Pass `--ntv3_embedding_path` only when that resource has moved.

To use an explicit checkpoint instead of a model directory:

```bash
python scripts/extract_community_embeddings.py \
  --checkpoint /path/to/model.ckpt \
  --config /path/to/config.json \
  --data_file community_input.pkl \
  --output community_embeddings.npz \
  --embedding_strategy mean_pool \
  --device cuda:0
```

The output format is selected by extension:

| Extension | Contents |
| --- | --- |
| `.h5` or `.hdf5` | `embeddings`, `sample_ids`, and model provenance attributes |
| `.npz` | compressed `embeddings`, `sample_ids`, and provenance arrays |
| `.csv` | one row per sample with `sample_id` and embedding columns |

For HDF5 output, `embeddings` has shape `[number_of_samples, hidden_size]` and follows the order in `sample_ids`.

## Load the Result

```python
import h5py

with h5py.File("community_embeddings.h5", "r") as handle:
    sample_ids = handle["sample_ids"].asstr()[:]
    embeddings = handle["embeddings"][:]

print(sample_ids.shape)
print(embeddings.shape)
```

## Citation

If you use MGM2 in your research, please cite the accompanying paper. Complete citation metadata will be added with the model release.
