<div align="center">

# MGM2 as a Unified Foundation Model for Microbiome World Exploration

MGM2 converts taxonomic composition and relative abundance into contextualized
microbial representations and one fixed-size community embedding.

<p>
  <a href="https://huggingface.co/LudensZhang/MGM2">
    <img alt="MGM2 on Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-MGM2-FFD21E?logo=huggingface&logoColor=000">
  </a>
  <a href="https://huggingface.co/InstaDeepAI/NTv3_650M_pre">
    <img alt="NTv3 on Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-NTv3-4C6EF5?logo=huggingface&logoColor=fff">
  </a>
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=fff">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Lightning-EE4C2C?logo=pytorch&logoColor=fff">
</p>

<p>
  <a href="#model">Model</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#two-input-routes">Input Routes</a> ·
  <a href="#extract-community-embeddings">Embedding Export</a>
</p>

</div>

<p align="center">
  <img src="pipeline.png" alt="MGM2 model pipeline" width="920">
</p>

---

## Model

MGM2 represents every nonzero microbial feature with two complementary signals:

| Signal | Role |
| --- | --- |
| **Microbial identity** | A 1,536-dimensional NTv3 representation derived from an OTU/ASV sequence or a taxonomy fallback prototype |
| **Community abundance** | The feature's relative abundance within the sample |

An abundance-aware Transformer contextualizes each microbial token against the
rest of the community. A learnable `[CLS]` token summarizes the complete sample
as one vector that can be used directly in downstream models.

MGM2 supports three community representations:

| Representation | Description | Recommended use |
| --- | --- | --- |
| `cls` | Final hidden state of the community `[CLS]` token | **Default general-purpose embedding** |
| `mean_pool` | Attention-mask-aware mean of contextualized microbial tokens | Token-balanced community summary |
| `qwen3_proj` | `[CLS]` projected through the alignment head | Models that use the aligned representation |

### Model family

All scales use the same input and inference interface.

| Model | Hidden size | Layers | Heads | Feed-forward size |
| --- | ---: | ---: | ---: | ---: |
| **MGM2-small** | 224 | 6 | 4 | 896 |
| **MGM2-medium** | 320 | 6 | 5 | 1,280 |
| **MGM2-large** | 384 | 8 | 6 | 1,536 |
| **MGM2-xlarge** | 512 | 10 | 8 | 2,048 |

> The exported `cls` and `mean_pool` dimensions equal the hidden size of the
> selected model.

## Quick Start

### 1. Install MGM2

```bash
git clone https://github.com/HUST-NingKang-Lab/MGM2.git
cd MGM2

conda create -n mgm2 python=3.10
conda activate mgm2
pip install torch pytorch-lightning numpy pandas scipy h5py tqdm transformers biopython huggingface_hub
```

### 2. Download the MGM2 release

The complete release is available at
[LudensZhang/MGM2](https://huggingface.co/LudensZhang/MGM2).

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LudensZhang/MGM2",
    local_dir="MGM2-release",
)
```

The downloaded directory contains all four model scales and shared embedding
resources:

```text
MGM2-release/
├── models/
│   ├── small/       # config.json + model.ckpt
│   ├── medium/
│   ├── large/
│   └── xlarge/
└── embeddings/
    ├── reference_ntv3_embeddings.h5
    └── taxonomy_fallback.h5
```

### 3. Prepare a community table

MGM2 accepts a feature-by-sample CSV or TSV table. The first column contains
feature identifiers, and each remaining column is one microbial community.
Values may be counts or abundances; each sample is converted to relative
abundance during input preparation.

```csv
feature_id,sample_A,sample_B
OTU_001,120,0
OTU_002,35,82
OTU_003,0,17
```

## Two Input Routes

Choose the route that matches the information available for your microbial
features.

| | **Sequence-aware** | **Taxonomy fallback** |
| --- | --- | --- |
| Input identity | OTU/ASV nucleotide sequence | Species name or taxonomy path |
| Identity encoder | [NTv3-650M](https://huggingface.co/InstaDeepAI/NTv3_650M_pre) | MGM2 taxonomy fallback |
| Best choice when | Representative sequences are available | Only taxonomic labels are available |
| Output | Feature-level identity HDF5 | Resolved taxonomy embedding HDF5 |

### Route A: OTU or ASV sequences

Download [InstaDeepAI/NTv3_650M_pre](https://huggingface.co/InstaDeepAI/NTv3_650M_pre):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="InstaDeepAI/NTv3_650M_pre",
    local_dir="NTv3_650M_pre",
)
```

FASTA record IDs must exactly match the feature IDs in the abundance table.

```fasta
>OTU_001
ACGT...
>OTU_002
TGCA...
```

Encode the sequences and organize the MGM2 input:

```bash
python ntv3_embedding.py \
  --fasta_path communities.fasta \
  --model_path NTv3_650M_pre \
  --output_path community_ntv3_embeddings.h5 \
  --gpu_ids 0

python src/data/build_dataset.py \
  --otu_table community_table.csv \
  --format csv \
  --ntv3_embedding community_ntv3_embeddings.h5 \
  --top_k_otus 768 \
  --output community_input.pkl
```

### Route B: Species names

Use the release fallback when representative sequences are unavailable. Feature
IDs may be bare species names, rank-prefixed names, or full taxonomy paths:

```csv
feature_id,sample_A,sample_B
Bacteroides fragilis,0.21,0.08
s__Faecalibacterium prausnitzii,0.09,0.17
g__Bifidobacterium,0.03,0.11
```

```bash
python src/data/build_dataset.py \
  --otu_table community_species.csv \
  --format csv \
  --ntv3_embedding MGM2-release/embeddings/reference_ntv3_embeddings.h5 \
  --fallback_embedding MGM2-release/embeddings/taxonomy_fallback.h5 \
  --resolved_embedding community_resolved_embeddings.h5 \
  --top_k_otus 768 \
  --output community_input.pkl
```

Name-only matching is rank aware and accepts only unambiguous names within a
taxonomic rank. A full rank-prefixed path is preferable when available.

> `community_resolved_embeddings.h5` records `embedding_source` and
> `matched_taxon_key` for every feature. The global fallback is used only when
> no exact ID, taxonomy path, or unambiguous taxon name can be resolved.

## Extract Community Embeddings

Both input routes produce the same `community_input.pkl` interface.

```bash
python scripts/extract_community_embeddings.py \
  --model_dir MGM2-release/models/small \
  --data_file community_input.pkl \
  --output community_embeddings.h5 \
  --embedding_strategy cls \
  --batch_size 64 \
  --device cuda:0
```

Change `models/small` to `models/medium`, `models/large`, or
`models/xlarge` to select another scale. The embedding resource path stored in
`community_input.pkl` is used automatically.

### Output formats

The format is selected by the output filename extension.

| Extension | Saved content |
| --- | --- |
| `.h5` / `.hdf5` | `embeddings`, `sample_ids`, and model provenance |
| `.npz` | Compressed embeddings, sample IDs, and provenance arrays |
| `.csv` | One sample per row with explicit embedding columns |

```python
import h5py

with h5py.File("community_embeddings.h5", "r") as handle:
    sample_ids = handle["sample_ids"].asstr()[:]
    embeddings = handle["embeddings"][:]

print(sample_ids.shape)
print(embeddings.shape)
```

For HDF5 output, `embeddings` has shape
`[number_of_samples, model_hidden_size]` and follows the order in
`sample_ids`.
