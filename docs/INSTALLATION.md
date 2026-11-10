# Installation

MGM2 uses Python 3.10 or later. The Python distribution is named
`mgm2-community`; its import namespace is `mgm2`. Model weights and embedding
banks are distributed separately through Hugging Face.

## Create an environment

```bash
git clone https://github.com/HUST-NingKang-Lab/MGM2.git
cd MGM2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
For GPU inference, install the PyTorch build matching your platform using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
before installing MGM2. Device selection remains explicit in the API and CLI.

## Install from the source checkout

Run these commands in the directory containing `pyproject.toml`:

```bash
python -m pip install '.[tables,hub]'
python -m pip check
mgm2-embed --version
mgm2-embed --help
```

For the interactive tutorial:

```bash
python -m pip install '.[notebook]'
python -m ipykernel install --user --name mgm2 --display-name "Python (MGM2)"
python -m jupyterlab notebooks/01_getting_started.ipynb
```

Select the **Python (MGM2)** kernel. The tutorial uses the same installed package
as the command-line interface.

## Model resources

Download the model bundle accompanying this code release:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="LudensZhang/MGM2", revision="v2.0.0", local_dir="../MGM2-models")'
export MODEL_ROOT="$(cd ../MGM2-models && pwd)"
```

The directory contains `models/{small,medium,large,xlarge}/`,
`embeddings/otus_99_embeddings.h5`, the taxonomy fallback bank, and
`sae/otu_l10_k64/`. Each model directory has `config.json` and
`model.safetensors`. Configurations identify the five-positive-bin format and
the original checkpoint SHA256. The tutorial loads Small for community
representations and XLarge for the formal SAE.

For an existing download, set `MODEL_ROOT` to its absolute path. The release tag
pins the model files independently of later updates to the Hub's default branch.

In the paired local release layout (`github/` beside `huggingface/`), use
`export MODEL_ROOT=../huggingface` from the code root. The notebook also discovers
this layout automatically. User-provided ASVs require their corresponding
NTv3 embedding bank; the supplied bank contains the release OTU vocabulary.

## Optional dependencies

| Extra | Purpose |
|---|---|
| `tables` | Feature-by-sample CSV input |
| `hub` | Programmatic Hugging Face downloads |
| `notebook` | JupyterLab, tutorial execution and visualization |
| `test` | Unit tests |
| `dev` | Tests, linting, distribution builds and metadata checks |

## Development installation

```bash
python -m pip install -e '.[tables,notebook,dev]'
python -m pytest
python -m ruff check src tests examples
python -m build
python -m twine check dist/*
```

The editable installation exposes `src/mgm2`; ordinary installation uses a
built wheel. See [development and release checks](DEVELOPMENT.md) for artifact
testing and package publication.
