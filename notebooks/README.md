# Interactive tutorial

[02_sequence_encoding.ipynb](02_sequence_encoding.ipynb) demonstrates FASTA
validation, the legacy NTv3 encoder and inspection of the resulting bank.
Encoding is opt-in because it loads the separately distributed NTv3 weights.

[01_getting_started.ipynb](01_getting_started.ipynb) covers input preparation,
Small community embeddings, contextual OTU representations, CSV export through
the CLI, and formal XLarge SAE features. It uses constructed abundances over
eight real OTU vocabulary entries. See [Installation](../docs/INSTALLATION.md)
for the environment and model bundle.

Set `MODEL_ROOT` to the bundle and use the **Python (MGM2)** kernel. CPU is the
default; `MGM2_DEVICE` selects another PyTorch device. `MGM2_OUTPUT_DIR` selects
the parent of per-run output folders. The notebook contains no automatic model
download and can run offline with the bundle available locally.

Committed notebooks contain source cells and empty outputs. Executed notebooks
and exported arrays belong in the chosen output directory.
