# Table inference

The CLI and Python table interface share the same preprocessing, batching and
export implementation. Install the `tables` extra to work with CSVs or pandas
DataFrames.

## Reuse a loaded model

```python
from mgm2 import MGM2, EmbeddingBank
from mgm2.data.tables import read_abundance_table
from mgm2.inference import embed_table, save_embeddings

model = MGM2.from_pretrained("MGM2-models/models/small", device="cpu")
table = read_abundance_table("communities.csv")
with EmbeddingBank("MGM2-models/embeddings/otus_99_embeddings.h5") as bank:
    result = embed_table(table, model, bank, batch_size=16)
save_embeddings(result, "outputs/communities.npz")
```

Keep the model and bank open while embedding additional tables. The input is a
feature-by-sample DataFrame; its index contains feature identifiers and its
columns contain sample identifiers. `embed_table` uses the model's device and
requires evaluation mode. Checkpoint loading already calls `eval()`; after
training or fine-tuning, call `model.eval()` explicitly before inference.

## Run a complete CSV job

```python
from mgm2.inference import embed_csv

sample_count = embed_csv(
    model_path="MGM2-models/models/small",
    table_path="communities.csv",
    embeddings_path="MGM2-models/embeddings/otus_99_embeddings.h5",
    output_path="outputs/communities.npz",
    batch_size=16,
    device="cpu",
)
```

Pass `fallback_path` to enable the matching taxonomy embedding bank.
`embed_csv` owns and closes its embedding-bank handle. `embed_table` leaves the
caller-owned bank open. Both run without gradients and preserve sample order.
The CSV and returned embedding arrays reside in host memory; only the active
inference batch is placed on the model device.

## Input and output contract

Feature and sample identifiers must be nonempty and unique. The CSV loader
preserves lexical identifiers such as `001` and `NA`, checks headers before
pandas can rename duplicate columns, and validates finite non-negative values
and positive sample totals before loading the model. Preprocessing normalizes
the complete community, selects up to 768 positive OTUs and floors positive
abundances below 1e-4 without renormalizing the retained subset.

The NumPy archive contains:

| Field | Content |
|---|---|
| `cls`, `mean_pool`, `qwen` | Sample-by-dimension embedding matrices |
| `sample_ids` | Ordered sample identifiers |
| `model_scale` | Model scale recorded in the checkpoint configuration |
| `source_checkpoint_sha256` | Source checkpoint identity |
| `resolution_counts` | Per-sample retained-feature counts by embedding source |
| `resolution_columns` | `ntv3`, `taxonomy`, `global`, in count-column order |

Archives use numeric and string arrays and can be opened with
`numpy.load(path, allow_pickle=False)`. Existing output files are never
overwritten. Input errors reported by the CLI exit with status 2; the Python
interface raises the underlying exception for callers to handle.
