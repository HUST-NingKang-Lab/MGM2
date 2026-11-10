"""Batched table inference shared by the Python API and command line."""

from pathlib import Path

import numpy as np
import torch

from ..data import EmbeddingBank, collate_samples, prepare_sample
from ..data.tables import read_abundance_table, validate_table
from ..models import MGM2


def embed_table(table, model, bank, *, batch_size=16):
    """Embed a feature-by-sample DataFrame using an existing model and bank.

    The model must be in evaluation mode. Batches use the model's device;
    returned NumPy arrays reside on CPU and retain the input sample order.
    Pandas is needed by the caller to construct the table, not at module import.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if any(module.training for module in model.modules()):
        raise ValueError("Inference requires evaluation mode; call model.eval() first")
    validate_table(table)
    device = next(model.parameters()).device
    result = {key: [] for key in ("cls", "mean_pool", "qwen")}
    provenance = []
    feature_ids = table.index.astype(str).to_numpy()
    with torch.inference_mode():
        for start in range(0, table.shape[1], batch_size):
            frame = table.iloc[:, start : start + batch_size]
            samples = [prepare_sample(feature_ids, frame[col].to_numpy(), bank) for col in frame]
            outputs = model(collate_samples(samples, device))
            for key in result:
                result[key].append(outputs[key].cpu().numpy())
            for sample in samples:
                sources = sample["resolution"]
                provenance.append(
                    [
                        sum(source == "ntv3" for source in sources),
                        sum(
                            source.startswith("fallback:") and source != "fallback:global"
                            for source in sources
                        ),
                        sum(source == "fallback:global" for source in sources),
                    ]
                )
    return dict(
        sample_ids=np.asarray(table.columns, dtype=str),
        model_scale=np.asarray(model.config["scale"], dtype=str),
        source_checkpoint_sha256=np.asarray(model.config["source_checkpoint_sha256"], dtype=str),
        resolution_counts=np.asarray(provenance, dtype=np.int64),
        resolution_columns=np.asarray(["ntv3", "taxonomy", "global"], dtype=str),
        **{key: np.concatenate(chunks) for key, chunks in result.items()},
    )


def save_embeddings(result, output):
    """Write a NumPy archive readable with allow_pickle=False; never overwrite."""
    if any(np.asarray(value).dtype.hasobject for value in result.values()):
        raise ValueError("Embedding archives cannot contain object arrays")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **result)


def embed_csv(
    model_path,
    table_path,
    embeddings_path,
    output_path,
    *,
    fallback_path=None,
    batch_size=16,
    device="cpu",
):
    """Load resources, embed a CSV, save its archive, and return the sample count.

    The CSV and returned arrays are held in host memory; GPU inference is batched.
    Embedding-bank handles are closed on success and on failure.
    """
    if Path(output_path).exists():
        raise FileExistsError("Output already exists; choose a new path")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    table = read_abundance_table(table_path)
    model = MGM2.from_pretrained(model_path, device)
    with EmbeddingBank(embeddings_path, fallback_path) as bank:
        result = embed_table(table, model, bank, batch_size=batch_size)
    save_embeddings(result, output_path)
    return len(result["sample_ids"])
