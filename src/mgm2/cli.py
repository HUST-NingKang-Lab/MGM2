"""Stream batches from a feature-by-sample CSV to a safe NumPy archive."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from . import MGM2, EmbeddingBank, prepare_sample, collate_samples, __version__


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--fallback", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if args.output.exists():
        p.error("Output already exists; choose a new path")
    with args.table.open(newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) < 2 or any(not name.strip() for name in header[1:]) or len(set(header[1:])) != len(header)-1:
        p.error("CSV sample identifiers must be nonempty and unique")
    try:
        import pandas as pd
    except ImportError:
        p.error("CSV support requires pandas: install 'mgm2-community[tables]'")
    table = pd.read_csv(args.table, index_col=0)
    if table.empty or table.index.has_duplicates or table.columns.has_duplicates:
        p.error("Expected a nonempty feature-by-sample CSV with unique identifiers")
    model = MGM2.from_pretrained(args.model, args.device)
    result = {k: [] for k in ("cls", "mean_pool", "qwen")}
    provenance = []
    with EmbeddingBank(args.embeddings, args.fallback) as bank, torch.inference_mode():
        for start in range(0, table.shape[1], args.batch_size):
            frame = table.iloc[:, start:start+args.batch_size]
            samples = [prepare_sample(table.index.astype(str), frame[col].to_numpy(), bank) for col in frame]
            outputs = model(collate_samples(samples, args.device))
            for key in result:
                result[key].append(outputs[key].cpu().numpy())
            for s in samples:
                provenance.append([sum(x == "ntv3" for x in s["resolution"]),
                                   sum(x.startswith("fallback:") and x != "fallback:global" for x in s["resolution"]),
                                   sum(x == "fallback:global" for x in s["resolution"])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, sample_ids=np.asarray(table.columns, dtype=str),
            model_scale=np.asarray(model.config["scale"], dtype=str),
            source_checkpoint_sha256=np.asarray(model.config["source_checkpoint_sha256"], dtype=str),
            resolution_counts=np.asarray(provenance, dtype=np.int64),
            resolution_columns=np.asarray(["ntv3", "taxonomy", "global"], dtype=str),
            **{k: np.concatenate(v) for k, v in result.items()})
    print(f"Saved {table.shape[1]} samples to {args.output}")


if __name__ == "__main__":
    main()
