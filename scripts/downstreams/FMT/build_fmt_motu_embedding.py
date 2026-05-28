#!/usr/bin/env python3
"""Aggregate marker-level FMT mOTU embeddings into one embedding per mOTU."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average fmt_motus_embeddings.h5 marker embeddings by mOTU prefix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source_embedding", type=Path, default=Path("ntv3_embeddings/fmt_motus_embeddings.h5"))
    parser.add_argument("--selected_motus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def decode_strings(values) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def motu_prefix(embedding_id: str) -> str:
    return str(embedding_id).split(".", 1)[0]


def main() -> None:
    args = parse_args()
    selected = pd.read_csv(args.selected_motus, sep="\t")
    if "mOTU.ID" not in selected.columns:
        raise KeyError(f"{args.selected_motus} must contain column 'mOTU.ID'")
    selected_ids = selected["mOTU.ID"].astype(str).tolist()
    selected_set = set(selected_ids)

    grouped: dict[str, list[np.ndarray]] = {motu_id: [] for motu_id in selected_ids}
    with h5py.File(args.source_embedding, "r") as handle:
        embedding_ids = decode_strings(handle["otu_ids"][:])
        embeddings = handle["embeddings"]
        for row_idx, embedding_id in enumerate(embedding_ids):
            prefix = motu_prefix(embedding_id)
            if prefix in selected_set:
                grouped[prefix].append(np.asarray(embeddings[row_idx], dtype=np.float32))

    missing = [motu_id for motu_id in selected_ids if not grouped[motu_id]]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} selected mOTUs have no embeddings in {args.source_embedding}: {preview}")

    averaged = np.stack([np.stack(grouped[motu_id], axis=0).mean(axis=0) for motu_id in selected_ids]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("embeddings", data=averaged, compression="gzip")
        handle.create_dataset("otu_ids", data=np.asarray(selected_ids, dtype=object), dtype=string_dtype)
        handle.attrs["source_embedding"] = str(args.source_embedding)
        handle.attrs["aggregation"] = "mean_by_motu_prefix"

    summary = {
        "source_embedding": str(args.source_embedding),
        "output": str(args.output),
        "n_selected_motus": len(selected_ids),
        "embedding_dim": int(averaged.shape[1]),
        "min_markers_per_motu": int(min(len(grouped[motu_id]) for motu_id in selected_ids)),
        "max_markers_per_motu": int(max(len(grouped[motu_id]) for motu_id in selected_ids)),
        "mean_markers_per_motu": float(np.mean([len(grouped[motu_id]) for motu_id in selected_ids])),
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
