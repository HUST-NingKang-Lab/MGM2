#!/usr/bin/env python3
"""Extract MGM2 backbone activations for the SAE pretraining manifest."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

DEFAULT_DATA_FILE = Path("data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.pkl")
DEFAULT_MANIFEST_DIR = Path("data/sae/pretrain_xlarge_sample50k")
DEFAULT_STAGE2_ROOT = Path("outputs_grid/stage2/20260505_094404")
DEFAULT_NTV3 = Path("ntv3_embeddings/otus_99_embeddings.h5")
DEFAULT_OUTPUT = Path("data/sae/pretrain_xlarge_sample50k/mgm2_scale_xlarge_last_hidden_activations.h5")

TOKEN_TYPE_TO_ID = {"cls": 0, "otu": 1}
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--manifest_dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--samples_csv", type=Path, default=None)
    parser.add_argument("--stage2_root", type=Path, default=DEFAULT_STAGE2_ROOT)
    parser.add_argument("--scale", type=str, default="scale_xlarge")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ntv3_embedding", type=Path, default=DEFAULT_NTV3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=4096,
        help="Upper bound for batch_size * max_seq_len. Set <=0 to only use --batch_size.",
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--token_type", choices=["all", "otu", "cls"], default="all")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="lzf")
    parser.add_argument("--chunk_tokens", type=int, default=8192)
    parser.add_argument("--limit_samples", type=int, default=0, help="Debug: only extract the first N selected samples")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output HDF5 using written sample metadata")
    return parser.parse_args()


def _load_pretrain_data(path: Path) -> Dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_samples(path: Path, splits: Sequence[str], token_type: str, limit_samples: int) -> List[Dict]:
    selected_splits = set(splits)
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] not in selected_splits:
                continue
            selected_count = int(row["selected_token_count"])
            if token_type == "otu":
                selected_count -= 1
            elif token_type == "cls":
                selected_count = 1
            if selected_count <= 0:
                continue
            copied = dict(row)
            copied["sample_index"] = int(row["sample_index"])
            copied["nonzero_otu_count"] = int(row["nonzero_otu_count"])
            copied["selected_token_count"] = selected_count
            rows.append(copied)
            if limit_samples > 0 and len(rows) >= limit_samples:
                break
    if not rows:
        raise ValueError("no samples selected for activation extraction")
    return rows


def _expected_num_tokens(sample_rows: Sequence[Dict]) -> int:
    return int(sum(row["selected_token_count"] for row in sample_rows))


def _read_existing_written_tokens(path: Path) -> tuple[int, tuple[int, ...] | None]:
    if not path.exists():
        return 0, None
    with h5py.File(path, "r") as handle:
        if "activations" not in handle:
            return 0, None
        return int(handle.attrs.get("written_tokens", 0)), tuple(handle["activations"].shape)


def _load_ntv3_embeddings(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["embeddings"][:].astype(np.float32)


def _token_rows_for_sample(data: Dict, sample_index: int, token_type: str):
    input_ids = np.asarray(data["input_ids"][sample_index], dtype=np.int64)
    attention_mask = np.asarray(data["attention_mask"][sample_index], dtype=np.int64)
    processed_abundance = np.asarray(data["abundance_values"][sample_index], dtype=np.float32)
    valid_otu_positions = np.flatnonzero((attention_mask == 1) & (input_ids != -1))

    rows = []
    if token_type in {"all", "cls"}:
        rows.append(
            {
                "token_type": "cls",
                "token_position": 0,
                "otu_index": -1,
                "processed_abundance": 0.0,
                "abundance": 0.0,
                "abundance_rank": 0,
            }
        )
    if token_type in {"all", "otu"} and len(valid_otu_positions):
        relative_abundance = np.power(10.0, processed_abundance[valid_otu_positions] - 2.0).astype(np.float32)
        order_desc = sorted(range(len(valid_otu_positions)), key=lambda idx: (-float(relative_abundance[idx]), int(valid_otu_positions[idx])))
        ranks = {local_idx: rank for rank, local_idx in enumerate(order_desc, start=1)}
        for local_idx, position in enumerate(valid_otu_positions):
            rows.append(
                {
                    "token_type": "otu",
                    "token_position": int(position),
                    "otu_index": int(input_ids[position]),
                    "processed_abundance": float(processed_abundance[position]),
                    "abundance": float(relative_abundance[local_idx]),
                    "abundance_rank": int(ranks[local_idx]),
                }
            )
    return rows


def _build_batch(data: Dict, sample_rows: Sequence[Dict], ntv3_embeddings: np.ndarray):
    seq_lengths = [len(data["input_ids"][row["sample_index"]]) for row in sample_rows]
    max_len = max(seq_lengths)
    batch_size = len(sample_rows)
    embedding_dim = ntv3_embeddings.shape[1]
    input_ids_batch = np.full((batch_size, max_len), -1, dtype=np.int64)
    attention_mask_batch = np.zeros((batch_size, max_len), dtype=np.int64)
    abundance_batch = np.zeros((batch_size, max_len), dtype=np.float32)
    embeddings_batch = np.zeros((batch_size, max_len, embedding_dim), dtype=np.float32)

    for batch_idx, row in enumerate(sample_rows):
        sample_index = row["sample_index"]
        input_ids = np.asarray(data["input_ids"][sample_index], dtype=np.int64)
        attention_mask = np.asarray(data["attention_mask"][sample_index], dtype=np.int64)
        abundance = np.asarray(data["abundance_values"][sample_index], dtype=np.float32)
        seq_len = len(input_ids)
        input_ids_batch[batch_idx, :seq_len] = input_ids
        attention_mask_batch[batch_idx, :seq_len] = attention_mask
        abundance_batch[batch_idx, :seq_len] = abundance
        valid_otu = (attention_mask == 1) & (input_ids != -1)
        if valid_otu.any():
            embeddings_batch[batch_idx, np.flatnonzero(valid_otu), :] = ntv3_embeddings[input_ids[valid_otu]]
    return input_ids_batch, attention_mask_batch, abundance_batch, embeddings_batch


def _iter_dynamic_batches(data: Dict, sample_rows: Sequence[Dict], start_row: int, batch_size: int, max_batch_tokens: int):
    idx = start_row
    while idx < len(sample_rows):
        batch = []
        max_len = 0
        while idx < len(sample_rows) and len(batch) < batch_size:
            row = sample_rows[idx]
            seq_len = len(data["input_ids"][row["sample_index"]])
            proposed_max_len = max(max_len, seq_len)
            proposed_batch_size = len(batch) + 1
            if batch and max_batch_tokens > 0 and proposed_batch_size * proposed_max_len > max_batch_tokens:
                break
            batch.append((idx, row))
            max_len = proposed_max_len
            idx += 1
        if not batch:
            row = sample_rows[idx]
            batch.append((idx, row))
            idx += 1
        yield batch


def _create_output(path: Path, num_tokens: int, hidden_size: int, dtype: str, compression: str, chunk_tokens: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    h5_dtype = np.float16 if dtype == "float16" else np.float32
    compression_arg = None if compression == "none" else compression
    handle = h5py.File(path, "w")
    handle.create_dataset(
        "activations",
        shape=(num_tokens, hidden_size),
        dtype=h5_dtype,
        chunks=(min(chunk_tokens, num_tokens), hidden_size),
        compression=compression_arg,
    )
    for name, dtype_obj in [
        ("sample_row", np.int32),
        ("sample_index", np.int64),
        ("token_position", np.int32),
        ("token_type", np.int8),
        ("otu_index", np.int64),
        ("split", np.int8),
        ("abundance_rank", np.int32),
    ]:
        handle.create_dataset(name, shape=(num_tokens,), dtype=dtype_obj, chunks=(min(chunk_tokens, num_tokens),), compression=compression_arg)
    for name in ("abundance", "processed_abundance"):
        handle.create_dataset(name, shape=(num_tokens,), dtype=np.float32, chunks=(min(chunk_tokens, num_tokens),), compression=compression_arg)
    return handle


def _open_output_for_resume(path: Path, expected_tokens: int, hidden_size: int):
    if not path.exists():
        raise FileNotFoundError(f"cannot resume because output does not exist: {path}")
    handle = h5py.File(path, "r+")
    if "activations" not in handle:
        handle.close()
        raise ValueError(f"output has no activations dataset: {path}")
    if tuple(handle["activations"].shape) != (expected_tokens, hidden_size):
        shape = tuple(handle["activations"].shape)
        handle.close()
        raise ValueError(f"output shape {shape} does not match expected {(expected_tokens, hidden_size)}")
    return handle


def _resume_offsets(handle, sample_rows: Sequence[Dict]) -> tuple[int, int]:
    written_tokens = int(handle.attrs.get("written_tokens", 0))
    if written_tokens <= 0:
        return 0, 0
    if written_tokens > handle["activations"].shape[0]:
        raise ValueError(f"written_tokens exceeds dataset size: {written_tokens}")
    sample_counts = [int(row["selected_token_count"]) for row in sample_rows]
    total = 0
    for sample_row, count in enumerate(sample_counts):
        if total + count > written_tokens:
            if total != written_tokens:
                raise ValueError(
                    f"cannot resume from partial sample: written_tokens={written_tokens}, "
                    f"last complete token offset={total}"
                )
            return sample_row, total
        total += count
        if total == written_tokens:
            return sample_row + 1, total
    if total != written_tokens:
        raise ValueError(f"written_tokens={written_tokens} does not align with selected samples total={total}")
    return len(sample_rows), total


def _write_sample_table(handle, sample_rows: Sequence[Dict], compression: str):
    string_dtype = h5py.string_dtype("utf-8")
    group = handle.create_group("samples")
    for name in ("sample_id", "split", "quota_group", "primary_env", "env_first2", "technology", "project", "institution", "keywords"):
        values = [str(row.get(name, "")) for row in sample_rows]
        group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=string_dtype)
    compression_arg = None if compression == "none" else compression
    for name, dtype_obj in [("sample_index", np.int64), ("nonzero_otu_count", np.int32), ("selected_token_count", np.int32)]:
        group.create_dataset(name, data=np.asarray([int(row[name]) for row in sample_rows], dtype=dtype_obj), compression=compression_arg)


def main():
    args = parse_args()
    samples_csv = args.samples_csv or (args.manifest_dir / "samples.csv")
    config_path = args.config or (args.stage2_root / args.scale / "config.json")
    checkpoint_path = args.checkpoint or (args.stage2_root / args.scale / "checkpoints" / "last.ckpt")
    if not samples_csv.exists():
        raise FileNotFoundError(f"samples.csv not found: {samples_csv}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    sample_rows = _load_samples(samples_csv, args.splits, args.token_type, args.limit_samples)
    num_tokens = _expected_num_tokens(sample_rows)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    print(f"Selected samples: {len(sample_rows)}")
    print(f"Selected tokens: {num_tokens}")
    print(f"Device: {device}")

    if args.resume:
        existing_written, existing_shape = _read_existing_written_tokens(args.output)
        if existing_shape is not None and existing_written == num_tokens:
            print(f"Output already complete: {args.output}")
            print(f"written_tokens={existing_written}, shape={existing_shape}")
            return

    print(f"Loading data: {args.data_file}")
    data = _load_pretrain_data(args.data_file)
    print(f"Loading NTv3 embeddings: {args.ntv3_embedding}")
    ntv3_embeddings = _load_ntv3_embeddings(args.ntv3_embedding)
    print(f"Loading MGM2 backbone: {checkpoint_path}")
    from scripts.downstreams.wastewater.extract_mgm2_cls_cache import _load_backbone

    backbone, config = _load_backbone(config_path, checkpoint_path, device)
    hidden_size = int(config.model.hidden_size)

    output = args.output
    if args.resume:
        handle = _open_output_for_resume(output, num_tokens, hidden_size)
        start_sample_row, write_pos = _resume_offsets(handle, sample_rows)
        print(f"Resuming from sample_row={start_sample_row}, token_offset={write_pos}")
    else:
        handle = _create_output(output, num_tokens, hidden_size, args.dtype, args.compression, args.chunk_tokens)
        _write_sample_table(handle, sample_rows, args.compression)
        handle.attrs["data_file"] = str(args.data_file)
        handle.attrs["samples_csv"] = str(samples_csv)
        handle.attrs["config"] = str(config_path)
        handle.attrs["checkpoint"] = str(checkpoint_path)
        handle.attrs["ntv3_embedding"] = str(args.ntv3_embedding)
        handle.attrs["scale"] = args.scale
        handle.attrs["hidden_size"] = hidden_size
        handle.attrs["activation_dtype"] = args.dtype
        handle.attrs["token_type_map"] = json.dumps(TOKEN_TYPE_TO_ID, sort_keys=True)
        handle.attrs["split_map"] = json.dumps(SPLIT_TO_ID, sort_keys=True)
        handle.attrs["max_batch_tokens"] = int(args.max_batch_tokens)
        write_pos = 0
        start_sample_row = 0

    if start_sample_row >= len(sample_rows):
        handle.close()
        print("Output already complete; nothing to do.")
        return

    try:
        total_remaining = len(sample_rows) - start_sample_row
        progress = tqdm(total=total_remaining, desc="extract", initial=0)
        for indexed_batch in _iter_dynamic_batches(data, sample_rows, start_sample_row, args.batch_size, args.max_batch_tokens):
            batch_indices = [item[0] for item in indexed_batch]
            batch_rows = [item[1] for item in indexed_batch]
            input_ids, attention_mask, abundance, embeddings = _build_batch(data, batch_rows, ntv3_embeddings)
            with torch.no_grad():
                outputs = backbone(
                    input_ids=torch.from_numpy(input_ids).to(device),
                    embeddings=torch.from_numpy(embeddings).to(device),
                    abundance_values=torch.from_numpy(abundance).to(device),
                    attention_mask=torch.from_numpy(attention_mask).to(device),
                )
            hidden = outputs["last_hidden_state"].detach().cpu().numpy()

            activation_chunks = []
            sample_row_values = []
            sample_index_values = []
            token_position_values = []
            token_type_values = []
            otu_index_values = []
            split_values = []
            abundance_values = []
            processed_abundance_values = []
            abundance_rank_values = []

            for local_idx, row in enumerate(batch_rows):
                token_rows = _token_rows_for_sample(data, row["sample_index"], args.token_type)
                if not token_rows:
                    continue
                positions = [token_row["token_position"] for token_row in token_rows]
                activation_chunks.append(hidden[local_idx, positions, :])
                n = len(token_rows)
                sample_row_values.extend([batch_indices[local_idx]] * n)
                sample_index_values.extend([row["sample_index"]] * n)
                split_values.extend([SPLIT_TO_ID[row["split"]]] * n)
                token_position_values.extend(positions)
                token_type_values.extend([TOKEN_TYPE_TO_ID[token_row["token_type"]] for token_row in token_rows])
                otu_index_values.extend([token_row["otu_index"] for token_row in token_rows])
                abundance_values.extend([token_row["abundance"] for token_row in token_rows])
                processed_abundance_values.extend([token_row["processed_abundance"] for token_row in token_rows])
                abundance_rank_values.extend([token_row["abundance_rank"] for token_row in token_rows])

            if not activation_chunks:
                continue
            batch_activations = np.concatenate(activation_chunks, axis=0)
            next_pos = write_pos + batch_activations.shape[0]
            handle["activations"][write_pos:next_pos] = batch_activations.astype(np.float16 if args.dtype == "float16" else np.float32)
            handle["sample_row"][write_pos:next_pos] = np.asarray(sample_row_values, dtype=np.int32)
            handle["sample_index"][write_pos:next_pos] = np.asarray(sample_index_values, dtype=np.int64)
            handle["token_position"][write_pos:next_pos] = np.asarray(token_position_values, dtype=np.int32)
            handle["token_type"][write_pos:next_pos] = np.asarray(token_type_values, dtype=np.int8)
            handle["otu_index"][write_pos:next_pos] = np.asarray(otu_index_values, dtype=np.int64)
            handle["split"][write_pos:next_pos] = np.asarray(split_values, dtype=np.int8)
            handle["abundance"][write_pos:next_pos] = np.asarray(abundance_values, dtype=np.float32)
            handle["processed_abundance"][write_pos:next_pos] = np.asarray(processed_abundance_values, dtype=np.float32)
            handle["abundance_rank"][write_pos:next_pos] = np.asarray(abundance_rank_values, dtype=np.int32)
            write_pos = next_pos
            handle.attrs["written_tokens"] = int(write_pos)
            progress.update(len(batch_rows))
        progress.close()
    finally:
        handle.attrs["written_tokens"] = int(write_pos)
        handle.close()

    if write_pos != num_tokens:
        raise RuntimeError(f"expected to write {num_tokens} tokens, wrote {write_pos}")
    summary = {
        "output": str(output),
        "num_samples": len(sample_rows),
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "dtype": args.dtype,
        "token_type": args.token_type,
        "splits": args.splits,
        "scale": args.scale,
    }
    with (output.with_suffix(output.suffix + ".json")).open("w") as handle_json:
        json.dump(summary, handle_json, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
