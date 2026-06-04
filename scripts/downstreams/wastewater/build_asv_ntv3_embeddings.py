#!/usr/bin/env python3
"""Extract NTv3 embeddings for plant-scoped wastewater ASV sequences."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("/home/zhanghaohong/data6/databases/wastewater/MC-prediction/data_organized")
DEFAULT_OUTPUT = Path("data/downstreams/wastewater/cache_ntv3_asv/wastewater_asv_ntv3_embeddings.h5")


def _clean_sequence(seq: str) -> str:
    return str(seq).upper().replace("U", "T")


def collect_plant_asv_sequences(data_root: Path, plants: Sequence[str] | None = None) -> List[Tuple[str, str]]:
    root = Path(data_root)
    selected = set(plants) if plants else None
    records: List[Tuple[str, str]] = []
    for plant_dir in sorted(root.iterdir()):
        if not plant_dir.is_dir() or plant_dir.name == "combined":
            continue
        if selected is not None and plant_dir.name not in selected:
            continue
        fasta_path = plant_dir / "ASVs.fasta"
        if not fasta_path.exists():
            continue
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq_id = f"{plant_dir.name}::{record.id}"
            records.append((seq_id, _clean_sequence(str(record.seq))))
    return records


def embed_seq(seq: str, model, tokenizer, device: str) -> torch.Tensor:
    seq_len = len(seq)
    pad_len = (128 - (seq_len % 128)) % 128
    padded_seq = seq + "N" * pad_len
    inputs = tokenizer(padded_seq, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1].squeeze(0).cpu()
    return hidden_states[:seq_len].mean(dim=0)


def _load_model(model_path: str, device: str):
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        config=config,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def worker(worker_idx: int, seq_chunk: Sequence[Tuple[str, str]], args) -> str:
    device = f"cuda:{worker_idx}" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_model(args.model_path, device)

    embeddings = []
    otu_ids = []
    for seq_id, seq in tqdm(seq_chunk, desc=f"worker {worker_idx}", position=worker_idx):
        embeddings.append(embed_seq(seq, model, tokenizer, device).numpy())
        otu_ids.append(seq_id.encode("utf-8"))

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_h5_path = temp_dir / f"wastewater_asv_ntv3_tmp_{worker_idx}.h5"
    with h5py.File(temp_h5_path, "w") as handle:
        handle.create_dataset("embeddings", data=np.stack(embeddings).astype(np.float32))
        handle.create_dataset("otu_ids", data=np.asarray(otu_ids))
    return str(temp_h5_path)


def _split_evenly(items: Sequence[Tuple[str, str]], n_chunks: int) -> List[Sequence[Tuple[str, str]]]:
    chunk_size = max(1, math.ceil(len(items) / max(1, n_chunks)))
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated physical GPU ids, e.g. 0,1,2")
    parser.add_argument("--plants", nargs="*", default=None)
    parser.add_argument("--temp_dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    records = collect_plant_asv_sequences(args.data_root, args.plants)
    if not records:
        raise ValueError(f"no ASV sequences found under {args.data_root}")

    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    n_workers = min(len(gpu_ids), len(records))
    chunks = _split_evenly(records, n_workers)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.temp_dir is None:
        args.temp_dir = Path(tempfile.mkdtemp(prefix="wastewater_ntv3_"))
    else:
        args.temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sequences: {len(records)}")
    print(f"Workers: {n_workers}")
    print(f"Output: {args.output_path}")

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=n_workers) as pool:
        results = [pool.apply_async(worker, args=(idx, chunk, args)) for idx, chunk in enumerate(chunks)]
        temp_files = [result.get() for result in results]

    all_embeddings = []
    all_otu_ids = []
    for temp_file in temp_files:
        with h5py.File(temp_file, "r") as handle:
            all_embeddings.append(handle["embeddings"][:])
            all_otu_ids.append(handle["otu_ids"][:])
        os.remove(temp_file)

    with h5py.File(args.output_path, "w") as handle:
        handle.create_dataset("embeddings", data=np.concatenate(all_embeddings, axis=0).astype(np.float32))
        handle.create_dataset("otu_ids", data=np.concatenate(all_otu_ids, axis=0))
        handle.attrs["data_root"] = str(args.data_root)
        handle.attrs["model_path"] = str(args.model_path)
        handle.attrs["id_format"] = "plant::asv_id"

    print(f"Saved wastewater ASV NTv3 embeddings: {args.output_path}")
    print(f"Embeddings: {sum(arr.shape[0] for arr in all_embeddings)}")


if __name__ == "__main__":
    main()
