#!/usr/bin/env python3
"""Train a BatchTopK SAE on cached MGM2 activations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import sys
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.BatchTopKSAE import BatchTopKSAE


DEFAULT_ACTIVATIONS = Path("data/sae/pretrain_xlarge_sample50k/mgm2_scale_xlarge_last_hidden_activations.h5")
TOKEN_TYPE_TO_ID = {"cls": 0, "otu": 1}
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--token_type", choices=["otu", "cls", "all"], default="otu")
    parser.add_argument("--dict_size", type=int, default=None, help="SAE dictionary size. Defaults to 8 * hidden_size.")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--aux_k", type=int, default=16)
    parser.add_argument("--aux_penalty", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--precision", choices=["float32", "bf16-mixed", "fp16-mixed"], default="float32")
    parser.add_argument("--dead_feature_threshold", type=int, default=10_000_000)
    parser.add_argument("--inactive_counter_mode", choices=["tokens", "forwards"], default="tokens")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--early_stopping", action="store_true", help="Stop when validation loss plateaus.")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Number of validation checks without sufficient improvement before stopping.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-5,
        help="Minimum val_loss improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--early_stopping_min_steps",
        type=int,
        default=0,
        help="Do not stop before this many training steps, even if patience is exhausted.",
    )
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_input_unit_norm", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--num_index_samples", type=int, default=0, help="Debug: subsample train/val index arrays")
    parser.add_argument(
        "--preload_to_memory",
        action="store_true",
        help="Load selected train/val/test activations into RAM once. Much faster than random HDF5 reads.",
    )
    parser.add_argument(
        "--preload_splits",
        nargs="*",
        default=["train", "val"],
        help="Splits to preload when --preload_to_memory is set",
    )
    parser.add_argument(
        "--preload_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="RAM dtype for preloaded activations. float16 is usually enough and halves memory use.",
    )
    parser.add_argument(
        "--preload_chunk_rows",
        type=int,
        default=262_144,
        help="Contiguous HDF5 rows to scan per preload block.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class H5ActivationStore:
    def __init__(
        self,
        path: Path,
        token_type: str,
        seed: int,
        num_index_samples: int = 0,
        preload_to_memory: bool = False,
        preload_splits: list[str] | None = None,
        preload_dtype: str = "float16",
        preload_chunk_rows: int = 262_144,
    ):
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        self.activations = self.handle["activations"]
        self.hidden_size = int(self.activations.shape[1])
        self.rng = np.random.default_rng(seed)
        split = self.handle["split"][:]
        token_types = self.handle["token_type"][:]
        token_mask = np.ones(len(split), dtype=bool) if token_type == "all" else token_types == TOKEN_TYPE_TO_ID[token_type]
        self.indices: Dict[str, np.ndarray] = {}
        for split_name, split_id in SPLIT_TO_ID.items():
            idx = np.flatnonzero((split == split_id) & token_mask).astype(np.int64)
            if num_index_samples > 0 and len(idx) > num_index_samples:
                idx = np.sort(self.rng.choice(idx, size=num_index_samples, replace=False).astype(np.int64))
            self.indices[split_name] = idx
        if len(self.indices["train"]) == 0:
            raise ValueError("no train activations selected")
        if len(self.indices["val"]) == 0:
            raise ValueError("no val activations selected")
        self.preloaded: Dict[str, np.ndarray] = {}
        if preload_to_memory:
            preload_splits = preload_splits or ["train", "val"]
            preload_splits = [split_name for split_name in preload_splits if split_name in self.indices]
            preload_np_dtype = np.dtype(preload_dtype)
            if num_index_samples > 0:
                for split_name in preload_splits:
                    idx = self.indices[split_name]
                    if len(idx) == 0:
                        continue
                    print(f"Preloading {split_name}: {len(idx):,} activations")
                    self.preloaded[split_name] = self._read_sorted_indices(idx, dtype=preload_np_dtype)
            else:
                self._preload_splits_by_scan(
                    preload_splits=preload_splits,
                    split=split,
                    token_mask=token_mask,
                    dtype=preload_np_dtype,
                    chunk_rows=preload_chunk_rows,
                )

    def _read_sorted_indices(self, idx: np.ndarray, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return np.empty((0, self.hidden_size), dtype=dtype)
        order = np.argsort(idx)
        sorted_idx = idx[order]
        values = self.activations[sorted_idx].astype(dtype, copy=False)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return values[inverse]

    def _preload_splits_by_scan(
        self,
        preload_splits: list[str],
        split: np.ndarray,
        token_mask: np.ndarray,
        dtype: np.dtype,
        chunk_rows: int,
    ):
        preload_splits = [split_name for split_name in preload_splits if len(self.indices[split_name]) > 0]
        if not preload_splits:
            return
        chunk_rows = max(1, int(chunk_rows))
        split_ids = {split_name: SPLIT_TO_ID[split_name] for split_name in preload_splits}
        expected_counts = {split_name: int(len(self.indices[split_name])) for split_name in preload_splits}
        buffers: Dict[str, np.ndarray] = {
            split_name: np.empty((count, self.hidden_size), dtype=dtype) for split_name, count in expected_counts.items()
        }
        offsets = {split_name: 0 for split_name in preload_splits}
        print(
            "Preloading by sequential HDF5 scan: "
            + ", ".join(f"{split_name}={count:,}" for split_name, count in expected_counts.items())
        )
        progress = tqdm(range(0, len(split), chunk_rows), desc="preload", unit="rows")
        for start in progress:
            end = min(start + chunk_rows, len(split))
            split_block = split[start:end]
            mask_base = token_mask[start:end]
            if not mask_base.any():
                continue
            block = self.activations[start:end]
            for split_name, split_id in split_ids.items():
                mask = mask_base & (split_block == split_id)
                if mask.any():
                    selected = block[mask].astype(dtype, copy=False)
                    offset = offsets[split_name]
                    next_offset = offset + len(selected)
                    buffers[split_name][offset:next_offset] = selected
                    offsets[split_name] = next_offset
        for split_name, values in buffers.items():
            self.preloaded[split_name] = values
            actual = offsets[split_name]
            expected = expected_counts[split_name]
            if actual != expected:
                raise RuntimeError(f"preloaded {split_name} count mismatch: expected {expected}, got {actual}")
            print(f"Preloaded {split_name}: {actual:,} activations ({self.preloaded[split_name].dtype})")

    def close(self):
        self.handle.close()

    def sample_batch(self, split: str, batch_size: int) -> torch.Tensor:
        if split in self.preloaded:
            values = self.preloaded[split]
            replace = len(values) < batch_size
            sampled = self.rng.choice(len(values), size=batch_size, replace=replace).astype(np.int64)
            batch = values[sampled]
            return torch.from_numpy(batch)
        idx = self.indices[split]
        replace = len(idx) < batch_size
        sampled = self.rng.choice(idx, size=batch_size, replace=replace).astype(np.int64)
        sampled.sort()
        batch = self.activations[sampled].astype(np.float32)
        return torch.from_numpy(batch)

    def counts(self) -> Dict[str, int]:
        return {split: int(len(idx)) for split, idx in self.indices.items()}


def compute_metrics(model: BatchTopKSAE, store: H5ActivationStore, args, device) -> Dict[str, float]:
    model.eval()
    losses = []
    mses = []
    evs = []
    l0s = []
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x = store.sample_batch("val", args.batch_size).to(device)
            out = model(x, update_inactive=False)
            losses.append(float(out.loss.item()))
            mses.append(float(torch.mean((out.sae_out - x.float()).pow(2)).item()))
            evs.append(float(model.explained_variance(x.float(), out.sae_out).item()))
            l0s.append(float((out.feature_acts > 0).sum(dim=-1).float().mean().item()))
    dead = int(model.num_dead_features().item())
    return {
        "val_loss": float(np.mean(losses)),
        "val_mse": float(np.mean(mses)),
        "val_explained_variance": float(np.mean(evs)),
        "val_l0": float(np.mean(l0s)),
        "dead_features": dead,
    }


def save_checkpoint(path: Path, model: BatchTopKSAE, optimizer, step: int, args, metrics: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def set_warmup_lr(optimizer, base_lr: float, step: int, warmup_steps: int):
    if warmup_steps <= 0:
        return
    scale = min(1.0, float(step) / float(max(1, warmup_steps)))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    store = H5ActivationStore(
        args.activations,
        args.token_type,
        args.seed,
        args.num_index_samples,
        preload_to_memory=args.preload_to_memory,
        preload_splits=args.preload_splits,
        preload_dtype=args.preload_dtype,
        preload_chunk_rows=args.preload_chunk_rows,
    )
    if args.dict_size is None:
        args.dict_size = store.hidden_size * 8
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(f"outputs_sae/mgm2_xlarge_{args.token_type}_batchtopk_d{args.dict_size}_k{args.top_k}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BatchTopKSAE(
        input_dim=store.hidden_size,
        dict_size=args.dict_size,
        top_k=args.top_k,
        aux_k=args.aux_k,
        aux_penalty=args.aux_penalty,
        input_unit_norm=not args.no_input_unit_norm,
        dead_feature_threshold=args.dead_feature_threshold,
        inactive_counter_mode=args.inactive_counter_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))

    summary = {
        "activations": str(args.activations),
        "output_dir": str(output_dir),
        "device": str(device),
        "hidden_size": store.hidden_size,
        "token_type": args.token_type,
        "dict_size": args.dict_size,
        "top_k": args.top_k,
        "aux_k": args.aux_k,
        "aux_penalty": args.aux_penalty,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "warmup_steps": args.warmup_steps,
        "precision": args.precision,
        "dead_feature_threshold": args.dead_feature_threshold,
        "inactive_counter_mode": args.inactive_counter_mode,
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "save_every": args.save_every,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_min_steps": args.early_stopping_min_steps,
        "preload_to_memory": args.preload_to_memory,
        "preload_splits": args.preload_splits,
        "preload_dtype": args.preload_dtype,
        "preload_chunk_rows": args.preload_chunk_rows,
        "index_counts": store.counts(),
    }
    with (output_dir / "config_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))

    history = []
    best_val = float("inf")
    bad_eval_count = 0
    progress = tqdm(range(start_step + 1, args.steps + 1), desc="train")
    try:
        for step in progress:
            model.train()
            x = store.sample_batch("train", args.batch_size).to(device)
            set_warmup_lr(optimizer, args.learning_rate, step, args.warmup_steps)
            with autocast_context(device, args.precision):
                out = model(x, update_inactive=True)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            model.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            model.normalize_decoder_weights()

            row = {
                "step": step,
                "train_loss": float(out.loss.item()),
                "train_mse_loss": float(out.mse_loss.item()),
                "train_aux_loss": float(out.aux_loss.item()),
                "train_l0": float((out.feature_acts > 0).sum(dim=-1).float().mean().item()),
            }
            progress.set_postfix(loss=row["train_loss"], l0=row["train_l0"])
            if step % args.eval_every == 0 or step == args.steps:
                metrics = compute_metrics(model, store, args, device)
                row.update(metrics)
                val_loss = float(metrics["val_loss"])
                improved = val_loss < best_val - args.early_stopping_min_delta
                if improved:
                    best_val = val_loss
                    bad_eval_count = 0
                    save_checkpoint(output_dir / "best.pt", model, optimizer, step, args, metrics)
                else:
                    bad_eval_count += 1
                should_stop = (
                    args.early_stopping
                    and step >= args.early_stopping_min_steps
                    and bad_eval_count >= args.early_stopping_patience
                )
                row["best_val_loss"] = best_val
                row["bad_eval_count"] = bad_eval_count
                row["early_stop"] = bool(should_stop)
                history.append(row)
                pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
                save_checkpoint(output_dir / "last.pt", model, optimizer, step, args, metrics)
                print(json.dumps(row, sort_keys=True))
                if should_stop:
                    print(
                        json.dumps(
                            {
                                "event": "early_stopping",
                                "step": step,
                                "best_val_loss": best_val,
                                "bad_eval_count": bad_eval_count,
                                "patience": args.early_stopping_patience,
                                "min_delta": args.early_stopping_min_delta,
                            },
                            sort_keys=True,
                        )
                    )
                    break
            elif step % args.save_every == 0:
                save_checkpoint(output_dir / "last.pt", model, optimizer, step, args, {})
    finally:
        store.close()


if __name__ == "__main__":
    main()
