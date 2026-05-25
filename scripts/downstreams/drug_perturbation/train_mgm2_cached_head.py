#!/usr/bin/env python3
"""Train drug perturbation cross-attention heads from cached MGM2 backbone states."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
script_dir = project_root / "scripts" / "downstreams" / "drug_perturbation"
sys.path.insert(0, str(script_dir))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

import baseline as baseline_utils  # noqa: E402


def _ddp_setup() -> Tuple[bool, int, int]:
    """Returns (is_ddp: bool, local_rank: int, world_size: int)."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _sync_scalar(scalar: float, device: torch.device, world_size: int) -> float:
    """All-reduce a scalar value (averaged across ranks)."""
    t = torch.tensor([scalar], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


class CachedBackboneDataset(Dataset):
    def __init__(self, cache_path: Path):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if data.get("format") != "mgm2_drug_perturbation_backbone_cache":
            raise ValueError(f"Unsupported cache format: {data.get('format')}")
        self.data = data
        self.hidden_states = data["hidden_states"]
        self.tensors = data["tensors"]
        self.examples = data["examples"]
        self.samples = data["samples"]
        self.donor_to_id = data["donor_to_id"]
        self.drug_embedding_dim = int(data["drug_embedding_dim"])
        self.selected_otus = pd.DataFrame(data["selected_otus"]) if data.get("selected_otus") is not None else None
        self.model_info = data["model"]

    def __len__(self) -> int:
        return self.hidden_states.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = {"hidden_states": self.hidden_states[idx]}
        for key, values in self.tensors.items():
            item[key] = values[idx]
        example = self.examples[idx]
        item["sample_id"] = example.get("sample_id", "")
        item["donor"] = example.get("donor", "")
        item["drug_key"] = example.get("drug_key", "")
        return item


class CachedCollator:
    def __call__(self, batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
        result: Dict[str, torch.Tensor | List[str]] = {
            "sample_id": [str(item["sample_id"]) for item in batch],
            "donor": [str(item["donor"]) for item in batch],
            "drug_key": [str(item["drug_key"]) for item in batch],
        }
        tensor_keys = [key for key in batch[0] if key not in result]
        for key in tensor_keys:
            result[key] = torch.stack([item[key] for item in batch])  # type: ignore[index]
        return result


class CachedDrugPerturbationHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        drug_embedding_dim: int,
        num_donors: int,
        num_drug_tokens: int,
        num_attention_heads: int,
        cross_attention_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_drug_tokens = int(num_drug_tokens)
        heads = int(cross_attention_heads or num_attention_heads)
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_embedding_dim, self.hidden_size * self.num_drug_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.donor_embedding = nn.Embedding(num_donors, self.hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.condition_layer_norm = nn.LayerNorm(self.hidden_size)
        self.output_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor, before_abundance_values: torch.Tensor, drug_embedding: torch.Tensor, donor_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        drug_tokens = self.drug_projector(drug_embedding).view(batch_size, self.num_drug_tokens, self.hidden_size)
        donor_token = self.donor_embedding(donor_id).unsqueeze(1)
        condition_tokens = self.condition_layer_norm(torch.cat([drug_tokens, donor_token], dim=1))
        conditioned, _ = self.cross_attention(
            query=hidden_states,
            key=condition_tokens,
            value=condition_tokens,
            need_weights=False,
        )
        conditioned_hidden = self.output_layer_norm(hidden_states + self.dropout(conditioned))
        pred_delta = self.delta_head(conditioned_hidden).squeeze(-1)
        return {"pred_delta": pred_delta, "pred_after_abundance": before_abundance_values + pred_delta}


def raw_to_log1p(values: torch.Tensor, pseudocount: float) -> torch.Tensor:
    return torch.log1p(torch.clamp(values, min=0.0) / float(pseudocount))


def log1p_to_raw(values: torch.Tensor, pseudocount: float) -> torch.Tensor:
    return torch.expm1(torch.clamp(values, min=0.0, max=20.0)) * float(pseudocount)


def target_space_before_after(tensors: Dict[str, torch.Tensor], args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.target_space == "mgm2":
        return tensors["before_abundance_values"], tensors["after_abundance_values"]
    if args.target_space == "log1p_raw":
        before = raw_to_log1p(tensors["before_abundance_eval_raw"], args.pseudocount)
        after = raw_to_log1p(tensors["after_abundance_eval_raw"], args.pseudocount)
        before[:, 0] = 0.0
        after[:, 0] = 0.0
        return before, after
    if args.target_space == "relative_raw":
        before = tensors["before_abundance_eval_raw"].float().clone()
        after = tensors["after_abundance_eval_raw"].float().clone()
        before[:, 0] = 0.0
        after[:, 0] = 0.0
        return before, after
    raise ValueError(f"Unsupported target_space: {args.target_space}")


def target_delta_from_tensors(tensors: Dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    before, after = target_space_before_after(tensors, args)
    return after - before


class DeltaStandardizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, dataset: CachedBackboneDataset, indices: np.ndarray, args: argparse.Namespace) -> "DeltaStandardizer":
        index = torch.as_tensor(indices, dtype=torch.long)
        tensors = {key: value[index] for key, value in dataset.tensors.items()}
        delta = target_delta_from_tensors(tensors, args)
        mean = delta.mean(dim=0).float()
        std = delta.std(dim=0, unbiased=False).float()
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        mean[0] = 0.0
        std[0] = 1.0
        return cls(mean=mean, std=std)

    def to(self, device: torch.device) -> "DeltaStandardizer":
        return DeltaStandardizer(self.mean.to(device), self.std.to(device))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cached MGM2 drug perturbation heads with leave-one-group evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_mode", choices=["pair", "drug", "donor"], default="pair")
    parser.add_argument("--max_folds", type=int, default=0, help="0 means all folds")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0, help="Use 0 for cached tensor data to avoid shared-memory file descriptor exhaustion")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--target_space", choices=["mgm2", "log1p_raw", "relative_raw"], default="mgm2")
    parser.add_argument("--standardize_delta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delta_shrinkage", type=float, default=1.0, help="Fixed multiplier applied to transformed-space predicted delta before raw evaluation")
    parser.add_argument("--raw_after_mse_weight", type=float, default=0.0, help="Auxiliary MSE on raw post-perturbation abundance")
    parser.add_argument("--magnitude_weight", type=float, default=0.0, help="Auxiliary raw delta magnitude MAE")
    parser.add_argument("--delta_cosine_weight", type=float, default=0.0, help="Auxiliary raw delta cosine loss")
    parser.add_argument("--clr_mse_weight", type=float, default=0.0, help="Auxiliary CLR MSE on raw post-perturbation abundance")
    parser.add_argument("--raw_zero_threshold", type=float, default=0.0, help="Set raw predictions <= threshold to zero before evaluation/export")
    parser.add_argument("--blend_no_effect", type=float, default=0.0, help="Blend final raw prediction toward the before profile: beta*before + (1-beta)*pred")
    parser.add_argument("--normalize_prediction", choices=["none", "topk"], default="none")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for single-GPU mode (ignored under torchrun)")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def group_labels(examples: List[Dict], mode: str) -> List[str]:
    if mode == "pair":
        return [f"{example['donor']}::{example['drug_key']}" for example in examples]
    if mode == "drug":
        return [str(example["drug_key"]) for example in examples]
    if mode == "donor":
        return [str(example["donor"]) for example in examples]
    raise ValueError(mode)


def examples_frame(examples: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Run": [example.get("sample_id", "") for example in examples],
            "Donor": [example.get("donor", "") for example in examples],
            "drug_key": [example.get("drug_key", "") for example in examples],
        }
    )


def train_val_indices(labels: List[str], val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=object)
    unique = np.asarray(pd.unique(labels), dtype=object)
    if len(unique) < 2:
        raise ValueError("Need at least two groups for train/validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    n_val = min(n_val, len(unique) - 1)
    val_groups = set(unique[:n_val])
    val_idx = np.flatnonzero(np.asarray([label in val_groups for label in labels]))
    train_idx = np.flatnonzero(np.asarray([label not in val_groups for label in labels]))
    return train_idx, val_idx


def inner_train_val_indices(labels: List[str], train_idx: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_labels = [labels[int(idx)] for idx in train_idx]
    inner_train_pos, inner_val_pos = train_val_indices(train_labels, val_fraction, seed)
    return train_idx[inner_train_pos], train_idx[inner_val_pos]


def make_loader(dataset: CachedBackboneDataset, indices: np.ndarray, args: argparse.Namespace, shuffle: bool, is_ddp: bool = False, world_size: int = 1) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    sampler = None
    drop_last = False
    if is_ddp and shuffle:
        sampler = DistributedSampler(subset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
        shuffle = False
        drop_last = True
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "collate_fn": CachedCollator(),
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": False,
        "drop_last": drop_last,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(subset, sampler=sampler, **kwargs)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def make_model(dataset: CachedBackboneDataset, args: argparse.Namespace) -> CachedDrugPerturbationHead:
    info = dataset.model_info
    return CachedDrugPerturbationHead(
        hidden_size=int(info["hidden_size"]),
        drug_embedding_dim=dataset.drug_embedding_dim,
        num_donors=len(dataset.donor_to_id),
        num_drug_tokens=int(info["num_drug_tokens"]),
        num_attention_heads=int(info["num_attention_heads"]),
        cross_attention_heads=info.get("cross_attention_heads"),
        dropout=float(info.get("mlp_dropout", 0.1)),
    )


def finalize_pred_delta(pred_delta: torch.Tensor, standardizer: DeltaStandardizer | None, delta_shrinkage: float) -> torch.Tensor:
    if standardizer is not None:
        pred_delta = standardizer.inverse_transform(pred_delta)
    if delta_shrinkage != 1.0:
        pred_delta = pred_delta * float(delta_shrinkage)
    return pred_delta


def batch_target_delta(batch: Dict, args: argparse.Namespace) -> torch.Tensor:
    tensors = {
        "before_abundance_values": batch["before_abundance_values"],
        "after_abundance_values": batch["after_abundance_values"],
        "before_abundance_eval_raw": batch["before_abundance_eval_raw"],
        "after_abundance_eval_raw": batch["after_abundance_eval_raw"],
    }
    return target_delta_from_tensors(tensors, args)


def pred_after_to_raw(batch: Dict, pred_delta: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.target_space == "mgm2":
        pred_after = batch["before_abundance_values"] + pred_delta
        return transformed_to_raw(pred_after)
    if args.target_space == "log1p_raw":
        before = raw_to_log1p(batch["before_abundance_eval_raw"], args.pseudocount)
        before[:, 0] = 0.0
        return log1p_to_raw(before + pred_delta, args.pseudocount)
    if args.target_space == "relative_raw":
        return torch.clamp(batch["before_abundance_eval_raw"] + pred_delta, min=0.0)
    raise ValueError(f"Unsupported target_space: {args.target_space}")


def clr_transform(values: torch.Tensor, pseudocount: float) -> torch.Tensor:
    logged = torch.log(torch.clamp(values, min=0.0) + float(pseudocount))
    return logged - logged.mean(dim=1, keepdim=True)


def shared_loss(model: nn.Module, batch: Dict, args: argparse.Namespace, device: torch.device, standardizer: DeltaStandardizer | None) -> torch.Tensor:
    with autocast_context(device, args.precision):
        outputs = model(
            hidden_states=batch["hidden_states"],
            before_abundance_values=batch["before_abundance_values"],
            drug_embedding=batch["drug_embedding"],
            donor_id=batch["donor_id"],
        )
        mask = batch["supervision_mask"].bool()
        target_delta = batch_target_delta(batch, args)
        target_for_primary = standardizer.transform(target_delta) if standardizer is not None else target_delta
        loss = F.mse_loss(outputs["pred_delta"][mask], target_for_primary[mask])
        pred_delta = finalize_pred_delta(outputs["pred_delta"], standardizer, args.delta_shrinkage)

        if args.raw_after_mse_weight > 0:
            pred_raw = pred_after_to_raw(batch, pred_delta, args)[:, 1:]
            target_raw = batch["after_abundance_eval_raw"][:, 1:]
            loss = loss + float(args.raw_after_mse_weight) * F.mse_loss(pred_raw, target_raw)

        if args.magnitude_weight > 0 or args.delta_cosine_weight > 0 or args.clr_mse_weight > 0:
            pred_raw = pred_after_to_raw(batch, pred_delta, args)[:, 1:]
            before_raw = batch["before_abundance_eval_raw"][:, 1:]
            target_raw = batch["after_abundance_eval_raw"][:, 1:]
            pred_raw_delta = pred_raw - before_raw
            target_raw_delta = target_raw - before_raw

        if args.clr_mse_weight > 0:
            pred_clr = clr_transform(pred_raw, args.pseudocount)
            target_clr = clr_transform(target_raw, args.pseudocount)
            loss = loss + float(args.clr_mse_weight) * F.mse_loss(pred_clr, target_clr)

        if args.magnitude_weight > 0:
            pred_magnitude = torch.abs(pred_raw_delta).sum(dim=1)
            target_magnitude = torch.abs(target_raw_delta).sum(dim=1)
            loss = loss + float(args.magnitude_weight) * F.l1_loss(pred_magnitude, target_magnitude)

        if args.delta_cosine_weight > 0:
            numer = (pred_raw_delta * target_raw_delta).sum(dim=1)
            denom = torch.linalg.norm(pred_raw_delta, dim=1) * torch.linalg.norm(target_raw_delta, dim=1)
            cosine = numer / torch.clamp(denom, min=1e-12)
            loss = loss + float(args.delta_cosine_weight) * (1.0 - cosine).mean()
    return loss


def train_fold(dataset: CachedBackboneDataset, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace, device: torch.device, is_ddp: bool = False, rank: int = 0, world_size: int = 1) -> Tuple[CachedDrugPerturbationHead | None, Dict[str, float]]:
    model = make_model(dataset, args).to(device)
    standardizer = DeltaStandardizer.fit(dataset, train_idx, args).to(device) if args.standardize_delta else None
    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_loader(dataset, train_idx, args, shuffle=True, is_ddp=is_ddp, world_size=world_size)
    val_loader = make_loader(dataset, val_idx, args, shuffle=False, is_ddp=is_ddp, world_size=world_size)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    epochs_trained = 0

    for epoch in range(args.max_epochs):
        if is_ddp and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_losses = []
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = shared_loss(model, batch, args, device, standardizer)
            loss.backward()
            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = batch_to_device(raw_batch, device)
                val_loss_sum += float(shared_loss(model, batch, args, device, standardizer).detach().cpu())
                val_count += 1
        if val_count > 0:
            val_loss = val_loss_sum / val_count
            if is_ddp:
                val_loss = _sync_scalar(val_loss, device, world_size)
        else:
            val_loss = float("inf")

        epochs_trained = epoch + 1
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            # Unwrap DDP model to save state dict
            model_ref = model.module if is_ddp else model
            best_state = {key: value.detach().cpu().clone() for key, value in model_ref.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if _is_rank_zero():
            train_mean = np.mean(train_losses) if train_losses else 0.0
            print(f"epoch={epoch + 1} train_loss={train_mean:.6f} val_loss={val_loss:.6f}", flush=True)
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model_ref = model.module if is_ddp else model
        model_ref.load_state_dict(best_state)
        model_ref.to(device)
    else:
        model_ref = model.module if is_ddp else model
        model_ref.to(device)

    if not _is_rank_zero():
        return None, {"best_val_loss": best_val, "epochs_trained": epochs_trained, "standardizer": standardizer}

    return model_ref, {"best_val_loss": best_val, "epochs_trained": epochs_trained, "standardizer": standardizer}


def transformed_to_raw(values: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(values, min=-2.0, max=2.0)
    return torch.pow(torch.tensor(10.0, device=values.device, dtype=values.dtype), clipped - 2.0)


def postprocess_raw_prediction(pred: np.ndarray, before: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    result = np.clip(pred.astype(np.float32, copy=True), 0.0, None)
    if args.raw_zero_threshold > 0:
        result[result <= float(args.raw_zero_threshold)] = 0.0
    if args.blend_no_effect > 0:
        beta = float(np.clip(args.blend_no_effect, 0.0, 1.0))
        result = (1.0 - beta) * result + beta * before.astype(np.float32)
    if args.normalize_prediction == "topk":
        row_sum = result.sum(axis=1, keepdims=True)
        normalized = np.zeros_like(result, dtype=np.float32)
        np.divide(result, np.clip(row_sum, 1e-12, None), out=normalized, where=row_sum > 0)
        result = normalized
    return result.astype(np.float32)


def predict(model: CachedDrugPerturbationHead, dataset: CachedBackboneDataset, test_idx: np.ndarray, args: argparse.Namespace, device: torch.device, standardizer: DeltaStandardizer | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = make_loader(dataset, test_idx, args, shuffle=False)
    preds, true_profiles, baseline_profiles = [], [], []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            with autocast_context(device, args.precision):
                outputs = model(
                    hidden_states=batch["hidden_states"],
                    before_abundance_values=batch["before_abundance_values"],
                    drug_embedding=batch["drug_embedding"],
                    donor_id=batch["donor_id"],
                )
                pred_delta = finalize_pred_delta(outputs["pred_delta"], standardizer, args.delta_shrinkage)
                pred_raw = pred_after_to_raw(batch, pred_delta, args)
            before_raw = batch["before_abundance_eval_raw"].detach().cpu().numpy()[:, 1:].astype(np.float32)
            pred_np = pred_raw.detach().float().cpu().numpy()[:, 1:].astype(np.float32)
            preds.append(postprocess_raw_prediction(pred_np, before_raw, args))
            true_profiles.append(batch["after_abundance_eval_raw"].detach().cpu().numpy()[:, 1:].astype(np.float32))
            baseline_profiles.append(before_raw)
    return np.concatenate(preds), np.concatenate(true_profiles), np.concatenate(baseline_profiles)


def _h5_string_array(values: List[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def write_prediction_h5(output_path: Path, selected_otus: pd.DataFrame, metadata_frames: List[pd.DataFrame], prediction_chunks: Dict[str, List[np.ndarray]]) -> None:
    if not metadata_frames:
        return
    metadata = pd.concat(metadata_frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("otu_ids", data=_h5_string_array(selected_otus["otu_id"].astype(str).to_list()), dtype=string_dtype)
        handle.create_dataset("otu_idx", data=selected_otus["otu_idx"].astype(int).to_numpy())
        handle.attrs["prediction_space"] = "raw_relative_abundance"
        meta_group = handle.create_group("metadata")
        for column in ["split_mode", "group", "Run", "Donor", "drug_key"]:
            meta_group.create_dataset(column, data=_h5_string_array(metadata[column].astype(str).to_list()), dtype=string_dtype)
        meta_group.create_dataset("fold", data=metadata["fold"].astype(int).to_numpy())
        pred_group = handle.create_group("predictions")
        for method, chunks in sorted(prediction_chunks.items()):
            if chunks:
                pred_group.create_dataset(method, data=np.concatenate(chunks, axis=0).astype(np.float32), compression="gzip")


def main() -> None:
    args = parse_args()
    is_ddp, rank, world_size = _ddp_setup()

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if torch.cuda.is_available():
        if is_ddp:
            device = torch.device(f"cuda:{rank}")
        else:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    dataset = CachedBackboneDataset(args.cache)
    examples = examples_frame(dataset.examples)
    labels = group_labels(dataset.examples, args.split_mode)
    if dataset.selected_otus is None:
        raise ValueError("Cache must include selected_otus for prediction export")

    if _is_rank_zero():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "cached_head_args.json").open("w", encoding="utf-8") as handle:
            json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)

    fold_frames = []
    prediction_metadata: List[pd.DataFrame] = []
    prediction_chunks: Dict[str, List[np.ndarray]] = {"mgm2_cached_head": []}
    max_folds = None if args.max_folds == 0 else args.max_folds
    for fold_idx, (train_idx, test_idx, group_label) in enumerate(baseline_utils.iter_leave_one_groups(examples, args.split_mode)):
        if max_folds is not None and fold_idx >= max_folds:
            break
        fold_train_idx, fold_val_idx = inner_train_val_indices(labels, train_idx, args.val_fraction, args.seed + fold_idx)
        if _is_rank_zero():
            print(f"{args.split_mode} fold {fold_idx + 1}: {group_label} train={len(fold_train_idx)} val={len(fold_val_idx)} test={len(test_idx)}", flush=True)

        model, history = train_fold(dataset, fold_train_idx, fold_val_idx, args, device, is_ddp=is_ddp, rank=rank, world_size=world_size)

        if _is_rank_zero():
            pred, true, before = predict(model, dataset, test_idx, args, device, history.get("standardizer"))
            prediction_chunks["mgm2_cached_head"].append(pred)

            test_examples = examples.iloc[test_idx].reset_index(drop=True)
            prediction_metadata.append(
                pd.DataFrame(
                    {
                        "split_mode": args.split_mode,
                        "fold": int(fold_idx),
                        "group": group_label,
                        "Run": test_examples["Run"].astype(str).to_list(),
                        "Donor": test_examples["Donor"].astype(str).to_list(),
                        "drug_key": test_examples["drug_key"].astype(str).to_list(),
                    }
                )
            )
            metrics = baseline_utils.evaluate_predictions(
                {"mgm2_cached_head": pred},
                true,
                before,
                pseudocount=args.pseudocount,
            )
            metrics.insert(0, "best_val_loss", history["best_val_loss"])
            metrics.insert(0, "epochs_trained", history["epochs_trained"])
            metrics.insert(0, "n_test", len(test_idx))
            metrics.insert(0, "group", group_label)
            metrics.insert(0, "fold", fold_idx)
            metrics.insert(0, "split_mode", args.split_mode)
            fold_frames.append(metrics)

        if is_ddp:
            dist.barrier()

    if _is_rank_zero():
        if not fold_frames:
            raise RuntimeError("No folds were evaluated")
        fold_metrics = pd.concat(fold_frames, ignore_index=True)
        summary = baseline_utils.summarize_metrics(fold_metrics)
        fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
        summary.to_csv(args.output_dir / "summary.csv", index=False)
        write_prediction_h5(args.output_dir / f"predicted_abundance_{args.split_mode}.h5", dataset.selected_otus, prediction_metadata, prediction_chunks)
        print(summary.to_string(index=False))

    _ddp_cleanup()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
