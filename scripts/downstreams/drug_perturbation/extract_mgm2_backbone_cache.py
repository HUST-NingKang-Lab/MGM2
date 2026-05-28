#!/usr/bin/env python3
"""Extract frozen MGM2 backbone states for drug perturbation head training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.configs.Config import MGM2Config
from src.data.DrugPerturbationDataset import DrugPerturbationCollator, DrugPerturbationPreprocessedDataset
from src.models.DrugPerturbationModel import DrugPerturbationCrossAttentionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen MGM2 backbone hidden states for drug perturbation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preprocessed_data", type=Path, default=Path("data/downstreams/drug_perturbation/drug_perturbation_top1024.pkl"))
    parser.add_argument("--ntv3_embedding_path", type=Path, default=None, help="Only needed for older pkl files without fixed_embeddings")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrained_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0, help="Optional sample cap for smoke tests; 0 means all samples")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_drug_tokens", type=int, default=4)
    parser.add_argument("--cross_attention_heads", type=int, default=None)
    parser.add_argument("--storage_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def load_backbone_state(model: DrugPerturbationCrossAttentionModel, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    backbone_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone."):
            backbone_state[key[len("model.backbone.") :]] = value
        elif key.startswith("backbone."):
            backbone_state[key[len("backbone.") :]] = value
    if not backbone_state:
        raise ValueError(f"No backbone weights found in {checkpoint_path}")
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"Loaded backbone: missing={len(missing)} unexpected={len(unexpected)}")


def tensor_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    if not args.preprocessed_data.exists():
        raise FileNotFoundError(args.preprocessed_data)
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(args.pretrained_checkpoint)

    config = MGM2Config.load(str(args.config))
    dataset = DrugPerturbationPreprocessedDataset(args.preprocessed_data, ntv3_embedding_path=args.ntv3_embedding_path)
    source_dataset = dataset
    if args.max_samples and args.max_samples > 0:
        source_dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    collator = DrugPerturbationCollator()
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(source_dataset, **loader_kwargs)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = DrugPerturbationCrossAttentionModel(
        model_config=config.model.__dict__,
        drug_embedding_dim=dataset.drug_embedding_dim,
        num_donors=len(dataset.donor_to_id),
        num_drug_tokens=args.num_drug_tokens,
        cross_attention_heads=args.cross_attention_heads,
        dropout=config.model.mlp_dropout,
    )
    load_backbone_state(model, args.pretrained_checkpoint)
    model.backbone.to(device)
    model.backbone.eval()

    hidden_chunks = []
    tensor_chunks: Dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "before_abundance_values": [],
        "after_abundance_values": [],
        "before_abundance_eval_raw": [],
        "after_abundance_eval_raw": [],
        "supervision_mask": [],
        "drug_embedding": [],
        "donor_id": [],
    }
    store_dtype = tensor_dtype(args.storage_dtype)
    with torch.no_grad():
        for batch in tqdm(loader, desc="extract backbone"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            embeddings = batch["embeddings"].to(device, non_blocking=True)
            before = batch["before_abundance_values"].to(device, non_blocking=True)
            attention = batch["attention_mask"].to(device, non_blocking=True)
            outputs = model.backbone(
                input_ids=input_ids,
                embeddings=embeddings,
                abundance_values=before,
                attention_mask=attention,
            )
            hidden_chunks.append(outputs["last_hidden_state"].detach().cpu().to(store_dtype))
            for key in tensor_chunks:
                tensor_chunks[key].append(batch[key].detach().cpu())

    tensors = {key: torch.cat(chunks, dim=0) for key, chunks in tensor_chunks.items()}
    hidden_states = torch.cat(hidden_chunks, dim=0)
    kept_count = int(hidden_states.shape[0])
    payload = {
        "format": "mgm2_drug_perturbation_backbone_cache",
        "format_version": 1,
        "hidden_states": hidden_states,
        "tensors": tensors,
        "examples": dataset.examples[:kept_count],
        "samples": dataset.samples[:kept_count],
        "donor_to_id": dataset.donor_to_id,
        "drug_embedding_dim": dataset.drug_embedding_dim,
        "selected_otus": dataset.selected_otus.to_dict(orient="list") if dataset.selected_otus is not None else None,
        "config_path": str(args.config),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "model": {
            "hidden_size": int(config.model.hidden_size),
            "num_attention_heads": int(config.model.num_attention_heads),
            "mlp_dropout": float(config.model.mlp_dropout),
            "num_drug_tokens": int(args.num_drug_tokens),
            "cross_attention_heads": args.cross_attention_heads,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    with (args.output.with_suffix(args.output.suffix + ".json")).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_samples": kept_count,
                "hidden_states_shape": list(payload["hidden_states"].shape),
                "storage_dtype": str(payload["hidden_states"].dtype),
                "config_path": str(args.config),
                "pretrained_checkpoint": str(args.pretrained_checkpoint),
            },
            handle,
            indent=2,
        )
    print(f"Wrote cache to {args.output}")
    print(f"hidden_states: {tuple(payload['hidden_states'].shape)} {payload['hidden_states'].dtype}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
