#!/usr/bin/env python3
"""Train MGM2 cross-attention models for drug perturbation prediction."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Subset

from src.configs.Config import MGM2Config
from src.data.DrugPerturbationDataset import DrugPerturbationCollator, DrugPerturbationDataset
from src.training.DrugPerturbationModule import DrugPerturbationLightningModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a drug-conditioned MGM2 perturbation model with cross-attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="MicrobeDrugPerturbation data directory")
    parser.add_argument("--ntv3_embedding_path", type=Path, required=True, help="NTv3 embedding HDF5")
    parser.add_argument("--config", type=Path, default=project_root / "src" / "configs" / "mgm2_default.json")
    parser.add_argument("--pretrained_checkpoint", type=Path, default=None, help="Optional MGM2 pretraining checkpoint")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/mgm2_cross_attention"))
    parser.add_argument("--experiment_name", type=str, default="drug_perturbation_cross_attention")
    parser.add_argument("--split_mode", choices=["pair", "drug", "donor"], default="pair")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_otus", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--backbone_learning_rate", type=float, default=None)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_drug_tokens", type=int, default=4)
    parser.add_argument("--cross_attention_heads", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def group_labels(dataset: DrugPerturbationDataset, mode: str) -> List[str]:
    labels = []
    for example in dataset.examples:
        if mode == "pair":
            labels.append(f"{example['donor']}::{example['drug_key']}")
        elif mode == "drug":
            labels.append(example["drug_key"])
        elif mode == "donor":
            labels.append(example["donor"])
        else:
            raise ValueError(f"Unsupported split mode: {mode}")
    return labels


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


def load_pretrained_backbone(module: DrugPerturbationLightningModule, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone."):
            backbone_state[key[len("model.backbone.") :]] = value
        elif key.startswith("backbone."):
            backbone_state[key[len("backbone.") :]] = value
    if not backbone_state:
        raise ValueError(f"No MGM2 backbone weights found in checkpoint: {checkpoint_path}")
    missing, unexpected = module.model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"Loaded pretrained backbone from {checkpoint_path}")
    print(f"Backbone missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")


def main():
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if not args.ntv3_embedding_path.exists():
        raise FileNotFoundError(args.ntv3_embedding_path)
    config = MGM2Config.load(str(args.config))
    config.output_dir = str(args.output_dir)
    config.experiment_name = args.experiment_name
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.devices = args.devices
    config.training.accelerator = args.accelerator
    config.training.precision = args.precision
    config.training.learning_rate = args.learning_rate
    config.seed = args.seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(args.output_dir / "mgm2_config.json"))
    with (args.output_dir / "drug_perturbation_args.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)

    pl.seed_everything(args.seed, workers=True)
    dataset = DrugPerturbationDataset(
        data_dir=args.data_dir,
        ntv3_embedding_path=args.ntv3_embedding_path,
        max_otus=args.max_otus,
        min_abundance=config.abundance.min_abundance,
        output_min=config.abundance.output_min,
        output_max=config.abundance.output_max,
        normalize=config.abundance.normalize,
    )
    labels = group_labels(dataset, args.split_mode)
    train_idx, val_idx = train_val_indices(labels, args.val_fraction, args.seed)
    collator = DrugPerturbationCollator()
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Examples: {len(dataset)} | train: {len(train_idx)} | val: {len(val_idx)}")

    module = DrugPerturbationLightningModule(
        config=config,
        drug_embedding_dim=dataset.drug_embeddings.shape[1],
        num_donors=len(dataset.donor_to_id),
        num_drug_tokens=args.num_drug_tokens,
        cross_attention_heads=args.cross_attention_heads,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
    )
    if args.pretrained_checkpoint is not None:
        load_pretrained_backbone(module, args.pretrained_checkpoint)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints",
            filename="drug-perturbation-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", patience=8, mode="min"),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(save_dir=args.output_dir / "logs", name=args.experiment_name),
            CSVLogger(save_dir=args.output_dir / "logs", name=args.experiment_name),
        ],
        gradient_clip_val=config.training.gradient_clip_val,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
