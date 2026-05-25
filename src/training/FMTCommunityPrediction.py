#!/usr/bin/env python3
"""Train MGM2 FMT donor-recipient models for post-FMT community prediction."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.configs.Config import MGM2Config
from src.data.FMTDataset import (
    FMTCommunityPredictionDataset,
    create_fmt_community_prediction_dataloaders,
    create_fmt_community_prediction_test_loader,
)
from src.training.FMTCommunityPredictionModule import FMTCommunityPredictionLightningModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MGM2 FMT model that receives donor and recipient communities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_data_file", type=Path, required=True)
    parser.add_argument("--train_metadata_file", type=Path, required=True)
    parser.add_argument("--test_data_file", type=Path, default=None)
    parser.add_argument("--test_metadata_file", type=Path, default=None)
    parser.add_argument("--ntv3_embedding_path", type=Path, required=True)
    parser.add_argument("--test_ntv3_embedding_path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=project_root / "src" / "configs" / "mgm2_default.json")
    parser.add_argument("--pretrained_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT/community_prediction"))
    parser.add_argument("--experiment_name", type=str, default="fmt_community_prediction")
    parser.add_argument("--prediction_file", type=Path, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_group_column", type=str, default="fmt_id")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross_attention_heads", type=int, default=None)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw_zero_threshold", type=float, default=5e-5)
    parser.add_argument("--renormalize_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_checkpoint", choices=["best", "last", "none"], default="best")
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def _jsonable_args(args) -> Dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def make_module(config: MGM2Config, args) -> FMTCommunityPredictionLightningModule:
    module = FMTCommunityPredictionLightningModule(
        config=config,
        cross_attention_heads=args.cross_attention_heads,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        raw_zero_threshold=args.raw_zero_threshold,
        renormalize_predictions=args.renormalize_predictions,
    )
    if args.pretrained_checkpoint is not None:
        module.load_pretrained_backbone_from_checkpoint(str(args.pretrained_checkpoint))
        print(f"Loaded pretrained MGM2 backbone from {args.pretrained_checkpoint}")
    return module


def make_trainer(config: MGM2Config, args) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints",
            filename="fmt-community-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", patience=12, mode="min"),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    devices = args.devices
    if args.accelerator == "cpu" or (args.accelerator == "auto" and not torch.cuda.is_available()):
        devices = 1
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
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


def resolve_eval_ckpt(trainer: pl.Trainer, mode: str):
    if mode == "none":
        return None
    if mode == "best":
        checkpoint = getattr(trainer, "checkpoint_callback", None)
        if checkpoint is not None and checkpoint.best_model_path:
            return "best"
        return None
    return "last"


def _flatten_predictions(predictions: Iterable[Any]) -> List[Dict[str, Any]]:
    flattened = []
    for item in predictions:
        if isinstance(item, list):
            flattened.extend(_flatten_predictions(item))
        else:
            flattened.append(item)
    return flattened


def _string_array(values: List[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def _bray_curtis(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    denom = pred.sum(axis=1) + true.sum(axis=1)
    return np.abs(pred - true).sum(axis=1) / np.clip(denom, 1e-12, None)


def _cosine(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    return (pred * true).sum(axis=1) / np.clip(denom, 1e-12, None)


def write_predictions(output_path: Path, dataset: FMTCommunityPredictionDataset, predictions: Iterable[Any]) -> None:
    chunks = _flatten_predictions(predictions)
    if not chunks:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, List[Any]] = {
        "target_id": [],
        "fmt_id": [],
        "study": [],
        "response_label": [],
        "donor_sample_id": [],
        "recipient_sample_id": [],
        "post_sample_id": [],
        "timepoint_fmt": [],
    }
    arrays = {"pred_post_raw": [], "true_post_raw": [], "recipient_raw": [], "donor_raw": []}
    for chunk in chunks:
        for key in metadata:
            if key == "timepoint_fmt":
                metadata[key].extend(chunk[key].numpy().astype(np.float32).tolist())
            else:
                metadata[key].extend(chunk[key])
        for key in arrays:
            arrays[key].append(chunk[key].numpy().astype(np.float32))

    stacked = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in arrays.items()}
    metrics = pd.DataFrame(
        {
            "target_id": metadata["target_id"],
            "fmt_id": metadata["fmt_id"],
            "study": metadata["study"],
            "response_label": metadata["response_label"],
            "donor_sample_id": metadata["donor_sample_id"],
            "recipient_sample_id": metadata["recipient_sample_id"],
            "post_sample_id": metadata["post_sample_id"],
            "timepoint_fmt": metadata["timepoint_fmt"],
            "mgm2_bray_curtis": _bray_curtis(stacked["pred_post_raw"], stacked["true_post_raw"]),
            "recipient_baseline_bray_curtis": _bray_curtis(stacked["recipient_raw"], stacked["true_post_raw"]),
            "mgm2_cosine": _cosine(stacked["pred_post_raw"], stacked["true_post_raw"]),
            "recipient_baseline_cosine": _cosine(stacked["recipient_raw"], stacked["true_post_raw"]),
        }
    )
    metrics.to_csv(output_path.with_suffix(".metrics.csv"), index=False)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.attrs["prediction_space"] = "raw_relative_abundance"
        handle.create_dataset("feature_ids", data=_string_array(dataset.feature_ids), dtype=string_dtype)
        meta_group = handle.create_group("metadata")
        for key, value in metadata.items():
            if key == "timepoint_fmt":
                meta_group.create_dataset(key, data=np.asarray(value, dtype=np.float32))
            else:
                meta_group.create_dataset(key, data=_string_array(value), dtype=string_dtype)
        pred_group = handle.create_group("predictions")
        for key, value in stacked.items():
            pred_group.create_dataset(key, data=value, compression="gzip")
    print(metrics[["mgm2_bray_curtis", "recipient_baseline_bray_curtis", "mgm2_cosine", "recipient_baseline_cosine"]].mean().to_string())


def main():
    args = parse_args()
    if not args.train_data_file.exists():
        raise FileNotFoundError(args.train_data_file)
    if not args.train_metadata_file.exists():
        raise FileNotFoundError(args.train_metadata_file)
    if not args.ntv3_embedding_path.exists():
        raise FileNotFoundError(args.ntv3_embedding_path)
    if args.test_data_file is not None and not args.test_data_file.exists():
        raise FileNotFoundError(args.test_data_file)
    if args.test_metadata_file is not None and not args.test_metadata_file.exists():
        raise FileNotFoundError(args.test_metadata_file)

    config = MGM2Config.load(str(args.config))
    config.output_dir = str(args.output_dir)
    config.experiment_name = args.experiment_name
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.devices = args.devices
    config.training.accelerator = args.accelerator
    config.training.precision = args.precision
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.seed = args.seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(args.output_dir / "mgm2_config.json"))
    with (args.output_dir / "fmt_community_prediction_args.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_args(args), handle, indent=2)

    pl.seed_everything(args.seed, workers=True)
    train_dataset = FMTCommunityPredictionDataset(
        community_data_file=args.train_data_file,
        metadata_file=args.train_metadata_file,
        ntv3_embedding_path=args.ntv3_embedding_path,
        min_abundance=config.abundance.min_abundance,
        output_min=config.abundance.output_min,
        output_max=config.abundance.output_max,
        normalize=config.abundance.normalize,
    )
    train_loader, val_loader, train_idx, val_idx = create_fmt_community_prediction_dataloaders(
        train_dataset=train_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        group_column=args.val_group_column,
    )
    print(f"FMT post targets: {len(train_dataset)} | train: {len(train_idx)} | val: {len(val_idx)}")
    print(f"Features: {train_dataset.num_features} | train communities: {len(train_dataset.sample_ids)}")

    module = make_module(config, args)
    trainer = make_trainer(config, args)
    trainer.fit(module, train_loader, val_loader)

    if args.test_data_file is None or args.test_metadata_file is None:
        return

    test_embedding = args.test_ntv3_embedding_path or args.ntv3_embedding_path
    test_dataset = FMTCommunityPredictionDataset(
        community_data_file=args.test_data_file,
        metadata_file=args.test_metadata_file,
        ntv3_embedding_path=test_embedding,
        min_abundance=config.abundance.min_abundance,
        output_min=config.abundance.output_min,
        output_max=config.abundance.output_max,
        normalize=config.abundance.normalize,
    )
    test_loader = create_fmt_community_prediction_test_loader(
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    ckpt_path = resolve_eval_ckpt(trainer, args.eval_checkpoint)
    trainer.test(module, test_loader, ckpt_path=ckpt_path)
    predictions = trainer.predict(module, test_loader, ckpt_path=ckpt_path)
    prediction_file = args.prediction_file or (args.output_dir / "predicted_abundance.h5")
    write_predictions(prediction_file, test_dataset, predictions)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
