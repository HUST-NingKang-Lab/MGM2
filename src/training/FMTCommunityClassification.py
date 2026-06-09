#!/usr/bin/env python3
"""Train MGM2 FMT donor-recipient models for response classification."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
from src.data.FMTDataset import (
    FMTCommunityClassificationCollator,
    FMTCommunityClassificationDataset,
    create_fmt_community_classification_dataloaders,
    create_fmt_community_classification_test_loader,
    grouped_kfold_indices,
    grouped_train_val_indices,
)
from src.training.FMTCommunityClassificationModule import FMTCommunityClassificationLightningModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MGM2 FMT response classifier that receives donor and recipient communities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_dir = Path("data/downstreams/FMT")
    parser.add_argument("--train_data_file", type=Path, default=data_dir / "community_train.pkl")
    parser.add_argument("--train_metadata_file", type=Path, default=data_dir / "response_train_meta_split.csv")
    parser.add_argument("--val_metadata_file", type=Path, default=None)
    parser.add_argument("--test_data_file", type=Path, default=data_dir / "community_test.pkl")
    parser.add_argument("--test_metadata_file", type=Path, default=data_dir / "response_test_meta_split.csv")
    parser.add_argument("--ntv3_embedding_path", type=Path, default=data_dir / "community_train.resolved_embeddings.h5")
    parser.add_argument("--test_ntv3_embedding_path", type=Path, default=data_dir / "community_test.resolved_embeddings.h5")
    parser.add_argument("--config", type=Path, default=project_root / "src" / "configs" / "mgm2_default.json")
    parser.add_argument("--pretrained_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT/community_classification"))
    parser.add_argument("--experiment_name", type=str, default="fmt_community_classification")
    parser.add_argument("--prediction_file", type=Path, default=None)
    parser.add_argument("--class_names", nargs="+", default=None)
    parser.add_argument("--sequence_mode", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--val_group_column", type=str, default="fmt_id")
    parser.add_argument("--fold_label_column", type=str, default="response_label")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--outer_cv", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross_attention_heads", type=int, default=None)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
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


def make_module(config: MGM2Config, args, class_names: List[str]) -> FMTCommunityClassificationLightningModule:
    module = FMTCommunityClassificationLightningModule(
        config=config,
        num_classes=len(class_names),
        class_names=class_names,
        cross_attention_heads=args.cross_attention_heads,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
    )
    if args.pretrained_checkpoint is not None:
        module.load_pretrained_backbone_from_checkpoint(str(args.pretrained_checkpoint))
        print(f"Loaded pretrained MGM2 backbone from {args.pretrained_checkpoint}")
    return module


def make_trainer(config: MGM2Config, args, devices=None, callbacks=True) -> pl.Trainer:
    callback_list = []
    if callbacks:
        callback_list = [
            ModelCheckpoint(
                dirpath=args.output_dir / "checkpoints",
                filename="fmt-response-{epoch:02d}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(monitor="val/loss", patience=12, mode="min"),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
        ]
    selected_devices = args.devices if devices is None else devices
    if args.accelerator == "cpu" or (args.accelerator == "auto" and not torch.cuda.is_available()):
        selected_devices = 1
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=selected_devices,
        precision=args.precision,
        callbacks=callback_list,
        logger=[
            TensorBoardLogger(save_dir=args.output_dir / "logs", name=args.experiment_name),
            CSVLogger(save_dir=args.output_dir / "logs", name=args.experiment_name),
        ] if callbacks else False,
        gradient_clip_val=config.training.gradient_clip_val,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=5,
    )


def resolve_eval_ckpt(trainer: pl.Trainer, mode: str):
    if mode == "none":
        return None
    checkpoint = getattr(trainer, "checkpoint_callback", None)
    if checkpoint is None:
        return None
    if mode == "best" and checkpoint.best_model_path:
        return checkpoint.best_model_path
    if mode == "last" and checkpoint.last_model_path:
        return checkpoint.last_model_path
    return None


def create_subset_loader(dataset, indices: np.ndarray, args, shuffle: bool) -> DataLoader:
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": FMTCommunityClassificationCollator(),
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "shuffle": shuffle,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, **loader_kwargs)


def _flatten_predictions(predictions: Iterable[Any]) -> List[Dict[str, Any]]:
    flattened = []
    for item in predictions:
        if isinstance(item, list):
            flattened.extend(_flatten_predictions(item))
        else:
            flattened.append(item)
    return flattened


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique_scores) != len(scores):
        rank_sums = np.zeros(len(unique_scores), dtype=np.float64)
        np.add.at(rank_sums, inverse, ranks)
        ranks = rank_sums[inverse] / counts[inverse]
    pos_rank_sum = ranks[positives].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / (np.arange(len(labels)) + 1)
    return float((precision * (sorted_labels == 1)).sum() / n_pos)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    metrics: Dict[str, float] = {
        "accuracy": float((preds == labels).mean()) if len(labels) else float("nan"),
    }
    recalls = []
    f1_values = []
    for class_id in range(len(class_names)):
        pred_mask = preds == class_id
        label_mask = labels == class_id
        tp = float((pred_mask & label_mask).sum())
        fp = float((pred_mask & ~label_mask).sum())
        fn = float((~pred_mask & label_mask).sum())
        if label_mask.any():
            recalls.append(tp / max(tp + fn, 1.0))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if (tp + fp) > 0 or (tp + fn) > 0:
            f1_values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    metrics["balanced_accuracy"] = float(np.mean(recalls)) if recalls else float("nan")
    metrics["macro_f1"] = float(np.mean(f1_values)) if f1_values else float("nan")
    if len(class_names) == 2:
        positive_id = class_names.index("responder") if "responder" in class_names else 1
        binary_labels = (labels == positive_id).astype(int)
        scores = probs[:, positive_id]
        metrics["roc_auc"] = _binary_auc(binary_labels, scores)
        metrics["average_precision"] = _average_precision(binary_labels, scores)
    return metrics


def write_predictions(output_path: Path, class_names: List[str], predictions: Iterable[Any], extra: Dict[str, Any]) -> None:
    chunks = _flatten_predictions(predictions)
    if not chunks:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    probabilities = []
    labels = []
    for chunk in chunks:
        probs = chunk["probabilities"].detach().float().cpu().numpy().astype(np.float32)
        logits = chunk["logits"].detach().float().cpu().numpy().astype(np.float32)
        probabilities.append(probs)
        labels.append(chunk["label_id"].detach().cpu().numpy().astype(np.int64))
        for idx in range(probs.shape[0]):
            row = {
                "target_id": chunk["target_id"][idx],
                "fmt_id": chunk["fmt_id"][idx],
                "study": chunk["study"][idx],
                "indication": chunk["indication"][idx],
                "response_label": chunk["response_label"][idx],
                "donor_sample_id": chunk["donor_sample_id"][idx],
                "recipient_sample_id": chunk["recipient_sample_id"][idx],
                "label_id": int(chunk["label_id"][idx]),
                "pred_id": int(chunk["pred_id"][idx]),
                "pred_label": class_names[int(chunk["pred_id"][idx])],
            }
            for class_id, class_name in enumerate(class_names):
                safe_name = class_name.replace("-", "_")
                row[f"prob_{safe_name}"] = float(probs[idx, class_id])
                row[f"logit_{safe_name}"] = float(logits[idx, class_id])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    prob_matrix = np.concatenate(probabilities, axis=0)
    label_vector = np.concatenate(labels, axis=0)
    metrics = compute_metrics(label_vector, prob_matrix, class_names)
    metrics.update(extra)
    with (output_path.parent / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_path.parent / "class_map.json").open("w", encoding="utf-8") as handle:
        json.dump({name: idx for idx, name in enumerate(class_names)}, handle, indent=2)
    print(pd.Series(metrics).to_string())


def main():
    args = parse_args()
    for required in [args.train_data_file, args.train_metadata_file, args.ntv3_embedding_path]:
        if not required.exists():
            raise FileNotFoundError(required)

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
    with (args.output_dir / "fmt_community_classification_args.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_args(args), handle, indent=2)

    pl.seed_everything(args.seed, workers=True)
    train_dataset = FMTCommunityClassificationDataset(
        community_data_file=args.train_data_file,
        metadata_file=args.train_metadata_file,
        ntv3_embedding_path=args.ntv3_embedding_path,
        min_abundance=config.abundance.min_abundance,
        output_min=config.abundance.output_min,
        output_max=config.abundance.output_max,
        normalize=config.abundance.normalize,
        class_names=args.class_names,
        sequence_mode=args.sequence_mode,
    )
    class_names = train_dataset.class_names
    if args.outer_cv:
        outer_folds = grouped_kfold_indices(
            train_dataset.metadata,
            n_splits=args.num_folds,
            seed=args.seed,
            group_column=args.val_group_column,
            label_column=args.fold_label_column,
        )
        outer_train_idx, test_idx = outer_folds[args.fold_index]
        inner_metadata = train_dataset.metadata.iloc[outer_train_idx]
        inner_train_rel, inner_val_rel = grouped_train_val_indices(
            inner_metadata,
            args.val_ratio,
            args.seed + args.fold_index + 1000,
            group_column=args.val_group_column,
        )
        train_idx = outer_train_idx[inner_train_rel]
        val_idx = outer_train_idx[inner_val_rel]
        train_loader = create_subset_loader(train_dataset, train_idx, args, shuffle=True)
        val_loader = create_subset_loader(train_dataset, val_idx, args, shuffle=False)
        print(
            f"FMT response events: {len(train_dataset)} | outer fold {args.fold_index + 1}/{args.num_folds} "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
    elif args.val_metadata_file is not None:
        if not args.val_metadata_file.exists():
            raise FileNotFoundError(args.val_metadata_file)
        val_dataset = FMTCommunityClassificationDataset(
            community_data_file=args.train_data_file,
            metadata_file=args.val_metadata_file,
            ntv3_embedding_path=args.ntv3_embedding_path,
            min_abundance=config.abundance.min_abundance,
            output_min=config.abundance.output_min,
            output_max=config.abundance.output_max,
            normalize=config.abundance.normalize,
            class_names=class_names,
            sequence_mode=args.sequence_mode,
        )
        train_idx = np.arange(len(train_dataset))
        val_idx = np.arange(len(val_dataset))
        test_idx = None
        train_loader = create_subset_loader(train_dataset, train_idx, args, shuffle=True)
        val_loader = create_fmt_community_classification_test_loader(
            test_dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
        print(
            f"FMT response events: train={len(train_idx)} val={len(val_idx)} "
            f"from explicit validation metadata"
        )
    else:
        train_loader, val_loader, train_idx, val_idx = create_fmt_community_classification_dataloaders(
            train_dataset=train_dataset,
            fold_index=args.fold_index,
            n_splits=args.num_folds,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            group_column=args.val_group_column,
        )
        test_idx = None
        print(
            f"FMT response events: {len(train_dataset)} | fold {args.fold_index + 1}/{args.num_folds} "
            f"train={len(train_idx)} val={len(val_idx)}"
        )
    print(f"Classes: {class_names} | Features: {train_dataset.num_features}")

    module = make_module(config, args, class_names)
    trainer = make_trainer(config, args)
    trainer.fit(module, train_loader, val_loader)

    if args.outer_cv:
        test_dataset = train_dataset
        test_loader = create_subset_loader(train_dataset, test_idx, args, shuffle=False)
        test_event_count = int(len(test_idx))
    elif args.test_data_file is None or args.test_metadata_file is None:
        return
    else:
        test_embedding = args.test_ntv3_embedding_path or args.ntv3_embedding_path
        test_dataset = FMTCommunityClassificationDataset(
            community_data_file=args.test_data_file,
            metadata_file=args.test_metadata_file,
            ntv3_embedding_path=test_embedding,
            min_abundance=config.abundance.min_abundance,
            output_min=config.abundance.output_min,
            output_max=config.abundance.output_max,
            normalize=config.abundance.normalize,
            class_names=class_names,
            sequence_mode=args.sequence_mode,
        )
        test_loader = create_fmt_community_classification_test_loader(
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
        test_event_count = int(len(test_dataset))
    ckpt_path = resolve_eval_ckpt(trainer, args.eval_checkpoint)
    eval_devices = [args.devices[0]] if isinstance(args.devices, list) and args.devices else None
    eval_trainer = make_trainer(config, args, devices=eval_devices, callbacks=False)
    eval_trainer.test(module, test_loader, ckpt_path=ckpt_path)
    predictions = eval_trainer.predict(module, test_loader, ckpt_path=ckpt_path)
    prediction_file = args.prediction_file or (args.output_dir / "test_predictions.csv")
    write_predictions(
        prediction_file,
        class_names,
        predictions,
        {
            "fold_index": args.fold_index,
            "num_folds": args.num_folds,
            "train_events": int(len(train_idx)),
            "val_events": int(len(val_idx)),
            "test_events": test_event_count,
            "outer_cv": bool(args.outer_cv),
            "eval_checkpoint": str(ckpt_path) if ckpt_path else "current",
        },
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
