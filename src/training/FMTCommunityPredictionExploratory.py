#!/usr/bin/env python3
"""Train exploratory MGM2 FMT community prediction models."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
import torch

from src.configs.Config import MGM2Config
from src.data.FMTDataset import (
    FMTCommunityPredictionDataset,
    create_fmt_community_prediction_dataloaders,
    create_fmt_community_prediction_test_loader,
    grouped_kfold_indices,
    grouped_train_val_indices,
)
from src.training.FMTCommunityPrediction import (
    _jsonable_args,
    create_subset_loader,
    make_trainer,
    resolve_eval_ckpt,
    write_predictions,
)
from src.training.FMTCommunityPredictionExploratoryModule import FMTCommunityPredictionExploratoryLightningModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an exploratory MGM2 FMT model with explicit delta fusion",
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
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/FMT/community_prediction_exploratory/delta_fusion"))
    parser.add_argument("--experiment_name", type=str, default="fmt_community_prediction_delta_fusion")
    parser.add_argument("--prediction_file", type=Path, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_group_column", type=str, default="fmt_id")
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_label_column", type=str, default="response_label")
    parser.add_argument("--outer_cv", action="store_true")
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
    parser.add_argument("--delta_loss_weight", type=float, default=0.5)
    parser.add_argument("--use_gated_delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def make_module(config: MGM2Config, args) -> FMTCommunityPredictionExploratoryLightningModule:
    module = FMTCommunityPredictionExploratoryLightningModule(
        config=config,
        cross_attention_heads=args.cross_attention_heads,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        raw_zero_threshold=args.raw_zero_threshold,
        renormalize_predictions=args.renormalize_predictions,
        delta_loss_weight=args.delta_loss_weight,
        use_gated_delta=args.use_gated_delta,
    )
    if args.pretrained_checkpoint is not None:
        module.load_pretrained_backbone_from_checkpoint(str(args.pretrained_checkpoint))
        print(f"Loaded pretrained MGM2 backbone from {args.pretrained_checkpoint}")
    return module


def main():
    args = parse_args()
    for required in [args.train_data_file, args.train_metadata_file, args.ntv3_embedding_path]:
        if not required.exists():
            raise FileNotFoundError(required)
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
    with (args.output_dir / "fmt_community_prediction_exploratory_args.json").open("w", encoding="utf-8") as handle:
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
            f"Exploratory FMT post targets: {len(train_dataset)} | outer fold {args.fold_index + 1}/{args.num_folds} "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
    else:
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
        test_idx = None
        print(f"Exploratory FMT post targets: {len(train_dataset)} | train: {len(train_idx)} | val: {len(val_idx)}")
    print(f"Features: {train_dataset.num_features} | train communities: {len(train_dataset.sample_ids)}")

    module = make_module(config, args)
    trainer = make_trainer(config, args)
    trainer.fit(module, train_loader, val_loader)

    if args.outer_cv:
        test_dataset = train_dataset
        test_loader = create_subset_loader(train_dataset, test_idx, args, shuffle=False)
    elif args.test_data_file is None or args.test_metadata_file is None:
        return
    else:
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
    eval_devices = [args.devices[0]] if isinstance(args.devices, list) and args.devices else None
    eval_trainer = make_trainer(config, args, devices=eval_devices, callbacks=False)
    eval_trainer.test(module, test_loader, ckpt_path=ckpt_path)
    predictions = eval_trainer.predict(module, test_loader, ckpt_path=ckpt_path)
    prediction_file = args.prediction_file or (args.output_dir / "predicted_abundance.h5")
    write_predictions(prediction_file, test_dataset, predictions)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
