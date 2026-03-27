#!/usr/bin/env python3
"""
MGM2 dual-task pretraining entry point with optional Qwen3 contrastive alignment.

Example:
    python src/training/Pretrain.py         --data_file data/pretrain_dataset.pkl         --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5         --config src/configs/mgm2_default.json

    python src/training/Pretrain.py         --data_file data/pretrain_dataset.pkl         --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5         --enable_qwen3_alignment         --qwen3_embedding_path qwen3_embeddings/microbeatlas_keywords_emb.h5
"""

import argparse
import sys
from pathlib import Path

# Allow running the script from outside the repository root.
project_root = Path(__file__).resolve().parents[2]
default_config_path = project_root / 'src' / 'configs' / 'mgm2_default.json'
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch

from src.configs.Config import MGM2Config
from src.data.Dataset import create_dataloaders_mgm2
from src.training.TrainerModule import MGM2LightningModule


def _optional_path(value):
    if value is None:
        return None
    return Path(value)


def _require_existing_path(path_value, argument_name: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f'{argument_name} not found: {path}')
    return path


def parse_args():
    """Parse command line arguments for pretraining."""
    parser = argparse.ArgumentParser(
        description='Pretrain the MGM2 dual-task masked abundance model with optional Qwen3 contrastive alignment',
        epilog=(
            'Example:\n'
            '  python src/training/Pretrain.py --data_file data/pretrain_dataset.pkl '
            '--ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5 --config src/configs/mgm2_default.json\n'
            '  python src/training/Pretrain.py --data_file data/pretrain_dataset.pkl '
            '--ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5 '
            '--enable_qwen3_alignment --qwen3_embedding_path qwen3_embeddings/microbeatlas_keywords_emb.h5'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--data_file', type=Path, required=True, help='Path to preprocessed dataset (.pkl file)')
    parser.add_argument('--ntv3_embedding_path', type=Path, required=True, help='Path to NTv3 embeddings (.h5 file)')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file (JSON). If not provided, loads src/configs/mgm2_default.json')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Masking probability')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='GPU devices to use')
    parser.add_argument(
        '--qwen3_embedding_path',
        type=Path,
        default=None,
        help='Optional path to Qwen3 keyword embeddings (.h5) for sample-level contrastive alignment',
    )
    parser.add_argument('--enable_qwen3_alignment', action='store_true', help='Enable optional CLS-to-Qwen3 contrastive alignment during pretraining')
    parser.add_argument('--output_dir', type=Path, default=Path('./outputs_mgm2'), help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='mgm2_experiment', help='Experiment name')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--resume', type=Path, default=None, help='Resume from checkpoint (path to .ckpt file)')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a single batch for testing')

    return parser.parse_args()


def create_config(args):
    """Create or load configuration after validating required paths."""
    config_path = _optional_path(getattr(args, 'config', None))
    data_file = _require_existing_path(getattr(args, 'data_file'), '--data_file')
    ntv3_embedding_path = _require_existing_path(
        getattr(args, 'ntv3_embedding_path', getattr(args, 'ntv3_embedding', None)),
        '--ntv3_embedding_path',
    )
    qwen3_embedding_path = _optional_path(getattr(args, 'qwen3_embedding_path', None))
    resume_path = _optional_path(getattr(args, 'resume', None))

    if config_path is not None:
        config_path = _require_existing_path(config_path, '--config')
        config = MGM2Config.load(str(config_path))
        print(f'Loaded config from {config_path}')
    else:
        config_path = _require_existing_path(default_config_path, 'default config')
        config = MGM2Config.load(str(config_path))
        print(f'Loaded default config from {config_path}')

    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.devices = args.devices
    config.ntv3_embedding_path = str(ntv3_embedding_path)
    config.data_path = str(data_file)
    config.output_dir = str(Path(args.output_dir))
    config.experiment_name = args.experiment_name

    if resume_path is not None:
        _require_existing_path(resume_path, '--resume')

    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if qwen3_embedding_path is not None:
        qwen3_embedding_path = _require_existing_path(qwen3_embedding_path, '--qwen3_embedding_path')
        config.qwen3_alignment.embedding_path = str(qwen3_embedding_path)
    if args.enable_qwen3_alignment:
        config.qwen3_alignment.enabled = True
    if config.qwen3_alignment.enabled:
        if not config.qwen3_alignment.embedding_path:
            raise FileNotFoundError('Qwen3 alignment is enabled but --qwen3_embedding_path was not provided')
        _require_existing_path(config.qwen3_alignment.embedding_path, '--qwen3_embedding_path')

    if not torch.isclose(
        torch.tensor(config.training.train_ratio + config.training.val_ratio, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    ):
        raise ValueError('Pretraining requires training.train_ratio + training.val_ratio == 1.0')

    return config


def summarize_experiment_config(config: MGM2Config) -> str:
    """Return a compact summary of the key experiment settings."""
    qwen3_status = 'enabled' if config.qwen3_alignment.enabled else 'disabled'
    lines = [
        f'Experiment: {config.experiment_name}',
        f'Output directory: {config.output_dir}',
        f'seed: {config.seed}',
        f'binning_method: {config.abundance.binning_method}',
        f'train_ratio: {config.training.train_ratio}',
        f'val_ratio: {config.training.val_ratio}',
        f'regression_top_k_bins: {config.abundance.regression_top_k_bins}',
        f'qwen3_alignment: {qwen3_status}',
    ]
    if config.qwen3_alignment.enabled and config.qwen3_alignment.embedding_path:
        lines.append(f'qwen3_embedding_path: {config.qwen3_alignment.embedding_path}')
    return '\n'.join(lines)


def create_callbacks(config):
    """Create the standard training callbacks."""
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.output_dir) / 'checkpoints',
        filename='mgm2-{epoch:02d}-{val/total_loss:.4f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor='val/total_loss',
        patience=10,
        mode='min',
        verbose=True,
    )
    callbacks.append(early_stop_callback)

    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(RichProgressBar())
    return callbacks


def main():
    """Main pretraining function."""
    args = parse_args()
    config = create_config(args)
    print()
    print(summarize_experiment_config(config))
    print()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(Path(config.output_dir) / 'config.json')

    pl.seed_everything(config.seed, workers=True)

    print('Creating data loaders...')
    train_loader, val_loader = create_dataloaders_mgm2(
        data_path=config.data_path,
        ntv3_embedding_path=config.ntv3_embedding_path,
        abundance_bin_config=config.abundance.__dict__,
        training_config={**config.training.__dict__, 'output_dir': config.output_dir, 'seed': config.seed},
        qwen3_alignment_config=config.qwen3_alignment.__dict__,
        mask_prob=args.mask_prob,
        num_workers=args.num_workers,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        include_test_loader=False,
    )

    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')

    print('\nInitializing model...')
    lightning_module = MGM2LightningModule(config=config)

    total_params = sum(p.numel() for p in lightning_module.parameters())
    trainable_params = sum(p.numel() for p in lightning_module.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    callbacks = create_callbacks(config)
    tb_logger = TensorBoardLogger(save_dir=Path(config.output_dir) / 'logs', name=config.experiment_name)
    csv_logger = CSVLogger(save_dir=Path(config.output_dir) / 'logs', name=config.experiment_name)

    print('\nCreating trainer...')
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        fast_dev_run=args.fast_dev_run,
        deterministic=False,
        log_every_n_steps=10,
    )

    if args.resume:
        print(f'\nResuming from checkpoint: {args.resume}')

    print('\nStarting training...')
    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=str(args.resume) if args.resume else None)


    print(f'\nPretraining complete. Results saved to {config.output_dir}')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
