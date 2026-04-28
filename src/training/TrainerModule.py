"""
MGM2 PyTorch Lightning module.

Training logic for masked abundance modeling with optional CLS-to-Qwen3 alignment.
"""

import logging
import math
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..configs.Config import MGM2Config

logger = logging.getLogger(__name__)


class MGM2LightningModule(pl.LightningModule):
    MODEL_INPUT_KEYS = {
        'input_ids',
        'embeddings',
        'abundance_values',
        'attention_mask',
        'bin_labels',
        'regression_targets',
        'regression_mask',
    }

    def __init__(self, config: 'MGM2Config', abundance_preprocessor=None, bin_manager=None):
        from ..losses import MGM2MultiTaskLoss
        from ..models import MGM2Model

        super().__init__()
        self.save_hyperparameters(ignore=['abundance_preprocessor', 'bin_manager'])

        self.config = config
        self.abundance_preprocessor = abundance_preprocessor
        self.bin_manager = bin_manager

        self.model = MGM2Model(
            config=config.model.__dict__,
            num_bins=config.abundance.num_bins,
            enable_regression=config.abundance.enable_regression,
            enable_qwen3_alignment=config.qwen3_alignment.enabled,
            qwen3_projection_dim=config.qwen3_alignment.projection_dim,
            mlp_dropout=config.model.mlp_dropout,
        )

        self.criterion = MGM2MultiTaskLoss(
            num_bins=config.abundance.num_bins,
            enable_regression=config.abundance.enable_regression,
            enable_qwen3_alignment=config.qwen3_alignment.enabled,
            bin_loss_type=config.loss.bin_loss_type,
            regression_loss_type=config.loss.regression_loss_type,
            lambda_bin=config.loss.lambda_bin,
            lambda_regression=config.loss.lambda_regression,
            lambda_qwen3_alignment=config.qwen3_alignment.loss_weight,
            qwen3_alignment_temperature=config.qwen3_alignment.temperature,
            focal_alpha=config.loss.focal_alpha,
            focal_gamma=config.loss.focal_gamma,
            device=self.device.type,
        )

    def _extract_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value for key, value in batch.items() if key in self.MODEL_INPUT_KEYS}

    def forward(self, **batch):
        return self.model(**self._extract_model_inputs(batch))

    def _split_predictions_and_labels(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        predictions = {'bin_logits': outputs['bin_logits']}
        labels = {'bin_labels': batch['bin_labels']}
        if 'regression_preds' in outputs:
            predictions['regression_preds'] = outputs['regression_preds']
        if 'regression_targets' in batch and 'regression_mask' in batch:
            labels['regression_targets'] = batch['regression_targets']
            labels['regression_mask'] = batch['regression_mask']
        if 'qwen3_proj' in outputs:
            predictions['qwen3_proj'] = outputs['qwen3_proj']
        if 'qwen3_embedding' in batch and 'qwen3_alignment_mask' in batch:
            labels['qwen3_embedding'] = batch['qwen3_embedding']
            labels['qwen3_alignment_mask'] = batch['qwen3_alignment_mask']
        return predictions, labels

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        predictions, labels = self._split_predictions_and_labels(outputs, batch)
        loss_dict = self.criterion(predictions, labels)
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, on_step=True, on_epoch=True, prog_bar=(name == 'total_loss'))
        self._log_metrics(predictions, labels, 'train')
        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        predictions, labels = self._split_predictions_and_labels(outputs, batch)
        loss_dict = self.criterion(predictions, labels)
        for name, value in loss_dict.items():
            self.log(f'val/{name}', value, on_step=False, on_epoch=True, prog_bar=(name == 'total_loss'))
        self._log_metrics(predictions, labels, 'val')
        return loss_dict['total_loss']

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        predictions, labels = self._split_predictions_and_labels(outputs, batch)
        loss_dict = self.criterion(predictions, labels)
        for name, value in loss_dict.items():
            self.log(f'test/{name}', value, on_step=False, on_epoch=True, prog_bar=(name == 'total_loss'))
        self._log_metrics(predictions, labels, 'test')
        return loss_dict['total_loss']

    def _log_metrics(self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], stage: str):
        if 'bin_logits' in predictions and 'bin_labels' in labels:
            valid_mask = labels['bin_labels'] != -100
            if valid_mask.any():
                valid_preds = predictions['bin_logits'][valid_mask].argmax(dim=-1)
                valid_targets = labels['bin_labels'][valid_mask]
                accuracy = (valid_preds == valid_targets).float().mean()
                self.log(f'{stage}/bin_accuracy', accuracy, prog_bar=True)

        if 'regression_preds' in predictions and 'regression_targets' in labels and 'regression_mask' in labels:
            regression_mask = labels['regression_mask']
            if regression_mask.any():
                valid_preds = predictions['regression_preds'][regression_mask]
                valid_targets = labels['regression_targets'][regression_mask]
                mse = F.mse_loss(valid_preds, valid_targets)
                mae = F.l1_loss(valid_preds, valid_targets)
                self.log(f'{stage}/regression_mse', mse)
                self.log(f'{stage}/regression_mae', mae)
                if len(valid_preds) > 1:
                    corr = torch.corrcoef(torch.stack([valid_preds.flatten(), valid_targets.flatten()]))[0, 1]
                    self.log(f'{stage}/regression_corr', corr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        scheduler_name = getattr(self.config.training, 'lr_scheduler', 'warmup_cosine')
        if scheduler_name == 'constant':
            return optimizer
        if scheduler_name == 'linear_warmup':
            scheduler = self._build_linear_warmup_scheduler(optimizer)
        elif scheduler_name == 'warmup_cosine':
            scheduler = self._build_warmup_cosine_scheduler(optimizer)
        else:
            raise ValueError(f'Unsupported lr_scheduler: {scheduler_name}')

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }

    def _build_linear_warmup_scheduler(self, optimizer: torch.optim.Optimizer):
        warmup_steps = int(getattr(self.config.training, 'warmup_steps', 0))
        if warmup_steps <= 0:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )

    def _build_warmup_cosine_scheduler(self, optimizer: torch.optim.Optimizer):
        total_steps = self._infer_total_training_steps()
        lr_lambda = self._build_warmup_cosine_lr_lambda(total_steps)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _infer_total_training_steps(self) -> int:
        warmup_steps = int(getattr(self.config.training, 'warmup_steps', 0))
        try:
            estimated_steps = int(self.trainer.estimated_stepping_batches)
        except (AttributeError, RuntimeError, TypeError):
            estimated_steps = 0
        return max(estimated_steps, warmup_steps + 1, 1)

    def _build_warmup_cosine_lr_lambda(self, total_steps: int):
        warmup_steps = int(getattr(self.config.training, 'warmup_steps', 0))
        min_lr_ratio = float(getattr(self.config.training, 'min_lr_ratio', 0.1))
        total_steps = max(int(total_steps), warmup_steps + 1, 1)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                warmup_progress = current_step / float(warmup_steps)
                return 0.1 + 0.9 * warmup_progress

            decay_steps = max(total_steps - warmup_steps, 1)
            decay_progress = min(max((current_step - warmup_steps) / float(decay_steps), 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        return lr_lambda

    def on_train_start(self):
        logger.info(f'Starting training: {self.config.experiment_name}')
        logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        if self.logger is not None:
            self.logger.log_hyperparams(self.config.to_dict())

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(path) / 'pytorch_model.bin')
        self.config.save(str(Path(path) / 'mgm2_config.json'))
        logger.info(f'Model and config saved to {path}')

    @classmethod
    def load_pretrained(cls, path: str):
        config = MGM2Config.load(str(Path(path) / 'mgm2_config.json'))
        module = cls(config)
        state_dict = torch.load(Path(path) / 'pytorch_model.bin', map_location='cpu')
        module.model.load_state_dict(state_dict)
        logger.info(f'Model and config loaded from {path}')
        return module
