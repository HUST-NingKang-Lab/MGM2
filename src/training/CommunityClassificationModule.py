"""PyTorch Lightning module for downstream community classification."""

import math
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.configs.Config import MGM2Config
from src.models import MGM2CommunityClassificationModel


class CommunityClassificationModule(pl.LightningModule):
    """Fine-tune an MGM2 backbone for sample-level community labels."""

    MODEL_INPUT_KEYS = {
        'input_ids',
        'embeddings',
        'abundance_values',
        'attention_mask',
        'labels',
    }

    def __init__(
        self,
        config: MGM2Config,
        num_classes: int,
        class_names: Optional[list] = None,
        freeze_backbone: bool = False,
        pretrained_checkpoint: Optional[str] = None,
        pooling_mode: str = 'mean_pool',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self.num_classes = int(num_classes)
        self.class_names = class_names or []
        self.pooling_mode = pooling_mode
        self.model = MGM2CommunityClassificationModel(
            config=config.model.__dict__,
            num_classes=self.num_classes,
            mlp_dropout=config.model.mlp_dropout,
            pooling_mode=pooling_mode,
        )
        if pretrained_checkpoint:
            self.load_pretrained_backbone_from_checkpoint(pretrained_checkpoint)
        if freeze_backbone:
            for parameter in self.model.backbone.parameters():
                parameter.requires_grad = False

    def load_pretrained_backbone_from_checkpoint(self, checkpoint_path: str):
        """Load only backbone weights from a trusted MGM2 Lightning checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        return self.model.load_pretrained_backbone(state_dict, strict=True)

    def _extract_model_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value for key, value in batch.items() if key in self.MODEL_INPUT_KEYS}

    def forward(self, **batch):
        return self.model(**self._extract_model_inputs(batch))

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self._log_metrics(outputs['classification_logits'], batch['labels'], 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metrics(outputs['classification_logits'], batch['labels'], 'val')
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metrics(outputs['classification_logits'], batch['labels'], 'test')
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        logits = outputs['classification_logits']
        probabilities = torch.softmax(logits, dim=-1)
        return {
            'sample_id': batch.get('sample_id'),
            'label_id': batch.get('labels'),
            'pred_id': probabilities.argmax(dim=-1),
            'probability': probabilities.max(dim=-1).values,
            'logits': logits,
            'probabilities': probabilities,
        }

    def _log_metrics(self, logits: torch.Tensor, labels: torch.Tensor, stage: str):
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()
        self.log(f'{stage}/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        if self.num_classes > 1:
            macro_recall_values = []
            macro_precision_values = []
            for class_id in range(self.num_classes):
                pred_mask = preds == class_id
                label_mask = labels == class_id
                tp = (pred_mask & label_mask).sum().float()
                recall_den = label_mask.sum().float()
                precision_den = pred_mask.sum().float()
                if recall_den > 0:
                    macro_recall_values.append(tp / recall_den)
                if precision_den > 0:
                    macro_precision_values.append(tp / precision_den)
            if macro_recall_values:
                self.log(f'{stage}/macro_recall', torch.stack(macro_recall_values).mean(), on_step=False, on_epoch=True)
            if macro_precision_values:
                self.log(f'{stage}/macro_precision', torch.stack(macro_precision_values).mean(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, self.model.parameters()),
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
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    def _build_warmup_cosine_scheduler(self, optimizer: torch.optim.Optimizer):
        total_steps = self._infer_total_training_steps()
        warmup_steps = int(getattr(self.config.training, 'warmup_steps', 0))
        min_lr_ratio = float(getattr(self.config.training, 'min_lr_ratio', 0.1))

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return 0.1 + 0.9 * (current_step / float(warmup_steps))
            decay_steps = max(total_steps - warmup_steps, 1)
            decay_progress = min(max((current_step - warmup_steps) / float(decay_steps), 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _infer_total_training_steps(self) -> int:
        warmup_steps = int(getattr(self.config.training, 'warmup_steps', 0))
        try:
            estimated_steps = int(self.trainer.estimated_stepping_batches)
        except (AttributeError, RuntimeError, TypeError):
            estimated_steps = 0
        return max(estimated_steps, warmup_steps + 1, 1)
