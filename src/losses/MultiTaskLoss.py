"""
Multi-task loss implementation for MGM2.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            smoothed_targets = torch.zeros_like(log_probs)
            smoothed_targets.fill_(self.smoothing / (self.num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smoothed_targets * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class MGM2MultiTaskLoss(nn.Module):
    """Loss for masked token-level tasks and optional CLS-to-Qwen3 alignment."""

    def __init__(
        self,
        num_bins: int,
        enable_regression: bool = True,
        enable_qwen3_alignment: bool = False,
        bin_loss_type: str = 'cross_entropy',
        regression_loss_type: str = 'mse',
        lambda_bin: float = 1.0,
        lambda_regression: float = 0.3,
        lambda_qwen3_alignment: float = 0.1,
        qwen3_alignment_temperature: float = 0.07,
        label_smoothing: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        device: str = 'cpu',
    ):
        super().__init__()
        self.num_bins = num_bins
        self.enable_regression = enable_regression
        self.enable_qwen3_alignment = enable_qwen3_alignment
        self.lambda_bin = lambda_bin
        self.lambda_regression = lambda_regression
        self.lambda_qwen3_alignment = lambda_qwen3_alignment
        self.qwen3_alignment_temperature = qwen3_alignment_temperature
        self.device = device

        if bin_loss_type == 'cross_entropy':
            if label_smoothing > 0:
                self.bin_loss = LabelSmoothingLoss(num_classes=num_bins, smoothing=label_smoothing)
            else:
                self.bin_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        elif bin_loss_type == 'focal':
            self.bin_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        else:
            raise ValueError(f'Unknown bin_loss_type: {bin_loss_type}')

        if enable_regression:
            if regression_loss_type == 'mse':
                self.regression_loss = nn.MSELoss(reduction='mean')
            elif regression_loss_type == 'mae':
                self.regression_loss = nn.L1Loss(reduction='mean')
            else:
                raise ValueError(f'Unknown regression_loss_type: {regression_loss_type}')

    def _compute_qwen3_alignment_loss(
        self,
        qwen3_proj: torch.Tensor,
        qwen3_embedding: torch.Tensor,
        qwen3_alignment_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_proj = qwen3_proj[qwen3_alignment_mask]
        valid_targets = qwen3_embedding[qwen3_alignment_mask]

        if valid_proj.size(0) < 2:
            return torch.tensor(0.0, device=qwen3_proj.device)

        valid_proj = F.normalize(valid_proj, p=2, dim=-1)
        valid_targets = F.normalize(valid_targets, p=2, dim=-1)
        similarity = valid_proj @ valid_targets.T
        similarity = similarity / self.qwen3_alignment_temperature
        targets = torch.arange(valid_proj.size(0), device=qwen3_proj.device)
        proj_to_qwen = F.cross_entropy(similarity, targets)
        qwen_to_proj = F.cross_entropy(similarity.T, targets)
        return 0.5 * (proj_to_qwen + qwen_to_proj)

    def forward(self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_dict = {}

        if 'bin_logits' in predictions and 'bin_labels' in labels:
            bin_logits = predictions['bin_logits']
            bin_targets = labels['bin_labels']
            valid_mask = bin_targets != -100
            if valid_mask.any():
                loss_dict['bin_loss'] = self.bin_loss(bin_logits[valid_mask], bin_targets[valid_mask]) * self.lambda_bin
            else:
                loss_dict['bin_loss'] = torch.tensor(0.0, device=bin_logits.device)

        if self.enable_regression and 'regression_preds' in predictions and 'regression_targets' in labels and 'regression_mask' in labels:
            regression_preds = predictions['regression_preds']
            regression_targets = labels['regression_targets']
            regression_mask = labels['regression_mask']
            if regression_mask.any():
                regression_loss = self.regression_loss(regression_preds[regression_mask], regression_targets[regression_mask])
                loss_dict['regression_loss'] = regression_loss * self.lambda_regression
            else:
                loss_dict['regression_loss'] = torch.tensor(0.0, device=regression_preds.device)

        if self.enable_qwen3_alignment and 'qwen3_proj' in predictions and 'qwen3_embedding' in labels and 'qwen3_alignment_mask' in labels:
            qwen3_alignment_mask = labels['qwen3_alignment_mask'].bool()
            alignment_loss = self._compute_qwen3_alignment_loss(
                predictions['qwen3_proj'],
                labels['qwen3_embedding'],
                qwen3_alignment_mask,
            )
            loss_dict['qwen3_alignment_loss'] = alignment_loss * self.lambda_qwen3_alignment

        loss_dict['total_loss'] = sum(loss_dict.values())
        return loss_dict

    def set_lambda_weights(
        self,
        lambda_bin: float,
        lambda_regression: Optional[float] = None,
        lambda_qwen3_alignment: Optional[float] = None,
    ):
        self.lambda_bin = lambda_bin
        if lambda_regression is not None:
            self.lambda_regression = lambda_regression
        if lambda_qwen3_alignment is not None:
            self.lambda_qwen3_alignment = lambda_qwen3_alignment

    def get_weights(self) -> Dict[str, float]:
        weights = {'lambda_bin': self.lambda_bin}
        if self.enable_regression:
            weights['lambda_regression'] = self.lambda_regression
        if self.enable_qwen3_alignment:
            weights['lambda_qwen3_alignment'] = self.lambda_qwen3_alignment
        return weights
