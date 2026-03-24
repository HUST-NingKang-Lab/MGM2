"""
MGM2 main model.

Masked dual-task abundance model with an optional CLS-to-Qwen3 alignment task.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .BaseModel import BaseModel
from .Heads import AbundanceBinHead, Qwen3AlignmentHead, RegressionHead


class MGM2Model(nn.Module):
    """Shared-backbone model with token-level abundance tasks and optional CLS alignment."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_bins: int = 5,
        enable_regression: bool = True,
        enable_qwen3_alignment: bool = False,
        qwen3_projection_dim: Optional[int] = None,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = config.get('hidden_size', 512)
        self.vocab_size = config.get('vocab_size', 10000)
        self.embedding_dim = config.get('embedding_dim', 1536)
        self.num_hidden_layers = config.get('num_hidden_layers', 12)
        self.num_attention_heads = config.get('num_attention_heads', 8)
        self.intermediate_size = config.get('intermediate_size', 2048)
        self.max_position_embeddings = config.get('max_position_embeddings', 2048)
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.1)
        self.attention_probs_dropout_prob = config.get('attention_probs_dropout_prob', 0.1)

        self.backbone = BaseModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=config.get('pad_token_id', -1),
            cls_token_id=config.get('cls_token_id', -1),
        )

        self.enable_regression = enable_regression
        self.enable_qwen3_alignment = enable_qwen3_alignment
        self.num_bins = num_bins
        self.qwen3_projection_dim = qwen3_projection_dim or config.get('qwen3_projection_dim')

        self.bin_head = AbundanceBinHead(
            hidden_size=self.hidden_size,
            num_bins=num_bins,
            dropout=mlp_dropout,
        )
        if self.enable_regression:
            self.regression_head = RegressionHead(
                hidden_size=self.hidden_size,
                dropout=mlp_dropout,
            )
        if self.enable_qwen3_alignment:
            if self.qwen3_projection_dim is None:
                raise ValueError('qwen3_projection_dim must be provided when enable_qwen3_alignment=True')
            self.qwen3_alignment_head = Qwen3AlignmentHead(
                hidden_size=self.hidden_size,
                output_dim=self.qwen3_projection_dim,
                dropout=mlp_dropout,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        abundance_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bin_labels: Optional[torch.Tensor] = None,
        regression_targets: Optional[torch.Tensor] = None,
        regression_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=abundance_values,
            attention_mask=attention_mask,
        )

        hidden_states = backbone_outputs['last_hidden_state']
        cls_hidden = hidden_states[:, 0, :]
        outputs = {
            'hidden_states': hidden_states,
            'cls_hidden': cls_hidden,
            'bin_logits': self.bin_head(hidden_states),
        }

        if self.enable_regression:
            outputs['regression_preds'] = self.regression_head(hidden_states)
        if self.enable_qwen3_alignment:
            outputs['qwen3_proj'] = self.qwen3_alignment_head(cls_hidden)

        if any([bin_labels is not None, regression_targets is not None]):
            loss_dict = self.compute_losses(
                outputs=outputs,
                bin_labels=bin_labels,
                regression_targets=regression_targets,
                regression_mask=regression_mask,
            )
            outputs['loss_dict'] = loss_dict
            outputs['loss'] = sum(loss_dict.values())

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        bin_labels: Optional[torch.Tensor],
        regression_targets: Optional[torch.Tensor],
        regression_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}

        if bin_labels is not None and 'bin_logits' in outputs:
            bin_logits = outputs['bin_logits']
            valid_mask = bin_labels != -100
            if valid_mask.any():
                valid_logits = bin_logits[valid_mask].view(-1, bin_logits.size(-1))
                valid_labels = bin_labels[valid_mask].view(-1)
                loss_dict['bin_loss'] = torch.nn.functional.cross_entropy(valid_logits, valid_labels)
            else:
                loss_dict['bin_loss'] = torch.tensor(0.0, device=bin_logits.device)

        if self.enable_regression and regression_targets is not None and regression_mask is not None and 'regression_preds' in outputs:
            regression_preds = outputs['regression_preds']
            if regression_mask.any():
                valid_targets = regression_targets[regression_mask]
                valid_preds = regression_preds[regression_mask]
                loss_dict['regression_loss'] = torch.nn.functional.mse_loss(valid_preds, valid_targets) * 0.1
            else:
                loss_dict['regression_loss'] = torch.tensor(0.0, device=regression_preds.device)

        return loss_dict

    def get_task_weights(self, config: Dict[str, float]) -> Dict[str, float]:
        weights = {
            'bin_loss': config.get('lambda_bin', 1.0),
        }
        if self.enable_regression:
            weights['regression_loss'] = config.get('lambda_regression', 0.3)
        if self.enable_qwen3_alignment:
            weights['qwen3_alignment_loss'] = config.get('lambda_qwen3_alignment', 0.1)
        return weights
