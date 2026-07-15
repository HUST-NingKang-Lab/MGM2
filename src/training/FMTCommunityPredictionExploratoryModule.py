"""Lightning modules for exploratory MGM2 FMT community prediction."""

from typing import Dict

import torch
import torch.nn.functional as F

from src.configs.Config import MGM2Config
from src.models.FMTExploratoryModel import FMTDeltaFusionPredictionModel
from src.training.FMTCommunityPredictionModule import FMTCommunityPredictionLightningModule


class FMTCommunityPredictionExploratoryLightningModule(FMTCommunityPredictionLightningModule):
    """FMT predictor with explicit transformed-delta supervision."""

    def __init__(
        self,
        config: MGM2Config,
        cross_attention_heads: int | None = None,
        freeze_backbone: bool = True,
        learning_rate: float | None = None,
        backbone_learning_rate: float | None = None,
        raw_zero_threshold: float = 5e-5,
        renormalize_predictions: bool = True,
        delta_loss_weight: float = 0.5,
        use_gated_delta: bool = True,
    ):
        super().__init__(
            config=config,
            cross_attention_heads=cross_attention_heads,
            freeze_backbone=False,
            learning_rate=learning_rate,
            backbone_learning_rate=backbone_learning_rate,
            raw_zero_threshold=raw_zero_threshold,
            renormalize_predictions=renormalize_predictions,
        )
        self.delta_loss_weight = float(delta_loss_weight)
        self.model = FMTDeltaFusionPredictionModel(
            model_config=config.model.__dict__,
            cross_attention_heads=cross_attention_heads,
            dropout=config.model.mlp_dropout,
            use_gated_delta=use_gated_delta,
        )
        if freeze_backbone:
            self.model.freeze_backbone()

    def _masked_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["supervision_mask"].bool()
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        post_loss = F.mse_loss(outputs["pred_post_abundance"][mask], batch["post_abundance_values"][mask])
        true_delta = batch["post_abundance_values"] - batch["recipient_abundance_values"]
        delta_loss = F.mse_loss(outputs["pred_delta"][mask], true_delta[mask])
        total = post_loss + self.delta_loss_weight * delta_loss
        self.log("aux/post_loss", post_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("aux/delta_loss", delta_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total
