"""Lightning module for MGM2 FMT community prediction."""

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.configs.Config import MGM2Config
from src.models.FMTModel import FMTCommunityPredictionModel


class FMTCommunityPredictionLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: MGM2Config,
        cross_attention_heads: int | None = None,
        freeze_backbone: bool = True,
        learning_rate: float | None = None,
        backbone_learning_rate: float | None = None,
        raw_zero_threshold: float = 5e-5,
        renormalize_predictions: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.learning_rate = learning_rate or config.training.learning_rate
        self.backbone_learning_rate = backbone_learning_rate
        self.raw_zero_threshold = float(raw_zero_threshold)
        self.renormalize_predictions = bool(renormalize_predictions)
        self.model = FMTCommunityPredictionModel(
            model_config=config.model.__dict__,
            cross_attention_heads=cross_attention_heads,
            dropout=config.model.mlp_dropout,
        )
        if freeze_backbone:
            self.model.freeze_backbone()

    def load_pretrained_backbone_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        return self.model.load_pretrained_backbone(state_dict, strict=True)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            input_ids=batch["input_ids"],
            embeddings=batch["embeddings"],
            recipient_abundance_values=batch["recipient_abundance_values"],
            donor_abundance_values=batch["donor_abundance_values"],
            attention_mask=batch["attention_mask"],
            timepoint=batch["timepoint_fmt"],
        )

    @staticmethod
    def _transformed_to_raw(values: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(values, min=-2.0, max=2.0)
        return torch.pow(torch.tensor(10.0, device=values.device, dtype=values.dtype), clipped - 2.0)

    def _postprocess_pred_raw(self, values: torch.Tensor) -> torch.Tensor:
        if self.raw_zero_threshold > 0:
            values = torch.where(values <= self.raw_zero_threshold, torch.zeros_like(values), values)
        if self.renormalize_predictions:
            values = values / torch.clamp(values.sum(dim=1, keepdim=True), min=1e-12)
        return values

    @staticmethod
    def _bray_curtis(pred_raw: torch.Tensor, true_raw: torch.Tensor) -> torch.Tensor:
        denom = pred_raw.sum(dim=1) + true_raw.sum(dim=1)
        return torch.abs(pred_raw - true_raw).sum(dim=1) / torch.clamp(denom, min=1e-12)

    @staticmethod
    def _cosine(pred_raw: torch.Tensor, true_raw: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(pred_raw, true_raw, dim=1, eps=1e-12)

    def _masked_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["supervision_mask"].bool()
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        return F.mse_loss(outputs["pred_post_abundance"][mask], batch["post_abundance_values"][mask])

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self(batch)
        loss = self._masked_loss(outputs, batch)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)

        with torch.no_grad():
            pred_raw = self._postprocess_pred_raw(self._transformed_to_raw(outputs["pred_post_abundance"])[:, 1:])
            true_raw = batch["post_abundance_raw"][:, 1:]
            recipient_raw = batch["recipient_abundance_raw"][:, 1:]
            pred_delta_raw = pred_raw - recipient_raw
            true_delta_raw = true_raw - recipient_raw
            bray = self._bray_curtis(pred_raw, true_raw)
            recipient_bray = self._bray_curtis(recipient_raw, true_raw)
            cosine = self._cosine(pred_raw, true_raw)
            delta_cosine = self._cosine(pred_delta_raw, true_delta_raw)
            self.log(f"{stage}/raw_bray_curtis", bray.mean(), prog_bar=(stage != "train"), on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/recipient_baseline_bray_curtis", recipient_bray.mean(), on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/raw_cosine", cosine.mean(), on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/delta_cosine", delta_cosine.mean(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        pred_raw = self._postprocess_pred_raw(self._transformed_to_raw(outputs["pred_post_abundance"])[:, 1:])
        return {
            "target_id": batch["target_id"],
            "fmt_id": batch["fmt_id"],
            "study": batch["study"],
            "response_label": batch["response_label"],
            "donor_sample_id": batch["donor_sample_id"],
            "recipient_sample_id": batch["recipient_sample_id"],
            "post_sample_id": batch["post_sample_id"],
            "timepoint_fmt": batch["timepoint_fmt"].detach().float().cpu(),
            "pred_post_raw": pred_raw.detach().float().cpu(),
            "true_post_raw": batch["post_abundance_raw"][:, 1:].detach().float().cpu(),
            "recipient_raw": batch["recipient_abundance_raw"][:, 1:].detach().float().cpu(),
            "donor_raw": batch["donor_abundance_raw"][:, 1:].detach().float().cpu(),
        }

    def configure_optimizers(self):
        if self.backbone_learning_rate is None:
            return torch.optim.AdamW(
                [param for param in self.parameters() if param.requires_grad],
                lr=self.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )

        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.backbone_learning_rate},
                {"params": head_params, "lr": self.learning_rate},
            ],
            weight_decay=self.config.training.weight_decay,
        )
