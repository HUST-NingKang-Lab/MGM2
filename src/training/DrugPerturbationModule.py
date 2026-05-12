"""Lightning module for drug-conditioned MGM2 perturbation prediction."""

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.configs.Config import MGM2Config
from src.models.DrugPerturbationModel import DrugPerturbationCrossAttentionModel


class DrugPerturbationLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: MGM2Config,
        drug_embedding_dim: int = 512,
        num_donors: int = 9,
        num_drug_tokens: int = 4,
        cross_attention_heads: int | None = None,
        freeze_backbone: bool = True,
        learning_rate: float | None = None,
        backbone_learning_rate: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.learning_rate = learning_rate or config.training.learning_rate
        self.backbone_learning_rate = backbone_learning_rate
        self.model = DrugPerturbationCrossAttentionModel(
            model_config=config.model.__dict__,
            drug_embedding_dim=drug_embedding_dim,
            num_donors=num_donors,
            num_drug_tokens=num_drug_tokens,
            cross_attention_heads=cross_attention_heads,
            dropout=config.model.mlp_dropout,
        )
        if freeze_backbone:
            self.model.freeze_backbone()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            input_ids=batch["input_ids"],
            embeddings=batch["embeddings"],
            before_abundance_values=batch["before_abundance_values"],
            attention_mask=batch["attention_mask"],
            drug_embedding=batch["drug_embedding"],
            donor_id=batch["donor_id"],
        )

    @staticmethod
    def _transformed_to_raw(values: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(values, min=-2.0, max=2.0)
        return torch.pow(torch.tensor(10.0, device=values.device, dtype=values.dtype), clipped - 2.0)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self(batch)
        mask = batch["supervision_mask"].bool()
        target_delta = batch["after_abundance_values"] - batch["before_abundance_values"]
        if mask.any():
            delta_loss = F.mse_loss(outputs["pred_delta"][mask], target_delta[mask])
            after_loss = F.mse_loss(outputs["pred_after_abundance"][mask], batch["after_abundance_values"][mask])
        else:
            delta_loss = torch.tensor(0.0, device=self.device)
            after_loss = torch.tensor(0.0, device=self.device)
        loss = delta_loss
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/delta_mse", delta_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/after_mse", after_loss, on_step=False, on_epoch=True)

        with torch.no_grad():
            pred_raw = self._transformed_to_raw(outputs["pred_after_abundance"])
            true_raw = batch["after_abundance_raw"]
            raw_mask = batch["supervision_mask"].bool()
            pred_raw = pred_raw * raw_mask.to(dtype=pred_raw.dtype)
            true_raw = true_raw * raw_mask.to(dtype=true_raw.dtype)
            denom = pred_raw.sum(dim=1) + true_raw.sum(dim=1)
            bray = torch.abs(pred_raw - true_raw).sum(dim=1) / torch.clamp(denom, min=1e-12)
            valid_rows = raw_mask.sum(dim=1) > 0
            if valid_rows.any():
                self.log(f"{stage}/raw_bray_curtis", bray[valid_rows].mean(), prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

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
