"""Lightning module for MGM2 FMT response classification."""

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.configs.Config import MGM2Config
from src.models.FMTModel import FMTCommunityClassificationModel


class FMTCommunityClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: MGM2Config,
        num_classes: int,
        class_names: list[str] | None = None,
        cross_attention_heads: int | None = None,
        freeze_backbone: bool = True,
        learning_rate: float | None = None,
        backbone_learning_rate: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.num_classes = int(num_classes)
        self.class_names = list(class_names or [])
        self.learning_rate = learning_rate or config.training.learning_rate
        self.backbone_learning_rate = backbone_learning_rate
        self.model = FMTCommunityClassificationModel(
            model_config=config.model.__dict__,
            num_classes=self.num_classes,
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
        if "recipient_input_ids" in batch:
            return self.model(
                input_ids=batch["recipient_input_ids"],
                embeddings=batch["recipient_embeddings"],
                recipient_abundance_values=batch["recipient_abundance_values"],
                donor_abundance_values=batch["donor_abundance_values"],
                attention_mask=batch["recipient_attention_mask"],
                donor_input_ids=batch["donor_input_ids"],
                donor_embeddings=batch["donor_embeddings"],
                donor_attention_mask=batch["donor_attention_mask"],
            )
        return self.model(
            input_ids=batch["input_ids"],
            embeddings=batch["embeddings"],
            recipient_abundance_values=batch["recipient_abundance_values"],
            donor_abundance_values=batch["donor_abundance_values"],
            attention_mask=batch["attention_mask"],
        )

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self(batch)
        labels = batch["labels"]
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=(stage != "train"), on_step=False, on_epoch=True, sync_dist=True)
        if self.num_classes > 1:
            recalls = []
            precisions = []
            for class_id in range(self.num_classes):
                pred_mask = preds == class_id
                label_mask = labels == class_id
                tp = (pred_mask & label_mask).sum().float()
                if label_mask.any():
                    recalls.append(tp / label_mask.sum().float())
                if pred_mask.any():
                    precisions.append(tp / pred_mask.sum().float())
            if recalls:
                self.log(f"{stage}/macro_recall", torch.stack(recalls).mean(), on_step=False, on_epoch=True, sync_dist=True)
            if precisions:
                self.log(f"{stage}/macro_precision", torch.stack(precisions).mean(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "target_id": batch["target_id"],
            "fmt_id": batch["fmt_id"],
            "study": batch["study"],
            "indication": batch["indication"],
            "response_label": batch["response_label"],
            "donor_sample_id": batch["donor_sample_id"],
            "recipient_sample_id": batch["recipient_sample_id"],
            "label_id": batch["labels"].detach().cpu(),
            "pred_id": probabilities.argmax(dim=-1).detach().cpu(),
            "probabilities": probabilities.detach().float().cpu(),
            "logits": logits.detach().float().cpu(),
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
