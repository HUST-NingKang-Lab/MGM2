"""MGM2 models for FMT donor-recipient community prediction."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.BaseModel import BaseModel


class FMTCommunityPredictionModel(nn.Module):
    """Encode recipient and donor communities, then predict the post-FMT community."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        cross_attention_heads: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = int(model_config.get("hidden_size", 512))
        self.backbone = BaseModel(
            vocab_size=model_config.get("vocab_size", 10000),
            embedding_dim=model_config.get("embedding_dim", 1536),
            hidden_size=self.hidden_size,
            num_hidden_layers=model_config.get("num_hidden_layers", 12),
            num_attention_heads=model_config.get("num_attention_heads", 8),
            intermediate_size=model_config.get("intermediate_size", 2048),
            hidden_dropout_prob=model_config.get("hidden_dropout_prob", dropout),
            attention_probs_dropout_prob=model_config.get("attention_probs_dropout_prob", dropout),
            layer_norm_eps=model_config.get("layer_norm_eps", 1e-12),
            max_position_embeddings=model_config.get("max_position_embeddings", 2048),
            pad_token_id=model_config.get("pad_token_id", -1),
            cls_token_id=model_config.get("cls_token_id", -1),
        )
        heads = int(cross_attention_heads or model_config.get("num_attention_heads", 8))
        self.timepoint_projector = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.condition_layer_norm = nn.LayerNorm(self.hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def load_pretrained_backbone(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith("model.backbone."):
                backbone_state[key[len("model.backbone.") :]] = value
            elif key.startswith("backbone."):
                backbone_state[key[len("backbone.") :]] = value
        if not backbone_state:
            raise ValueError("No MGM2 backbone weights were found in the checkpoint state_dict")
        return self.backbone.load_state_dict(backbone_state, strict=strict)

    @staticmethod
    def _normalize_timepoint(timepoint: torch.Tensor) -> torch.Tensor:
        timepoint = torch.nan_to_num(timepoint.float(), nan=0.0, posinf=365.0, neginf=0.0)
        timepoint = torch.clamp(timepoint, min=0.0)
        return torch.log1p(timepoint).unsqueeze(-1) / torch.log(torch.tensor(366.0, device=timepoint.device))

    def forward(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        recipient_abundance_values: torch.Tensor,
        donor_abundance_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        timepoint: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        recipient_outputs = self.backbone(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=recipient_abundance_values,
            attention_mask=attention_mask,
        )
        donor_outputs = self.backbone(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=donor_abundance_values,
            attention_mask=attention_mask,
        )
        recipient_hidden = recipient_outputs["last_hidden_state"]
        donor_hidden = donor_outputs["last_hidden_state"]

        time_token = self.timepoint_projector(self._normalize_timepoint(timepoint)).unsqueeze(1)
        condition_tokens = self.condition_layer_norm(torch.cat([donor_hidden, time_token], dim=1))
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = torch.cat(
                [
                    attention_mask == 0,
                    torch.zeros((attention_mask.shape[0], 1), dtype=torch.bool, device=attention_mask.device),
                ],
                dim=1,
            )

        conditioned, attention_weights = self.cross_attention(
            query=recipient_hidden,
            key=condition_tokens,
            value=condition_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        conditioned_hidden = self.output_layer_norm(recipient_hidden + self.dropout(conditioned))
        pred_delta = self.delta_head(conditioned_hidden).squeeze(-1)
        pred_post = recipient_abundance_values + pred_delta
        return {
            "recipient_hidden_states": recipient_hidden,
            "donor_hidden_states": donor_hidden,
            "conditioned_hidden_states": conditioned_hidden,
            "cross_attention_weights": attention_weights,
            "pred_delta": pred_delta,
            "pred_post_abundance": pred_post,
        }


class FMTCommunityClassificationModel(nn.Module):
    """Encode recipient and donor communities, then classify FMT response."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        num_classes: int,
        cross_attention_heads: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.hidden_size = int(model_config.get("hidden_size", 512))
        self.num_classes = int(num_classes)
        self.backbone = BaseModel(
            vocab_size=model_config.get("vocab_size", 10000),
            embedding_dim=model_config.get("embedding_dim", 1536),
            hidden_size=self.hidden_size,
            num_hidden_layers=model_config.get("num_hidden_layers", 12),
            num_attention_heads=model_config.get("num_attention_heads", 8),
            intermediate_size=model_config.get("intermediate_size", 2048),
            hidden_dropout_prob=model_config.get("hidden_dropout_prob", dropout),
            attention_probs_dropout_prob=model_config.get("attention_probs_dropout_prob", dropout),
            layer_norm_eps=model_config.get("layer_norm_eps", 1e-12),
            max_position_embeddings=model_config.get("max_position_embeddings", 2048),
            pad_token_id=model_config.get("pad_token_id", -1),
            cls_token_id=model_config.get("cls_token_id", -1),
        )
        heads = int(cross_attention_heads or model_config.get("num_attention_heads", 8))
        self.condition_layer_norm = nn.LayerNorm(self.hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def load_pretrained_backbone(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith("model.backbone."):
                backbone_state[key[len("model.backbone.") :]] = value
            elif key.startswith("backbone."):
                backbone_state[key[len("backbone.") :]] = value
        if not backbone_state:
            raise ValueError("No MGM2 backbone weights were found in the checkpoint state_dict")
        return self.backbone.load_state_dict(backbone_state, strict=strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        recipient_abundance_values: torch.Tensor,
        donor_abundance_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        recipient_outputs = self.backbone(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=recipient_abundance_values,
            attention_mask=attention_mask,
        )
        donor_outputs = self.backbone(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=donor_abundance_values,
            attention_mask=attention_mask,
        )
        recipient_hidden = recipient_outputs["last_hidden_state"]
        donor_hidden = donor_outputs["last_hidden_state"]
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        conditioned, attention_weights = self.cross_attention(
            query=recipient_hidden,
            key=self.condition_layer_norm(donor_hidden),
            value=self.condition_layer_norm(donor_hidden),
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        conditioned_hidden = self.output_layer_norm(recipient_hidden + self.dropout(conditioned))
        recipient_cls = recipient_hidden[:, 0, :]
        donor_cls = donor_hidden[:, 0, :]
        conditioned_cls = conditioned_hidden[:, 0, :]
        features = torch.cat(
            [
                recipient_cls,
                donor_cls,
                conditioned_cls,
                torch.abs(recipient_cls - donor_cls),
            ],
            dim=-1,
        )
        logits = self.classifier(features)
        return {
            "recipient_hidden_states": recipient_hidden,
            "donor_hidden_states": donor_hidden,
            "conditioned_hidden_states": conditioned_hidden,
            "cross_attention_weights": attention_weights,
            "logits": logits,
            "classification_logits": logits,
        }
