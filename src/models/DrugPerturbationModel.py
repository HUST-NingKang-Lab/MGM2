"""Drug-conditioned MGM2 perturbation prediction models."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.BaseModel import BaseModel


class DrugPerturbationCrossAttentionModel(nn.Module):
    """MGM2 backbone with drug/donor cross-attention conditioning."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        drug_embedding_dim: int = 512,
        num_donors: int = 9,
        num_drug_tokens: int = 4,
        cross_attention_heads: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = int(model_config.get("hidden_size", 512))
        self.num_drug_tokens = int(num_drug_tokens)
        if self.num_drug_tokens <= 0:
            raise ValueError("num_drug_tokens must be positive")

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

        self.drug_projector = nn.Sequential(
            nn.Linear(drug_embedding_dim, self.hidden_size * self.num_drug_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.donor_embedding = nn.Embedding(num_donors, self.hidden_size)
        heads = int(cross_attention_heads or model_config.get("num_attention_heads", 8))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.condition_layer_norm = nn.LayerNorm(self.hidden_size)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        before_abundance_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        drug_embedding: torch.Tensor,
        donor_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=before_abundance_values,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_outputs["last_hidden_state"]
        batch_size = hidden_states.shape[0]

        drug_tokens = self.drug_projector(drug_embedding).view(
            batch_size,
            self.num_drug_tokens,
            self.hidden_size,
        )
        donor_token = self.donor_embedding(donor_id).unsqueeze(1)
        condition_tokens = self.condition_layer_norm(torch.cat([drug_tokens, donor_token], dim=1))

        conditioned, attention_weights = self.cross_attention(
            query=hidden_states,
            key=condition_tokens,
            value=condition_tokens,
            need_weights=True,
        )
        conditioned_hidden = self.output_layer_norm(hidden_states + self.dropout(conditioned))
        pred_delta = self.delta_head(conditioned_hidden).squeeze(-1)
        pred_after = before_abundance_values + pred_delta
        return {
            "hidden_states": hidden_states,
            "conditioned_hidden_states": conditioned_hidden,
            "cross_attention_weights": attention_weights,
            "pred_delta": pred_delta,
            "pred_after_abundance": pred_after,
        }
