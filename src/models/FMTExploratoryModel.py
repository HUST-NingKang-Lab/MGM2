"""Exploratory MGM2 FMT community prediction models."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.FMTModel import FMTCommunityPredictionModel


class FMTDeltaFusionPredictionModel(FMTCommunityPredictionModel):
    """Cross-attention FMT predictor with explicit donor-recipient delta fusion."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        cross_attention_heads: Optional[int] = None,
        dropout: float = 0.1,
        use_gated_delta: bool = True,
    ):
        super().__init__(
            model_config=model_config,
            cross_attention_heads=cross_attention_heads,
            dropout=dropout,
        )
        self.use_gated_delta = bool(use_gated_delta)
        self.delta_value_projector = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        fusion_in = self.hidden_size * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        if self.use_gated_delta:
            self.delta_gate = nn.Sequential(
                nn.Linear(fusion_in, self.hidden_size),
                nn.Sigmoid(),
            )
        else:
            self.delta_gate = None
        self.fusion_layer_norm = nn.LayerNorm(self.hidden_size)
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )

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

        attended, attention_weights = self.cross_attention(
            query=recipient_hidden,
            key=condition_tokens,
            value=condition_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        donor_recipient_hidden_delta = donor_hidden - recipient_hidden
        abundance_delta_token = self.delta_value_projector(
            (donor_abundance_values - recipient_abundance_values).unsqueeze(-1)
        )
        fusion_input = torch.cat(
            [
                recipient_hidden,
                attended,
                donor_recipient_hidden_delta,
                abundance_delta_token,
            ],
            dim=-1,
        )
        fused_delta = self.fusion(fusion_input)
        if self.delta_gate is not None:
            fused_delta = fused_delta * self.delta_gate(fusion_input)
        conditioned_hidden = self.fusion_layer_norm(recipient_hidden + self.dropout(fused_delta))
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
