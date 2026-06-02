"""MAE and JEPA-style wastewater world models built on cached MGM2 token states."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.WastewaterModel import (
    ABUNDANCE_HEADS,
    TOKEN_QUERY_MODES,
    abundance_from_context_residual,
    bray_curtis_loss_torch,
    bray_curtis_raw_loss_torch,
    normalize_abundance_torch,
)

BC_LOSS_MODES = {"normalized", "raw"}


def _bray_loss(outputs: torch.Tensor, target: torch.Tensor, bc_loss_mode: str) -> torch.Tensor:
    if bc_loss_mode == "normalized":
        return bray_curtis_loss_torch(outputs, target)
    if bc_loss_mode == "raw":
        return bray_curtis_raw_loss_torch(outputs, target)
    raise ValueError(f"bc_loss_mode must be one of {sorted(BC_LOSS_MODES)}, got {bc_loss_mode!r}")


class _TokenStateEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, token_states: torch.Tensor) -> torch.Tensor:
        token_states = token_states.float()
        return self.output_norm(token_states + self.block(token_states))


class _FutureTokenPredictor(nn.Module):
    """Predict future ASV token states by attending future queries over context token memory."""

    def __init__(
        self,
        hidden_size: int,
        horizon: int,
        num_asvs: int,
        num_heads: int,
        dropout: float,
        query_mode: str = "temporal_attention",
        max_context_len: int = 64,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if num_asvs <= 0:
            raise ValueError("num_asvs must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if query_mode not in TOKEN_QUERY_MODES:
            raise ValueError(f"query_mode must be one of {sorted(TOKEN_QUERY_MODES)}, got {query_mode!r}")
        if max_context_len <= 0:
            raise ValueError("max_context_len must be positive")
        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.num_asvs = int(num_asvs)
        self.query_mode = query_mode
        self.max_context_len = int(max_context_len)
        self.horizon_embedding = nn.Embedding(horizon, hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        if query_mode == "temporal_attention":
            self.temporal_position_embedding = nn.Parameter(torch.zeros(1, self.max_context_len, hidden_size))
            self.temporal_query = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.temporal_norm = nn.LayerNorm(hidden_size)
            self.temporal_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def _build_future_queries(self, context_states: torch.Tensor) -> torch.Tensor:
        batch_size, context_len, num_asvs, hidden_size = context_states.shape
        if self.query_mode == "last":
            query_base = context_states[:, -1, :, :]
        else:
            if context_len > self.max_context_len:
                raise ValueError(f"context length {context_len} exceeds max_context_len={self.max_context_len}")
            temporal_tokens = context_states.permute(0, 2, 1, 3).reshape(batch_size * num_asvs, context_len, hidden_size)
            temporal_tokens = self.temporal_norm(temporal_tokens + self.temporal_position_embedding[:, :context_len, :])
            temporal_query = self.temporal_query.expand(batch_size * num_asvs, -1, -1)
            query_base, _ = self.temporal_attention(temporal_query, temporal_tokens, temporal_tokens, need_weights=False)
            query_base = query_base.view(batch_size, num_asvs, hidden_size)

        horizon_ids = torch.arange(self.horizon, device=context_states.device)
        horizon_emb = self.horizon_embedding(horizon_ids).view(1, self.horizon, 1, hidden_size)
        queries = self.query_norm(query_base[:, None, :, :] + horizon_emb)
        return queries.reshape(batch_size, self.horizon * num_asvs, hidden_size)

    def forward(self, context_states: torch.Tensor) -> torch.Tensor:
        if context_states.ndim != 4:
            raise ValueError("context_states must have shape [batch, context_len, num_asvs, hidden_size]")
        batch_size, context_len, num_asvs, hidden_size = context_states.shape
        if num_asvs != self.num_asvs:
            raise ValueError(f"expected {self.num_asvs} ASVs, got {num_asvs}")
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")

        memory = self.context_norm(context_states.reshape(batch_size, context_len * num_asvs, hidden_size))
        queries = self._build_future_queries(context_states)
        attended_delta, _ = self.cross_attention(queries, memory, memory, need_weights=False)
        attended = queries + attended_delta
        attended = attended + self.ffn(attended)
        return attended.view(batch_size, self.horizon, num_asvs, hidden_size)


class _AbundanceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        residual_persistence: bool,
        residual_scale: float,
        abundance_head: str = "additive_residual",
    ):
        super().__init__()
        self.residual_persistence = bool(residual_persistence)
        self.residual_scale = float(residual_scale)
        if abundance_head not in ABUNDANCE_HEADS:
            raise ValueError(f"abundance_head must be one of {sorted(ABUNDANCE_HEADS)}, got {abundance_head!r}")
        self.abundance_head = abundance_head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, future_states: torch.Tensor, context_abundance: torch.Tensor) -> torch.Tensor:
        raw = self.head(future_states).squeeze(-1)
        return abundance_from_context_residual(
            raw,
            context_abundance,
            residual_persistence=self.residual_persistence,
            residual_scale=self.residual_scale,
            abundance_head=self.abundance_head,
        )


class _MGM2WorldModelBase(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        horizon: int,
        num_asvs: int,
        num_heads: int,
        dropout: float,
        residual_persistence: bool,
        residual_scale: float,
        abundance_head: str = "logratio_residual",
        query_mode: str = "temporal_attention",
        max_context_len: int = 64,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.num_asvs = int(num_asvs)
        self.online_encoder = _TokenStateEncoder(hidden_size, dropout)
        self.predictor = _FutureTokenPredictor(hidden_size, horizon, num_asvs, num_heads, dropout, query_mode, max_context_len)
        self.abundance_head = _AbundanceHead(hidden_size, dropout, residual_persistence, residual_scale, abundance_head)

    def predict_latents(self, context_states: torch.Tensor) -> torch.Tensor:
        context_latents = self.online_encoder(context_states)
        return self.predictor(context_latents)

    def decode_abundance(self, predicted_states: torch.Tensor, context_abundance: torch.Tensor) -> torch.Tensor:
        return self.abundance_head(predicted_states, context_abundance)

    def freeze_world_model(self):
        for module in (self.online_encoder, self.predictor):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def unfreeze_world_model(self):
        for module in (self.online_encoder, self.predictor):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


class MGM2MAEWorldModel(_MGM2WorldModelBase):
    """Future abundance reconstruction model trained directly in abundance space."""

    def __init__(
        self,
        hidden_size: int,
        horizon: int,
        num_asvs: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_persistence: bool = True,
        residual_scale: float = 1.0,
        abundance_head: str = "logratio_residual",
        query_mode: str = "temporal_attention",
        max_context_len: int = 64,
    ):
        super().__init__(
            hidden_size,
            horizon,
            num_asvs,
            num_heads,
            dropout,
            residual_persistence,
            residual_scale,
            abundance_head,
            query_mode,
            max_context_len,
        )

    def forward(self, context_states: torch.Tensor, context_abundance: torch.Tensor) -> Dict[str, torch.Tensor]:
        predicted_states = self.predict_latents(context_states)
        abundance = self.decode_abundance(predicted_states, context_abundance)
        return {"abundance": abundance, "predicted_states": predicted_states}

    def training_loss(
        self,
        context_states: torch.Tensor,
        context_abundance: torch.Tensor,
        target_abundance: torch.Tensor,
        loss_mse_weight: float = 0.1,
        bc_loss_mode: str = "normalized",
    ) -> Dict[str, torch.Tensor]:
        outputs = self(context_states, context_abundance)
        bray_loss = _bray_loss(outputs["abundance"], target_abundance, bc_loss_mode)
        mse_loss = F.mse_loss(outputs["abundance"], target_abundance)
        loss = bray_loss + loss_mse_weight * mse_loss
        return {"loss": loss, "bray_loss": bray_loss.detach(), "mse_loss": mse_loss.detach()}


class MGM2DeltaMAEWorldModel(MGM2MAEWorldModel):
    """MAE variant that explicitly supervises future abundance changes."""

    def forward(self, context_states: torch.Tensor, context_abundance: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = super().forward(context_states, context_abundance)
        base = normalize_abundance_torch(context_abundance)[:, -1:, :].repeat(1, outputs["abundance"].shape[1], 1)
        outputs["base_abundance"] = base
        outputs["delta"] = outputs["abundance"] - base
        return outputs

    def training_loss(
        self,
        context_states: torch.Tensor,
        context_abundance: torch.Tensor,
        target_abundance: torch.Tensor,
        loss_mse_weight: float = 0.1,
        delta_loss_weight: float = 0.5,
        bc_loss_mode: str = "normalized",
    ) -> Dict[str, torch.Tensor]:
        outputs = self(context_states, context_abundance)
        target_abundance = normalize_abundance_torch(target_abundance)
        target_delta = target_abundance - outputs["base_abundance"]
        bray_loss = _bray_loss(outputs["abundance"], target_abundance, bc_loss_mode)
        mse_loss = F.mse_loss(outputs["abundance"], target_abundance)
        delta_loss = F.mse_loss(outputs["delta"], target_delta)
        loss = bray_loss + loss_mse_weight * mse_loss + delta_loss_weight * delta_loss
        return {
            "loss": loss,
            "bray_loss": bray_loss.detach(),
            "mse_loss": mse_loss.detach(),
            "delta_loss": delta_loss.detach(),
        }


class MGM2JEPAWorldModel(_MGM2WorldModelBase):
    """Latent-only JEPA model with an EMA target encoder and a separate abundance probe."""

    def __init__(
        self,
        hidden_size: int,
        horizon: int,
        num_asvs: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_persistence: bool = True,
        residual_scale: float = 1.0,
        abundance_head: str = "logratio_residual",
        query_mode: str = "temporal_attention",
        max_context_len: int = 64,
    ):
        super().__init__(
            hidden_size,
            horizon,
            num_asvs,
            num_heads,
            dropout,
            residual_persistence,
            residual_scale,
            abundance_head,
            query_mode,
            max_context_len,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        context_states: torch.Tensor,
        context_abundance: torch.Tensor | None = None,
        return_abundance: bool = False,
    ) -> Dict[str, torch.Tensor]:
        predicted_states = self.predict_latents(context_states)
        outputs = {"predicted_states": predicted_states}
        if return_abundance:
            if context_abundance is None:
                raise ValueError("context_abundance is required when return_abundance=True")
            outputs["abundance"] = self.decode_abundance(predicted_states, context_abundance)
        return outputs

    @torch.no_grad()
    def encode_target(self, target_states: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(target_states)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        momentum = float(momentum)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1")
        for online_parameter, target_parameter in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_parameter.data.mul_(momentum).add_(online_parameter.data, alpha=1.0 - momentum)
        for online_buffer, target_buffer in zip(self.online_encoder.buffers(), self.target_encoder.buffers()):
            target_buffer.copy_(online_buffer)

    def training_loss(self, context_states: torch.Tensor, target_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        predicted = F.normalize(self(context_states)["predicted_states"], dim=-1)
        target = F.normalize(self.encode_target(target_states).detach(), dim=-1)
        loss = F.smooth_l1_loss(predicted, target)
        return {"loss": loss, "latent_loss": loss.detach()}

    def probe_loss(
        self,
        context_states: torch.Tensor,
        context_abundance: torch.Tensor,
        target_abundance: torch.Tensor,
        loss_mse_weight: float = 0.1,
        bc_loss_mode: str = "normalized",
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            predicted_states = self(context_states)["predicted_states"]
        abundance = self.decode_abundance(predicted_states.detach(), context_abundance)
        bray_loss = _bray_loss(abundance, target_abundance, bc_loss_mode)
        mse_loss = F.mse_loss(abundance, target_abundance)
        loss = bray_loss + loss_mse_weight * mse_loss
        return {"loss": loss, "bray_loss": bray_loss.detach(), "mse_loss": mse_loss.detach()}
