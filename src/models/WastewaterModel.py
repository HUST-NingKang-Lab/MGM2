"""Temporal forecasting heads for wastewater MGM2 embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ABUNDANCE_HEADS = {"additive_residual", "logratio_residual", "gated_softmax", "softmax", "paper_relu"}
TOKEN_QUERY_MODES = {"last", "temporal_attention"}


def normalize_abundance_torch(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize nonnegative abundance predictions across taxa."""
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.clamp(values, min=0.0)
    totals = values.sum(dim=-1, keepdim=True)
    return torch.where(totals > eps, values / totals.clamp_min(eps), torch.zeros_like(values))


def bray_curtis_loss_torch(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable Bray-Curtis dissimilarity averaged over batch and horizon."""
    pred = normalize_abundance_torch(y_pred, eps=eps)
    true = normalize_abundance_torch(y_true, eps=eps)
    return bray_curtis_raw_loss_torch(pred, true, eps=eps)


def bray_curtis_raw_loss_torch(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Bray-Curtis dissimilarity without closure normalization, matching the paper implementation."""
    pred = torch.nan_to_num(y_pred.float(), nan=0.0, posinf=0.0, neginf=0.0)
    true = torch.nan_to_num(y_true.float(), nan=0.0, posinf=0.0, neginf=0.0)
    shared = torch.minimum(pred, true).sum(dim=-1)
    denom = pred.sum(dim=-1) + true.sum(dim=-1)
    loss = 1.0 - (2.0 * shared / (denom + eps))
    return loss.mean()


def abundance_from_context_residual(
    raw: torch.Tensor,
    context_abundance: torch.Tensor,
    residual_persistence: bool,
    residual_scale: float,
    abundance_head: str = "additive_residual",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map raw future scores to compositional abundance predictions."""
    if abundance_head not in ABUNDANCE_HEADS:
        raise ValueError(f"abundance_head must be one of {sorted(ABUNDANCE_HEADS)}, got {abundance_head!r}")
    if abundance_head == "paper_relu":
        return F.relu(raw)
    if not residual_persistence or abundance_head == "softmax":
        return F.softmax(raw, dim=-1)

    context_abundance = normalize_abundance_torch(context_abundance, eps=eps)
    base = context_abundance[:, -1:, :].repeat(1, raw.shape[1], 1)
    residual_scale = float(residual_scale)

    if abundance_head == "additive_residual":
        return normalize_abundance_torch(base + residual_scale * torch.tanh(raw), eps=eps)
    if abundance_head == "logratio_residual":
        return F.softmax(torch.log(base.clamp_min(eps)) + residual_scale * raw, dim=-1)

    gate = torch.sigmoid(residual_scale * raw.mean(dim=-1, keepdim=True))
    predicted = (1.0 - gate) * base + gate * F.softmax(raw, dim=-1)
    return normalize_abundance_torch(predicted, eps=eps)


class PlantTemporalForecaster(nn.Module):
    """A small GRU head that maps context CLS states to future abundance."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        horizon: int,
        num_asvs: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if num_asvs <= 0:
            raise ValueError("num_asvs must be positive")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.num_asvs = int(num_asvs)
        self.num_layers = int(num_layers)

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * num_asvs),
        )

    def forward(self, context_cls: torch.Tensor) -> torch.Tensor:
        if context_cls.ndim != 3:
            raise ValueError("context_cls must have shape [batch, context_len, input_size]")
        _, hidden = self.encoder(context_cls.float())
        final_hidden = hidden[-1]
        logits = self.head(final_hidden).view(-1, self.horizon, self.num_asvs)
        return F.softmax(logits, dim=-1)


class VectorTemporalForecaster(nn.Module):
    """Forecast future abundance directly from past abundance vectors."""

    def __init__(
        self,
        num_asvs: int,
        horizon: int,
        hidden_size: int = 256,
        model_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.1,
        residual_persistence: bool = True,
        residual_scale: float = 0.1,
        abundance_head: str = "additive_residual",
    ):
        super().__init__()
        if num_asvs <= 0:
            raise ValueError("num_asvs must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if model_type not in {"gru", "tcn"}:
            raise ValueError("model_type must be 'gru' or 'tcn'")
        self.num_asvs = int(num_asvs)
        self.horizon = int(horizon)
        self.model_type = model_type
        self.residual_persistence = bool(residual_persistence)
        self.residual_scale = float(residual_scale)
        if abundance_head not in ABUNDANCE_HEADS:
            raise ValueError(f"abundance_head must be one of {sorted(ABUNDANCE_HEADS)}, got {abundance_head!r}")
        self.abundance_head = abundance_head

        if model_type == "gru":
            self.encoder = nn.GRU(
                input_size=num_asvs,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            head_input = hidden_size
        else:
            layers = []
            in_channels = num_asvs
            for layer_idx in range(num_layers):
                dilation = 2 ** layer_idx
                layers.extend(
                    [
                        nn.Conv1d(
                            in_channels,
                            hidden_size,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_channels = hidden_size
            self.encoder = nn.Sequential(*layers)
            head_input = hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Dropout(dropout),
            nn.Linear(head_input, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * num_asvs),
        )

    def _encode(self, context_abundance: torch.Tensor) -> torch.Tensor:
        if self.model_type == "gru":
            _, hidden = self.encoder(context_abundance.float())
            return hidden[-1]
        encoded = self.encoder(context_abundance.float().transpose(1, 2))
        return encoded[:, :, -1]

    def forward(self, context_abundance: torch.Tensor) -> torch.Tensor:
        if context_abundance.ndim != 3:
            raise ValueError("context_abundance must have shape [batch, context_len, num_asvs]")
        context_abundance = normalize_abundance_torch(context_abundance)
        features = self._encode(context_abundance)
        raw = self.head(features).view(-1, self.horizon, self.num_asvs)

        return abundance_from_context_residual(
            raw,
            context_abundance,
            residual_persistence=self.residual_persistence,
            residual_scale=self.residual_scale,
            abundance_head=self.abundance_head,
        )


class MGM2TokenCrossAttentionForecaster(nn.Module):
    """Predict future abundance from frozen MGM2 token hidden states."""

    def __init__(
        self,
        hidden_size: int,
        horizon: int,
        num_asvs: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_persistence: bool = True,
        residual_scale: float = 0.1,
        abundance_head: str = "additive_residual",
        query_mode: str = "last",
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

        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.num_asvs = int(num_asvs)
        self.residual_persistence = bool(residual_persistence)
        self.residual_scale = float(residual_scale)
        if abundance_head not in ABUNDANCE_HEADS:
            raise ValueError(f"abundance_head must be one of {sorted(ABUNDANCE_HEADS)}, got {abundance_head!r}")
        if query_mode not in TOKEN_QUERY_MODES:
            raise ValueError(f"query_mode must be one of {sorted(TOKEN_QUERY_MODES)}, got {query_mode!r}")
        if max_context_len <= 0:
            raise ValueError("max_context_len must be positive")
        self.abundance_head = abundance_head
        self.query_mode = query_mode
        self.max_context_len = int(max_context_len)

        self.horizon_embedding = nn.Embedding(horizon, hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        if query_mode == "temporal_attention":
            self.temporal_position_embedding = nn.Parameter(torch.zeros(1, self.max_context_len, hidden_size))
            self.temporal_query = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.temporal_norm = nn.LayerNorm(hidden_size)
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _build_future_queries(self, context_token_states: torch.Tensor) -> torch.Tensor:
        batch_size, context_len, num_asvs, hidden_size = context_token_states.shape
        if self.query_mode == "last":
            query_base = context_token_states[:, -1, :, :]
        else:
            if context_len > self.max_context_len:
                raise ValueError(f"context length {context_len} exceeds max_context_len={self.max_context_len}")
            temporal_tokens = context_token_states.permute(0, 2, 1, 3).reshape(batch_size * num_asvs, context_len, hidden_size)
            temporal_tokens = self.temporal_norm(temporal_tokens + self.temporal_position_embedding[:, :context_len, :])
            temporal_query = self.temporal_query.expand(batch_size * num_asvs, -1, -1)
            query_base, _ = self.temporal_attention(temporal_query, temporal_tokens, temporal_tokens, need_weights=False)
            query_base = query_base.view(batch_size, num_asvs, hidden_size)

        horizon_ids = torch.arange(self.horizon, device=context_token_states.device)
        horizon_emb = self.horizon_embedding(horizon_ids).view(1, self.horizon, 1, hidden_size)
        queries = self.query_norm(query_base[:, None, :, :] + horizon_emb)
        return queries.reshape(batch_size, self.horizon * num_asvs, hidden_size)

    def forward(self, context_token_states: torch.Tensor, context_abundance: torch.Tensor) -> torch.Tensor:
        if context_token_states.ndim != 4:
            raise ValueError("context_token_states must have shape [batch, context_len, num_asvs, hidden_size]")
        if context_abundance.ndim != 3:
            raise ValueError("context_abundance must have shape [batch, context_len, num_asvs]")
        batch_size, context_len, num_asvs, hidden_size = context_token_states.shape
        if num_asvs != self.num_asvs:
            raise ValueError(f"expected {self.num_asvs} ASVs, got {num_asvs}")
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")

        context_token_states = context_token_states.float()
        context_abundance = normalize_abundance_torch(context_abundance)
        memory = self.context_norm(context_token_states.reshape(batch_size, context_len * num_asvs, hidden_size))

        queries = self._build_future_queries(context_token_states)
        attended, _ = self.cross_attention(queries, memory, memory, need_weights=False)
        raw = self.head(attended).view(batch_size, self.horizon, num_asvs)
        return abundance_from_context_residual(
            raw,
            context_abundance,
            residual_persistence=self.residual_persistence,
            residual_scale=self.residual_scale,
            abundance_head=self.abundance_head,
        )
