"""BatchTopK sparse autoencoder for MGM2 activation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BatchTopKSAEOutput:
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    num_active: int
    mse_loss: torch.Tensor
    aux_loss: torch.Tensor
    loss: torch.Tensor


class BatchTopKSAE(nn.Module):
    """Sparse autoencoder with batch-level TopK gating.

    This follows the core BatchTopK idea from bartbussmann/BatchTopK: select
    `top_k * batch_size` activations globally across the batch, rather than
    enforcing exactly `top_k` active features for every sample.
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        top_k: int,
        aux_k: int | None = None,
        aux_penalty: float = 0.0,
        input_unit_norm: bool = True,
        dead_feature_threshold: int = 10_000_000,
        inactive_counter_mode: str = "tokens",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if dict_size <= 0:
            raise ValueError("dict_size must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.input_dim = int(input_dim)
        self.dict_size = int(dict_size)
        self.top_k = int(top_k)
        self.aux_k = int(aux_k if aux_k is not None else max(1, top_k // 2))
        self.aux_penalty = float(aux_penalty)
        self.input_unit_norm = bool(input_unit_norm)
        if inactive_counter_mode not in {"tokens", "forwards"}:
            raise ValueError("inactive_counter_mode must be 'tokens' or 'forwards'")
        self.dead_feature_threshold = int(dead_feature_threshold)
        self.inactive_counter_mode = inactive_counter_mode

        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.W_enc = nn.Parameter(torch.empty(input_dim, dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, input_dim))
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)
        self.normalize_decoder_weights()
        self.register_buffer("num_tokens_since_fired", torch.zeros(dict_size, dtype=torch.long))

    def _inactive_increment(self, batch_size: int) -> int:
        return 1 if self.inactive_counter_mode == "forwards" else int(batch_size)

    def dead_feature_mask(self) -> torch.Tensor:
        return self.num_tokens_since_fired > self.dead_feature_threshold

    def num_dead_features(self) -> torch.Tensor:
        return self.dead_feature_mask().sum()

    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        self.normalize_decoder_weights()
        if self.W_dec.grad is None:
            return
        parallel_component = (self.W_dec.grad * self.W_dec.data).sum(dim=-1, keepdim=True) * self.W_dec.data
        self.W_dec.grad.sub_(parallel_component)

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.input_unit_norm:
            return x, {}
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        x_scale = x_centered.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x_normalized = x_centered / x_scale * (self.input_dim**0.5)
        return x_normalized, {"x_mean": x_mean, "x_scale": x_scale}

    def postprocess(self, x: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.input_unit_norm:
            return x
        return x / (self.input_dim**0.5) * info["x_scale"] + info["x_mean"]

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc)

    def _batch_topk(self, acts: torch.Tensor, k: int) -> torch.Tensor:
        flat = acts.flatten()
        total_k = min(max(1, int(k) * acts.shape[0]), flat.numel())
        top_values, top_indices = flat.topk(total_k, sorted=False)
        sparse_flat = torch.zeros_like(flat)
        sparse_flat.scatter_(0, top_indices, top_values)
        return sparse_flat.view_as(acts)

    def encode(self, x: torch.Tensor, update_inactive: bool = True) -> torch.Tensor:
        dense_acts = self.encode_dense(x)
        feature_acts = self._batch_topk(dense_acts, self.top_k)
        if update_inactive and self.training:
            fired = feature_acts.detach().sum(dim=0) > 0
            self.num_tokens_since_fired += self._inactive_increment(x.shape[0])
            self.num_tokens_since_fired[fired] = 0
        return feature_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def _auxiliary_loss(self, residual: torch.Tensor, dense_acts: torch.Tensor) -> torch.Tensor:
        if self.aux_penalty <= 0:
            return residual.new_tensor(0.0)
        dead_features = self.dead_feature_mask()
        if not dead_features.any():
            return residual.new_tensor(0.0)
        aux_acts = torch.zeros_like(dense_acts)
        dead_dense = dense_acts[:, dead_features]
        aux_sparse_dead = self._batch_topk(dead_dense, self.aux_k)
        aux_acts[:, dead_features] = aux_sparse_dead
        aux_recon = aux_acts @ self.W_dec
        residual_target = residual.detach()
        denom = residual_target.pow(2).sum(dim=-1).mean().clamp_min(1e-8)
        return self.aux_penalty * F.mse_loss(aux_recon, residual_target) / denom

    def forward(self, x: torch.Tensor, update_inactive: bool = True) -> BatchTopKSAEOutput:
        x = x.float()
        x_processed, info = self.preprocess(x)
        dense_acts = self.encode_dense(x_processed)
        feature_acts = self._batch_topk(dense_acts, self.top_k)
        if update_inactive and self.training:
            fired = feature_acts.detach().sum(dim=0) > 0
            self.num_tokens_since_fired += self._inactive_increment(x.shape[0])
            self.num_tokens_since_fired[fired] = 0
        sae_processed = self.decode(feature_acts)
        sae_out = self.postprocess(sae_processed, info)
        mse_loss = F.mse_loss(sae_processed, x_processed)
        aux_loss = self._auxiliary_loss(x_processed - sae_processed, dense_acts)
        loss = mse_loss + aux_loss
        return BatchTopKSAEOutput(
            sae_out=sae_out,
            feature_acts=feature_acts,
            num_active=int((feature_acts > 0).sum().item()),
            mse_loss=mse_loss,
            aux_loss=aux_loss,
            loss=loss,
        )

    @torch.no_grad()
    def explained_variance(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        residual_var = (x - recon).pow(2).sum()
        total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
        return 1.0 - residual_var / total_var
