"""
Task-specific heads for the MGM2 masked abundance model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbundanceBinHead(nn.Module):
    """Head for abundance bin classification."""

    def __init__(self, hidden_size: int, num_bins: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bins = num_bins

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, num_bins)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.decoder(x)


class RegressionHead(nn.Module):
    """Head for continuous abundance regression in log-abundance space."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.decoder(x).squeeze(-1)


class Qwen3AlignmentHead(nn.Module):
    """Projection head for aligning CLS states to Qwen3 embedding space."""

    def __init__(self, hidden_size: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        x = self.dense(cls_hidden)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return F.normalize(x, p=2, dim=-1)


class ClassificationHead(nn.Module):
    """CLS-level classification head for downstream sample labels."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        if num_classes <= 0:
            raise ValueError('num_classes must be positive')
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        x = self.dense(cls_hidden)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.decoder(x)
