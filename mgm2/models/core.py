"""Community encoder and CLS-conditioned reconstruction modules."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .backbone import BaseModel
from .config import CONFIG, VARIANTS
from .heads import AbundanceBinHead, Qwen3AlignmentHead, RegressionHead


class CheckpointEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.layers = encoder.layers
        self.norm = encoder.norm

    def forward(self, hidden, src_key_padding_mask=None):
        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    layer, hidden, src_key_padding_mask=src_key_padding_mask, use_reentrant=False
                )
            else:
                hidden = layer(hidden, src_key_padding_mask=src_key_padding_mask)
        return self.norm(hidden) if self.norm is not None else hidden


def build_backbone(config=CONFIG, legacy=False):
    model = BaseModel(
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["layers"],
        num_attention_heads=config["heads"],
        intermediate_size=config["intermediate_size"],
        hidden_dropout_prob=config["dropout"],
        attention_probs_dropout_prob=config["attention_dropout"],
        layer_norm_eps=config["layer_norm_eps"],
    )
    if not legacy:
        for layer in model.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            layer.self_attn.dropout = config["attention_dropout"]
            layer.norm1.eps = layer.norm2.eps = config["layer_norm_eps"]
        model.encoder = CheckpointEncoder(model.encoder)
    return model


class CLSDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["hidden_size"]
        self.identity = nn.Linear(config["embedding_dim"], d)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(config["dropout"])
        )
        self.bin_head = nn.Linear(d, 5)
        self.regression_head = nn.Linear(d, 1)

    def forward(self, cls, identity, sample_index):
        hidden = self.fusion(torch.cat([cls[sample_index], self.identity(identity)], dim=-1))
        return self.bin_head(hidden), self.regression_head(hidden).squeeze(-1)


class PilotModel(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.backbone = build_backbone(config)
        d = config["hidden_size"]
        self.token_bin = AbundanceBinHead(d, 5, config["dropout"])
        self.token_regression = RegressionHead(d, config["dropout"])
        self.cls_decoder = CLSDecoder(config)
        self.qwen_head = Qwen3AlignmentHead(d, config["qwen_dim"], config["dropout"])

    def forward(self, batch, variant):
        inputs = {
            k: batch[k] for k in ["input_ids", "embeddings", "abundance_values", "attention_mask"]
        }
        hidden = self.backbone(**inputs)["last_hidden_state"]
        cls = hidden[:, 0]
        selected = batch["selected"]
        sample_index = selected.nonzero(as_tuple=True)[0]
        result = dict(hidden=hidden, cls=cls, qwen=self.qwen_head(cls))
        wt, wc = VARIANTS[variant]
        if wt:
            tokens = hidden[selected]
            result["token"] = (self.token_bin(tokens), self.token_regression(tokens))
        if wc:
            # Identity is the static upstream vector, never contextual token state
            # or abundance. This is the CLS information bottleneck.
            result["cls_reconstruction"] = self.cls_decoder(
                cls, batch["embeddings"][selected], sample_index
            )
        return result
