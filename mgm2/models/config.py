"""Defaults for the released encoder and reconstruction objectives."""

CONFIG = {
    "hidden_size": 224,
    "layers": 6,
    "heads": 4,
    "intermediate_size": 896,
    "embedding_dim": 1536,
    "qwen_dim": 4096,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "layer_norm_eps": 1e-12,
    "regression_weight": 0.3,
    "qwen_weight": 0.1,
    "temperature": 0.07,
}
VARIANTS = {"S0": (1.0, 0.0), "S1": (0.0, 1.0)}
