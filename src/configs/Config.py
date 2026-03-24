"""Configuration for the MGM2 masked abundance model."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


@dataclass
class AbundanceConfig:
    min_abundance: float = 1e-4
    output_min: float = -2.0
    output_max: float = 2.0
    normalize: bool = False
    num_bins: int = 5
    binning_method: str = 'quantile'
    fixed_thresholds: List[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    enable_regression: bool = True
    regression_top_k_bins: int = 1

    def __post_init__(self):
        if self.min_abundance <= 0:
            raise ValueError('min_abundance must be positive')
        if self.output_min >= self.output_max:
            raise ValueError('output_min must be smaller than output_max')
        if self.num_bins < 2:
            raise ValueError('At least 2 bins are required')
        if self.binning_method not in {'quantile', 'fixed_threshold'}:
            raise ValueError('binning_method must be quantile or fixed_threshold')
        if self.binning_method == 'fixed_threshold' and len(self.fixed_thresholds) != self.num_bins:
            raise ValueError(
                f'Number of thresholds ({len(self.fixed_thresholds)}) must match num_bins ({self.num_bins})'
            )
        if not 1 <= self.regression_top_k_bins <= (self.num_bins - 1):
            raise ValueError('regression_top_k_bins must be between 1 and num_bins - 1')


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    embedding_dim: int = 1536
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 2048
    pad_token_id: int = -1
    cls_token_id: int = -1
    mlp_dropout: float = 0.1


@dataclass
class LossConfig:
    lambda_bin: float = 1.0
    lambda_regression: float = 0.3
    bin_loss_type: str = 'cross_entropy'
    regression_loss_type: str = 'mse'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self):
        if self.lambda_bin < 0:
            raise ValueError('lambda_bin must be non-negative')
        if self.lambda_regression < 0:
            raise ValueError('lambda_regression must be non-negative')
        if self.bin_loss_type not in {'cross_entropy', 'focal'}:
            raise ValueError('bin_loss_type must be cross_entropy or focal')
        if self.regression_loss_type not in {'mse', 'mae'}:
            raise ValueError('regression_loss_type must be mse or mae')


@dataclass
class TrainingConfig:
    mask_prob: float = 0.15
    mask_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    precision: str = 'bf16'
    devices: List[int] = field(default_factory=lambda: [0])
    accelerator: str = 'auto'

    def __post_init__(self):
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError('mask_prob must be between 0 and 1')
        if len(self.mask_ratio) != 3:
            raise ValueError('mask_ratio must contain exactly three values: (mask, random, keep)')


@dataclass
class Qwen3AlignmentConfig:
    enabled: bool = False
    embedding_path: str = ''
    projection_dim: int = 1536
    temperature: float = 0.07
    loss_weight: float = 0.1

    def __post_init__(self):
        if self.projection_dim <= 0:
            raise ValueError('projection_dim must be positive')
        if self.temperature <= 0:
            raise ValueError('temperature must be positive')
        if self.loss_weight < 0:
            raise ValueError('loss_weight must be non-negative')


@dataclass
class MGM2Config:
    abundance: AbundanceConfig = field(default_factory=AbundanceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    qwen3_alignment: Qwen3AlignmentConfig = field(default_factory=Qwen3AlignmentConfig)
    ntv3_embedding_path: str = ''
    data_path: str = ''
    output_dir: str = './outputs_mgm2'
    experiment_name: str = 'mgm2_experiment'
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MGM2Config':
        allowed_keys = {
            'abundance',
            'model',
            'loss',
            'training',
            'qwen3_alignment',
            'ntv3_embedding_path',
            'data_path',
            'output_dir',
            'experiment_name',
            'seed',
        }
        unexpected_keys = sorted(set(config_dict) - allowed_keys)
        if unexpected_keys:
            raise TypeError(f'Unexpected MGM2Config keys: {unexpected_keys}')

        return cls(
            abundance=AbundanceConfig(**config_dict.get('abundance', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            qwen3_alignment=Qwen3AlignmentConfig(**config_dict.get('qwen3_alignment', {})),
            ntv3_embedding_path=config_dict.get('ntv3_embedding_path', ''),
            data_path=config_dict.get('data_path', ''),
            output_dir=config_dict.get('output_dir', './outputs_mgm2'),
            experiment_name=config_dict.get('experiment_name', 'mgm2_experiment'),
            seed=config_dict.get('seed', 42),
        )

    @classmethod
    def load(cls, path: str) -> 'MGM2Config':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __str__(self):
        return (
            f"MGM2Config(experiment_name='{self.experiment_name}', "
            f"model={self.model.hidden_size}h_{self.model.num_hidden_layers}l)"
        )
