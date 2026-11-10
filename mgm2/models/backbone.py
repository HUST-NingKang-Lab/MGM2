from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    微生物组基础Transformer模型

    模仿transformers库的BaseModel结构
    """

    def __init__(
        self,
        vocab_size: int = 10000,  # 虽然不直接使用，但保留以兼容tokenizer
        embedding_dim: int = 1536,  # NTv3 embedding维度
        hidden_size: int = 512,  # Transformer隐藏层维度
        num_hidden_layers: int = 12,  # Transformer层数
        num_attention_heads: int = 8,  # 注意力头数
        intermediate_size: int = 2048,  # Feed-forward维度
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 2048,
        pad_token_id: int = -1,  # padding token id
        cls_token_id: int = -1,  # CLS token id
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

        # Project OTU identity into hidden space, then modulate it with FiLM from abundance.
        self.embedding_projection = nn.Linear(embedding_dim, hidden_size)
        self.abundance_conditioner = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        # 针对CLS token的独立可学习Embedding (不使用丰度条件化)
        self.cls_embedding = nn.Embedding(1, hidden_size)

        # Layer Norm
        self.input_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN结构，更稳定
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_hidden_layers,
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.hidden_size**-0.5)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.hidden_size**-0.5)

    def get_input_embeddings(self):
        """获取输入embeddings（用于微调时更新）"""
        return self.embedding_projection

    def set_input_embeddings(self, value):
        """设置输入embeddings"""
        self.embedding_projection = value

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch_size, seq_len] 用于区分特殊token
        embeddings: torch.Tensor,  # [batch_size, seq_len, embedding_dim]
        abundance_values: torch.Tensor,  # [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        head_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            input_ids: 用于标识 CLS token
            embeddings: NTv3 embeddings + CLS embedding
            abundances: 丰度值
            attention_mask: Attention mask（0表示padding）
            head_mask: 注意力头mask
            output_attentions: 是否返回注意力权重
            output_hidden_states: 是否返回隐藏状态
            return_dict: 是否返回字典格式

        Returns:
            根据参数返回不同格式的输出
        """
        output_hidden_states = self.output_hidden_states
        output_attentions = self.output_attentions
        batch_size, seq_len, _ = embeddings.size()

        # Project OTU identity, then let abundance condition it through FiLM.
        base_hidden = self.embedding_projection(embeddings)
        abundances_expanded = abundance_values.unsqueeze(-1)  # [batch_size, seq_len, 1]
        film_params = self.abundance_conditioner(abundances_expanded)
        gamma, beta = film_params.chunk(2, dim=-1)
        hidden_states = base_hidden * (1.0 + gamma) + beta

        # 替换 CLS token 的特征
        cls_positions = input_ids == self.cls_token_id
        if cls_positions.any():
            cls_embs = self.cls_embedding(
                torch.zeros_like(input_ids[cls_positions], dtype=torch.long)
            )
            hidden_states[cls_positions] = cls_embs.to(dtype=hidden_states.dtype)

        # Layer Norm
        hidden_states = self.input_layer_norm(hidden_states)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # 处理attention mask
        if attention_mask is not None:
            # 将0/1 mask转换为bool mask (True表示忽略)
            attention_mask = attention_mask == 0

        # Transformer encoder
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.encoder(
            hidden_states,
            src_key_padding_mask=attention_mask,
        )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
        else:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None
            )
