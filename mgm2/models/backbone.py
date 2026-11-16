from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Transformer encoder for microbial communities."""

    def __init__(
        self,
        vocab_size: int = 10000,  # Retained for tokenizer compatibility; not used directly.
        embedding_dim: int = 1536,  # Static NTv3 embedding dimension.
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
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

        # Learn a separate CLS embedding without abundance conditioning.
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
            norm_first=True,  # Pre-layer-normalization architecture.
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_hidden_layers,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize linear and embedding weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.hidden_size**-0.5)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.hidden_size**-0.5)

    def get_input_embeddings(self):
        """Return the input projection for fine-tuning."""
        return self.embedding_projection

    def set_input_embeddings(self, value):
        """Replace the input embedding projection."""
        self.embedding_projection = value

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch_size, seq_len]; identifies special tokens.
        embeddings: torch.Tensor,  # [batch_size, seq_len, embedding_dim]
        abundance_values: torch.Tensor,  # [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        head_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode microbial identities conditioned on abundance.

        Args:
            input_ids: Token IDs identifying CLS positions.
            embeddings: NTv3 vectors with a placeholder at the CLS position.
            abundance_values: Transformed abundance values for each token.
            attention_mask: Valid-token mask; zero marks padding.
            head_mask: Reserved for API compatibility; not used.
            return_dict: Return a dictionary when True, otherwise a tuple.

        Returns:
            Final hidden states and optional outputs, as a dictionary or tuple.
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

        # Replace CLS positions with the learned embedding.
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

        if attention_mask is not None:
            # Convert the valid-token mask to a padding mask (True means ignored).
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
