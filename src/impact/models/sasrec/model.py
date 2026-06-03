from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast, final, override

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable


@final
class SASRecNBR(nn.Module):
    """SASRec adapted for Next-Basket Recommendation.

    Sessions of K items (baskets) are mean-pooled into session vectors and processed by
    a Transformer encoder over L context sessions. The final hidden state is used as
    the user intent vector for scoring.
    """

    LAYER_NORM_EPS: ClassVar[float] = 1e-9

    def __init__(
        self,
        num_items: int,
        context_sessions: int = 20,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        dim_feedforward: int | None = None,
        activation: Callable[..., torch.Tensor] | None = None,
        initializer_range: float = 0.02,
    ) -> None:
        """Initialize the SASRecNBR model.

        Args:
            num_items: Catalog size. Embedding table is (num_items + 1), with index
                0 reserved for padding.
            context_sessions: Number of historical sessions used as context. Denoted as
                L in the thesis.
            embedding_dim: Dimension of item and positional embeddings.
            num_heads: Number of attention heads per Transformer layer.
            num_layers: Number of Transformer encoder layers.
            dropout: Dropout probability after embedding and within each layer.
            dim_feedforward: FFN hidden size. Defaults to 4 * embedding_dim.
            activation: FFN activation. Defaults to GELU.
            initializer_range: Standard deviation for normal weight initialization.
        """
        super().__init__()
        self.num_items = num_items
        self._initializer_range = initializer_range

        # Reserve index 0 for padding, valid item indices must start at 1
        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=context_sessions,
            embedding_dim=embedding_dim,
        )
        self._layernorm = nn.LayerNorm(embedding_dim, eps=self.LAYER_NORM_EPS)
        self._dropout = nn.Dropout(dropout)

        # Standard Transformer Encoder layers (using batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or embedding_dim * 4,
            dropout=dropout,
            activation=activation or nn.GELU(),
            layer_norm_eps=self.LAYER_NORM_EPS,
            batch_first=True,
        )
        self._transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self._init_weights()

    @property
    def item_embeddings(self) -> nn.Embedding:
        """Item embedding table, exposed for external scoring."""
        return self._item_embeddings

    def _init_weights(self) -> None:
        """Initialize model weights using a truncated normal distribution.

        Truncated normal distribution is used for embeddings and linear layers,
        zeros for biases, ones/zeros for LayerNorm. Padding index is explicitly zeroed
        after init.
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                _ = nn.init.trunc_normal_(
                    tensor=module.weight, mean=0.0, std=self._initializer_range
                )
                if module.padding_idx is not None:
                    _ = nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.Linear):
                _ = nn.init.trunc_normal_(
                    tensor=module.weight, mean=0.0, std=self._initializer_range
                )
                _ = nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                _ = nn.init.ones_(module.weight)
                _ = nn.init.zeros_(module.bias)

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        """Encode a batch of context sessions into a user intent vector.

        Item embeddings are mean-pooled per session (padding-aware), positional
        embeddings are added, then the sequence is passed through the Transformer.

        Args:
            context: [B, L, K] context session item indices. Zero indicates padding.

        Returns:
            User intent vector of shape [B, D] from the last hidden state.
        """
        batch_size, seq_len, _ = context.shape

        # Lookup item embeddings: [B, L, K] -> [B, L, K, D]
        item_vectors = cast(torch.Tensor, self._item_embeddings(context))

        mask = (context != 0).unsqueeze(-1)  # [B, L, K, 1]
        item_vectors = item_vectors * mask

        # Mean pooling (NBR core): [B, L, K, D] -> [B, L, D]
        # Average across the K dimension to construct a single session representation
        session_vectors = item_vectors.sum(dim=2) / mask.sum(dim=2).clamp(min=1)

        # Add learned positional embeddings
        positions = (
            torch.arange(seq_len, device=context.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        position_vectors = cast(torch.Tensor, self._position_embeddings(positions))
        session_vectors = session_vectors + position_vectors
        session_vectors = cast(torch.Tensor, self._layernorm(session_vectors))
        session_vectors = cast(torch.Tensor, self._dropout(session_vectors))

        # Transformer self-attention processing
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=seq_len, device=context.device
        )
        hidden_states = cast(
            torch.Tensor,
            self._transformer(session_vectors, mask=causal_mask, is_causal=True),
        )

        # Extract the user representation (at index -1)
        return hidden_states[:, -1, :]  # [B, D]

    def score_items(
        self,
        hidden_state: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Score the intent vector against a set of items.

        Used during training against explicit positive/negative sets.

        Args:
            hidden_state: [B, D] intent vector.
            item_indices: [B, K] item indices to score.

        Returns:
            Scores (logits) of shape [B, K].
        """
        # Lookup item embeddings: [B, K] -> [B, K, D]
        item_embeddings = cast(torch.Tensor, self._item_embeddings(item_indices))
        # Sum([B, 1, D] * [B, K, D], dim=-1) -> [B, K]
        return torch.sum(hidden_state.unsqueeze(1) * item_embeddings, dim=-1)

    def score_all_items(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Score the intent vector against the full catalog.

        Used during evaluation. Padding index 0 is masked to -inf.

        Args:
            hidden_state: [B, D] intent vector.

        Returns:
            Scores (logits) of shape [B, num_items + 1].
        """
        # Get all item embeddings: # [num_items + 1, D]
        item_embeddings = self._item_embeddings.weight
        # Matrix multiplication: [B, D] * [D, num_items + 1] -> [B, num_items + 1]
        logits = torch.matmul(hidden_state, item_embeddings.t())

        # Mask out the padding index (0) so it is never recommended
        logits = logits.masked_fill(
            mask=(torch.arange(logits.size(1), device=logits.device) == 0),
            value=-torch.inf,
        )
        return logits

    @override
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Encode context into a user intent vector. Delegates to `encode()`.

        Scoring is decoupled: use `score_items()` for training and `score_all_items()`
        for evaluation.
        """
        return self.encode(context)
