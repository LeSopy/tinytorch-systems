__all__ = ["rng", "MASK_VALUE", "scaled_dot_product_attention", "MultiHeadAttention"]


import numpy as np

rng = np.random.default_rng(7)
import math
import time
from typing import Optional, Tuple, List

# Import dependencies from previous modules - following TinyTorch dependency chain
from ..foundation.tensor import Tensor
from ..foundation.layers import Linear
from ..foundation.activations import Softmax

# Constants for attention computation
MASK_VALUE = (
    -1e9
)  # Large negative value used for attention masking (becomes ~0 after softmax)


def _compute_attention_scores(Q: Tensor, K: Tensor) -> Tensor:
    """
    Compute raw attention scores via Q @ K^T.
    """

    K_t = K.transpose(-2, -1)
    return Q.matmul(K_t)


def _scale_scores(scores: Tensor, d_model: int) -> Tensor:
    """
    Scale attention scores by 1/sqrt(d_model).
    """

    scale_factor = 1.0 / math.sqrt(d_model)
    return scores * scale_factor


def _apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """Apply causal mask by setting masked positions to -infinity."""

    adder = (Tensor(np.ones_like(mask.data)) - mask) * MASK_VALUE
    return scores + adder


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Complete scaled dot-product attention."""

    scores = _compute_attention_scores(Q, K)
    scores = _scale_scores(scores, Q.shape[-1])
    if mask is not None:
        scores = _apply_mask(scores, mask)
    softmax = Softmax()
    attention_weights = softmax(scores, dim=-1)
    output = attention_weights.matmul(V)
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention mechanism.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize multi-head attention.
        """

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Multi-head attention dimension mismatch\n"
                f"   embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
                f"   Multi-head attention splits embed_dim equally among heads, so embed_dim must be a multiple of num_heads\n"
                f"   Try: embed_dim={num_heads * (embed_dim // num_heads + 1)} (next valid size) or num_heads={embed_dim // (embed_dim // num_heads)} (fewer heads)"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        # Output projection to mix information across heads
        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Reshape to separate attention heads for parallel processing.
        """

        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Merge attention heads back into single embedding dimension.
        """

        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through multi-head attention.
        """

        # Step 1: Extract dimensions and validate
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention input dimension mismatch\n"
                f"   Expected embed_dim={self.embed_dim}, got {embed_dim} from input shape {x.shape}\n"
                f"   The last dimension of input must match embed_dim from initialization (MultiHeadAttention({self.embed_dim}, {self.num_heads}))\n"
                f"   Try: x.reshape({x.shape[0]}, {x.shape[1]}, {self.embed_dim}) or create new MultiHeadAttention({embed_dim}, num_heads)"
            )

        # Step 2: Project to Q, K, V
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # Step 3: Split into heads
        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)

        # Step 4: Apply attention -- reshape mask (batch, seq_q, seq_k) to
        # (batch, 1, seq_q, seq_k) for broadcasting across heads.
        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
            batch_size_mask, seq_q_mask, seq_k_mask = mask.shape
            mask_reshaped = mask.reshape(batch_size_mask, 1, seq_q_mask, seq_k_mask)

        attended, _ = scaled_dot_product_attention(Q, K, V, mask=mask_reshaped)

        # Step 5: Merge heads back together
        concat_output = self._merge_heads(attended, batch_size, seq_len)

        # Step 6: Apply output projection
        output = self.out_proj.forward(concat_output)

        return output

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Make MultiHeadAttention callable like attention(x)."""
        return self.forward(x, mask)

    def parameters(self) -> List[Tensor]:
        """
        Return all trainable parameters.
        """

        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
