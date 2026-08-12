__all__ = [
    "rng",
    "BYTES_PER_FLOAT32",
    "KB_TO_BYTES",
    "MB_TO_BYTES",
    "EmbeddingBackward",
    "Embedding",
    "PositionalEncoding",
    "create_sinusoidal_embeddings",
    "EmbeddingLayer",
    "emblayer_forward",
]

import numpy as np

rng = np.random.default_rng(7)
import math
from typing import List, Optional, Tuple

# Import from previous modules - following dependency chain
from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.autograd import Function

# Enable autograd for gradient tracking (required for learnable embeddings)
# from tinytorch.foundation.autograd import Function, enable_autograd

# enable_autograd()

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


class EmbeddingBackward(Function):
    """
    Gradient computation for embedding lookup operation.
    """

    def __init__(self, weight, indices):
        """
        Args:
            weight: Embedding weight matrix
            indices: Indices used for lookup
        """
        super().__init__(weight)
        self.indices = indices

    def apply(self, grad_output):
        """
        Compute gradient for embedding lookup.
        """

        (weight,) = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:
            # Initialize gradient with zeros
            grad_weight = np.zeros_like(weight.data)

            # Scatter gradients back to embedding weights
            # np.add.at accumulates gradients for repeated indices
            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight,)


class Embedding:
    """
    Learnable embedding layer that maps token indices to dense vectors.

    This is the fundamental building block for converting discrete tokens
    into continuous representations that neural networks can process.

    We'll build this in two steps: first initialize the weight matrix,
    then implement the forward lookup.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize embedding layer with Xavier-uniform weights.
        """

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier initialization for better gradient flow
        limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Tensor(rng.uniform(-limit, limit, (vocab_size, embed_dim)))

    def forward(self, indices: Tensor) -> Tensor:
        """
        Forward pass: lookup embeddings for given indices.
        """

        # Handle input validation
        if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
            min_idx = int(np.min(indices.data))
            max_idx = int(np.max(indices.data))
            raise ValueError(
                f"Embedding index out of range for vocabulary size {self.vocab_size}\n"
                f"  ❌ Found indices: min={min_idx}, max={max_idx} (valid range: 0 to {self.vocab_size - 1})\n"
                f"  💡 Token IDs must be within the vocabulary. IDs >= vocab_size reference non-existent tokens\n"
                f"  🔧 Check your tokenizer output, or increase vocab_size to at least {max_idx + 1}"
            )

        # Perform embedding lookup using advanced indexing
        # This is equivalent to one-hot multiplication but much more efficient
        embedded = self.weight.data[indices.data.astype(int)]

        result = Tensor(embedded)

        # Attach gradient function for backpropagation
        # EmbeddingBackward (defined above) handles sparse gradient accumulation
        if self.weight.requires_grad:
            result.requires_grad = True
            result._grad_fn = EmbeddingBackward(self.weight, indices)

        return result

    def __call__(self, indices: Tensor) -> Tensor:
        """Allows the embedding to be called like a function."""
        return self.forward(indices)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.weight]

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


class PositionalEncoding:
    """
    Learnable positional encoding layer.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Initialize learnable positional encoding.
        """

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Initialize position embedding matrix
        # Smaller initialization than token embeddings since these are additive
        limit = math.sqrt(2.0 / embed_dim)
        self.position_embeddings = Tensor(
            rng.uniform(-limit, limit, (max_seq_len, embed_dim))
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to input embeddings.
        """

        if len(x.shape) == 2:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got 2D: {x.shape}\n"
                f"   Missing batch dimension\n"
                f"   PositionalEncoding expects batched embeddings, not single sequences\n"
                f"   Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]})"
            )
        elif len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got {len(x.shape)}D: {x.shape}\n"
                f"   Input must have exactly 3 dimensions\n"
                f"   PositionalEncoding expects shape (batch_size, sequence_length, embedding_dim)"
            )

        batch_size, seq_len, embed_dim = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeds maximum: {seq_len} > {self.max_seq_len}\n"
                f"   Input sequence has {seq_len} positions, but max_seq_len is {self.max_seq_len}\n"
                f"   Learned positional encodings have a fixed maximum length set at initialization\n"
                f"   Either truncate input to {self.max_seq_len} tokens, or create a new PositionalEncoding(max_seq_len={seq_len}, ...)"
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f"   PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n"
                f"   Token embeddings and positional encodings must have the same dimension to be added together\n"
                f"   Ensure your Embedding layer uses embed_dim={self.embed_dim}, or create PositionalEncoding(embed_dim={embed_dim}, ...)"
            )

        # Slice position embeddings for this sequence length using Tensor slicing
        pos_embeddings = self.position_embeddings[:seq_len]  # (seq_len, embed_dim)

        # Reshape to add batch dimension: (1, seq_len, embed_dim)
        pos_data = pos_embeddings.data[np.newaxis, :, :]
        pos_embeddings_batched = Tensor(pos_data)

        # Add positional information
        result = x + pos_embeddings_batched

        return result

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the positional encoding to be called like a function."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.position_embeddings]

    def __repr__(self):
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"


def _compute_sinusoidal_table(max_len: int, embed_dim: int) -> np.ndarray:
    """
    Compute the raw sinusoidal positional encoding table as a numpy array.
    """

    # Create position indices [0, 1, 2, ..., max_len-1]
    position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]  # (max_len, 1)

    # Create dimension indices for calculating frequencies
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) * -(math.log(10000.0) / embed_dim)
    )  # (embed_dim//2,)

    # Initialize the positional encoding matrix
    pe = np.zeros((max_len, embed_dim), dtype=np.float32)

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    if embed_dim % 2 == 1:
        # Handle odd embed_dim by only filling available positions
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)

    return pe


def create_sinusoidal_embeddings(max_seq_len: int, embed_dim: int) -> Tensor:
    """
    Create sinusoidal positional encodings as used in "Attention Is All You Need".
    """

    pe = _compute_sinusoidal_table(max_seq_len, embed_dim)
    return Tensor(pe)


class EmbeddingLayer:
    """
    Complete embedding system combining token and positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        pos_encoding: str = "learned",
        scale_embeddings: bool = False,
    ):
        """
        Initialize complete embedding system.
        """

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding
        self.scale_embeddings = scale_embeddings

        # Token embedding layer
        self.token_embedding = Embedding(vocab_size, embed_dim)

        # Positional encoding
        if pos_encoding == "learned":
            self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        elif pos_encoding == "sinusoidal":
            # Create fixed sinusoidal encodings (no parameters)
            self.pos_encoding = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        elif pos_encoding is None:
            self.pos_encoding = None
        else:
            raise ValueError(
                f"Unknown positional encoding type: '{pos_encoding}'\n"
                f"  ❌ pos_encoding must be 'learned', 'sinusoidal', or None\n"
                f"  💡 'learned' = trainable position embeddings (task-specific but fixed max length)\n"
                f"     'sinusoidal' = mathematical sin/cos patterns (no parameters, can extrapolate)\n"
                f"     None = no positional encoding (order-agnostic model)\n"
                f"  🔧 Use: EmbeddingLayer(..., pos_encoding='learned') or pos_encoding='sinusoidal'"
            )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass through complete embedding system.
        """

        # Handle 1D input by adding batch dimension
        if len(tokens.shape) == 1:
            # NOTE: Tensor reshape preserves gradients
            tokens = tokens.reshape(1, -1)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Get token embeddings
        token_embeds = self.token_embedding.forward(tokens)  # (batch, seq, embed)

        # Scale embeddings if requested (transformer convention)
        if self.scale_embeddings:
            scale_factor = math.sqrt(self.embed_dim)
            token_embeds = (
                token_embeds * scale_factor
            )  # Use Tensor multiplication to preserve gradients

        # Add positional encoding
        if self.pos_encoding_type == "learned":
            # Use learnable positional encoding
            output = self.pos_encoding.forward(token_embeds)
        elif self.pos_encoding_type == "sinusoidal":
            # Use fixed sinusoidal encoding (not learnable)
            batch_size, seq_len, embed_dim = token_embeds.shape
            pos_embeddings = self.pos_encoding[:seq_len]  # Slice using Tensor slicing

            # Reshape to add batch dimension
            pos_data = pos_embeddings.data[np.newaxis, :, :]
            pos_embeddings_batched = Tensor(pos_data)  # Sinusoidal are fixed

            output = token_embeds + pos_embeddings_batched
        else:
            # No positional encoding
            output = token_embeds

        # Remove batch dimension if it was added
        if squeeze_batch:
            # Use Tensor slicing (now supported in Module 01)
            output = output[0]

        return output

    def __call__(self, tokens: Tensor) -> Tensor:
        """Allows the embedding layer to be called like a function."""
        return self.forward(tokens)

    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        params = self.token_embedding.parameters()
        if self.pos_encoding_type == "learned":
            params.extend(self.pos_encoding.parameters())
        return params

    def __repr__(self):
        return (
            f"EmbeddingLayer(vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"pos_encoding='{self.pos_encoding_type}')"
        )
