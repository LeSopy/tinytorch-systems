__all__ = [
    "rng",
    "BYTES_PER_FLOAT32",
    "MB_TO_BYTES",
    "create_causal_mask",
    "LayerNorm",
    "MLP",
    "TransformerBlock",
    "GPT",
]


import numpy as np

rng = np.random.default_rng(7)

from tinytorch.foundation.activations import GELU
from tinytorch.architecture.attention import MultiHeadAttention
from tinytorch.foundation.autograd import Function
from tinytorch.architecture.embeddings import EmbeddingLayer
from tinytorch.foundation.layers import Linear

# Import from previous modules - following proper dependency chain
from tinytorch.foundation.tensor import Tensor

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create a causal (autoregressive) attention mask.
    """
    # Lower triangular matrix: 1 = can attend, 0 = cannot attend
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])  # Add batch dimension


class _LayerNormBackward(Function):
    """
    Gradient computation for the full layer normalization operation.
    """

    def __init__(self, x, gamma, beta, normalized_data, std_data):
        """Initialize with forward pass values needed for gradient computation."""
        super().__init__(x, gamma, beta)
        self.normalized_data = normalized_data
        self.std_data = std_data

    def apply(self, grad_output):
        """Compute gradients for LayerNorm (x, gamma, beta)."""
        x, gamma, beta = self.saved_tensors

        grad_x = grad_gamma = grad_beta = None
        normalized = self.normalized_data
        std_data = self.std_data

        # Gradient for beta: sum over all dims except last
        if isinstance(beta, Tensor) and beta.requires_grad:
            # Sum over batch and sequence dimensions
            grad_beta = grad_output.copy()
            while grad_beta.ndim > 1:
                grad_beta = grad_beta.sum(axis=0)

        # Gradient for gamma: sum of (grad_output * normalized) over batch/seq dims
        if isinstance(gamma, Tensor) and gamma.requires_grad:
            grad_gamma = (grad_output * normalized).copy()
            while grad_gamma.ndim > 1:
                grad_gamma = grad_gamma.sum(axis=0)

        # Gradient for x: full LayerNorm backward formula
        if isinstance(x, Tensor) and x.requires_grad:
            # grad flowing through gamma: grad_output * gamma
            gamma_data = gamma.data if isinstance(gamma, Tensor) else gamma
            grad_norm = grad_output * gamma_data

            mean_grad = np.mean(grad_norm, axis=-1, keepdims=True)
            mean_grad_norm = np.mean(grad_norm * normalized, axis=-1, keepdims=True)
            grad_x = (1.0 / std_data) * (
                grad_norm - mean_grad - normalized * mean_grad_norm
            )

        return (grad_x, grad_gamma, grad_beta)


class LayerNorm:
    """
    Layer Normalization for transformer blocks.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize LayerNorm with learnable parameters.
        """

        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters: scale and shift
        self.gamma = Tensor(
            np.ones(normalized_shape), requires_grad=True
        )  # Scale parameter
        self.beta = Tensor(
            np.zeros(normalized_shape), requires_grad=True
        )  # Shift parameter

    def forward(self, x):
        """
        Apply layer normalization.
        """

        # Compute statistics across last dimension (features)
        mean_data = np.mean(x.data, axis=-1, keepdims=True)

        # Compute variance: E[(x - μ)²]
        diff = x.data - mean_data
        variance = np.mean(diff * diff, axis=-1, keepdims=True)

        # Normalize: (x - mean) / sqrt(variance + eps)
        std_data = np.sqrt(variance + self.eps)
        normalized_data = diff / std_data

        # Apply learnable transformation: gamma * normalized + beta
        output_data = self.gamma.data * normalized_data + self.beta.data
        output = Tensor(output_data)

        # Attach gradient function for full LayerNorm backward
        if x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad:
            output.requires_grad = True
            output._grad_fn = _LayerNormBackward(
                x, self.gamma, self.beta, normalized_data, std_data
            )

        return output

    def __call__(self, x):
        """Allows the layer norm to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return learnable parameters."""
        return [self.gamma, self.beta]


class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Network) for transformer blocks.
    """

    def __init__(self, embed_dim, hidden_dim=None, dropout_prob=0.1):
        """
        Initialize MLP with two linear layers.
        """

        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Standard 4x expansion

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Two-layer feed-forward network
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()  # Use GELU activation from activations module
        self.linear2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass through MLP.
        """

        # First linear layer with expansion
        hidden = self.linear1.forward(x)

        # GELU activation (YOUR activation from Module 03!)
        hidden = self.gelu.forward(hidden)

        # Second linear layer back to original size
        output = self.linear2.forward(hidden)

        return output

    def __call__(self, x):
        """Allows the MLP to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params


class TransformerBlock:
    """
    Complete Transformer Block with self-attention, MLP, and residual connections.
    """

    def __init__(
        self, embed_dim, num_heads, mlp_ratio=4, ff_dim=None, dropout_prob=0.1
    ):
        """
        Initialize a complete transformer block.
        """

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Layer normalizations (pre-norm architecture)
        self.ln1 = LayerNorm(embed_dim)  # Before attention
        self.ln2 = LayerNorm(embed_dim)  # Before MLP

        # Feed-forward network
        # Support both mlp_ratio and explicit ff_dim for backward compatibility
        if ff_dim is not None:
            hidden_dim = ff_dim
        else:
            hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.
        """

        # First sub-layer: Multi-head self-attention with residual connection
        # Pre-norm: LayerNorm before attention
        normed1 = self.ln1.forward(x)
        # Self-attention: query, key, value are all the same (normed1)
        attention_out = self.attention.forward(normed1, mask)

        # Residual connection
        x = x + attention_out

        # Second sub-layer: MLP with residual connection
        # Pre-norm: LayerNorm before MLP
        normed2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(normed2)

        # Residual connection
        output = x + mlp_out

        return output

    def __call__(self, x, mask=None):
        """Allows the transformer block to be called like a function."""
        return self.forward(x, mask)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params


class GPT:
    """
    Complete GPT (Generative Pre-trained Transformer) model.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024):
        """
        Initialize complete GPT model.
        """

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Embedding layer
        self.embedding_layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ]

        # Final layer normalization
        self.ln_f = LayerNorm(embed_dim)

    def forward(self, tokens):
        """
        Forward pass through GPT model.
        """

        batch_size, seq_len = tokens.shape

        # Pass tokens to embedding layer to get token embeddings and positional embeddings
        x = self.embedding_layer.forward(tokens)

        # Create causal mask for autoregressive generation
        mask = self._create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer normalization
        x = self.ln_f.forward(x)

        # Tied projection: x @ embedding_weight.T
        # embedding weight is (vocab_size, embed_dim), so .T gives (embed_dim, vocab_size)
        # which is exactly what we need to project (batch, seq, embed_dim) → (batch, seq, vocab_size)
        logits = x.matmul(self.embedding_layer.token_embedding.weight.transpose(0, 1))

        return logits

    def __call__(self, tokens):
        """Allows the GPT model to be called like a function."""
        return self.forward(tokens)

    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions."""

        # Lower triangular binary mask: 1=allow (past/present), 0=block (future)
        # _apply_mask in module 12 expects this convention: adder = (1-mask)*MASK_VALUE
        mask = np.tril(np.ones((seq_len, seq_len)))
        return Tensor(mask)

    def _sample_next_token(self, logits, temperature=1.0):
        """
        Sample one token from vocabulary logits using temperature scaling.
        """

        batch_size = logits.shape[0]

        # Greedy decoding for zero or near-zero temperature
        if temperature <= 1e-7:
            return np.argmax(logits, axis=-1, keepdims=True)

        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Convert to probabilities (softmax with numerical stability)
        exp_logits = np.exp(
            scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        )
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Sample independently across the batch dimension
        next_tokens = np.empty((batch_size, 1), dtype=np.int64)
        for b in range(batch_size):
            next_tokens[b, 0] = rng.choice(self.vocab_size, p=probs[b])

        return next_tokens

    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """
        Generate text autoregressively by repeatedly sampling next tokens.
        """

        current_tokens = Tensor(prompt_tokens.data.copy())
        batch_size, prompt_len = current_tokens.shape

        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self.forward(current_tokens)

            # Get logits for last position (next token prediction)
            last_logits = logits.data[:, -1, :]  # (batch_size, vocab_size)

            # Sample next token using helper
            next_token_id = self._sample_next_token(last_logits, temperature)

            # Append to sequence
            current_tokens = Tensor(
                np.concatenate([current_tokens.data, next_token_id], axis=1)
            )

        return current_tokens

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.embedding_layer.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.ln_f.parameters())

        return params
