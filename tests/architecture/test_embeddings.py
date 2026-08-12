"""
Module 11: Embeddings - Core Functionality Tests
"""

import numpy as np

rng = np.random.default_rng(7)
import pytest

from tinytorch.architecture.embeddings import Embedding, PositionalEncoding
from tinytorch.foundation.tensor import Tensor


class TestEmbeddingLayer:
    """Test Embedding layer functionality."""

    def test_embedding_lookup(self):
        """
        Verify embedding maps IDs to vectors.
        """
        vocab_size = 100
        embed_dim = 64

        embed = Embedding(vocab_size, embed_dim)

        # Token IDs
        tokens = Tensor(np.array([3, 7, 2]))

        output = embed(tokens)

        assert output.shape == (3, 64), (
            f"Embedding output shape wrong.\n"
            f"  Input: 3 token IDs\n"
            f"  Expected: (3, 64)\n"
            f"  Got: {output.shape}"
        )

    def test_embedding_batch(self):
        """
        Verify embedding handles batched sequences.
        """
        embed = Embedding(vocab_size=100, embed_dim=32)

        # Batch of 4 sequences, each length 10
        tokens = Tensor(rng.integers(0, 100, (4, 10)))

        output = embed(tokens)

        assert output.shape == (4, 10, 32), (
            f"Batched embedding shape wrong.\n"
            f"  Input: (4, 10) token IDs\n"
            f"  Expected: (4, 10, 32)\n"
            f"  Got: {output.shape}"
        )


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_positional_encoding_shape(self):
        """
        Verify positional encoding has correct shape.
        """
        max_len = 100
        embed_dim = 64

        pos_enc = PositionalEncoding(max_len, embed_dim)

        # Sequence of embeddings
        x = Tensor(rng.standard_normal((2, 50, 64)))  # (batch, seq, embed)

        output = pos_enc(x)

        assert output.shape == x.shape, "Positional encoding should preserve shape"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
