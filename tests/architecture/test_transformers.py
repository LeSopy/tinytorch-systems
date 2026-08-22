"""
Module 13: Transformers
"""

import numpy as np

rng = np.random.default_rng(7)
import pytest


from tinytorch.foundation.autograd import enable_autograd
from tinytorch.foundation.tensor import Tensor
from tinytorch.architecture.transformers import TransformerBlock


class TestTransformerBlock:
    """Test Transformer block functionality."""

    def test_transformer_block_shape(self):
        """
        Verify TransformerBlock preserves shape.
        """
        block = TransformerBlock(embed_dim=256, num_heads=8)

        # Sequence of embeddings
        x = Tensor(rng.standard_normal((2, 20, 256)))  # (batch, seq, embed)

        output = block(x)

        assert output.shape == x.shape, (
            f"TransformerBlock should preserve shape.\n"
            f"  Input: {x.shape}\n"
            f"  Output: {output.shape}"
        )

    def test_transformer_stack(self):
        """
        Verify multiple transformer blocks can be stacked.
        """
        # Stack of 4 blocks
        blocks = [TransformerBlock(embed_dim=128, num_heads=4) for _ in range(4)]

        x = Tensor(rng.standard_normal((2, 10, 128)))

        for block in blocks:
            x = block(x)

        assert x.shape == (2, 10, 128), "Shape should be preserved through all blocks"


class TestTransformerGradients:
    """Test gradient flow through transformers."""

    def test_transformer_gradients(self):
        """
        Verify gradients flow through TransformerBlock.
        """
        enable_autograd()

        block = TransformerBlock(embed_dim=64, num_heads=4)
        x = Tensor(rng.standard_normal((1, 5, 64)), requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, (
            "Input should receive gradients through Transformer.\n"
            "Check attention and FFN gradient implementations."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
