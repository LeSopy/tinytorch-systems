"""
Module 04: Losses - Core Functionality Tests
=============================================
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss


class TestMSELoss:
    """Test Mean Squared Error loss."""

    def test_mse_computation(self):
        """
        WHAT: Verify MSE = mean((pred - target)²).
        """
        loss_fn = MSELoss()

        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 4.0])  # Error of 1 on last element

        loss = loss_fn(pred, target)

        # MSE = (0² + 0² + 1²) / 3 = 1/3
        expected = 1.0 / 3.0
        assert np.isclose(float(loss.data), expected, atol=1e-5), (
            f"MSE wrong.\n"
            f"  Errors: [0, 0, 1]\n"
            f"  MSE = (0+0+1)/3 = 0.333\n"
            f"  Got: {loss.data}"
        )

class TestCrossEntropyLoss:
    """Test Cross-Entropy loss for classification."""

    def test_cross_entropy_basic(self):
        """
        WHAT: Verify cross-entropy for classification.
        """
        loss_fn = CrossEntropyLoss()

        # Logits for 3 classes
        logits = Tensor([[1.0, 2.0, 0.5]])  # Class 1 has highest
        target = Tensor([1])  # True class is 1

        loss = loss_fn(logits, target)

        # Loss should be small (predicted correct class)
        assert float(loss.data) < 1.0, (
            "CE loss should be small when predicting correct class"
        )

    def test_cross_entropy_wrong_prediction(self):
        """
        WHAT: Verify CE is high when prediction is wrong.
        """
        loss_fn = CrossEntropyLoss()

        # Confident wrong prediction
        logits = Tensor([[10.0, 0.0, 0.0]])  # Very confident class 0
        target = Tensor([2])  # But true class is 2

        loss = loss_fn(logits, target)

        # Loss should be high
        assert float(loss.data) > 1.0, (
            "CE loss should be high for confident wrong predictions"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
