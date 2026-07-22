"""
Module 08: Training - Core Functionality Tests
===============================================
"""

import numpy as np
import pytest
import sys
from pathlib import Path


class TestTrainingLoop:
    """Test basic training loop functionality."""

    def test_weights_change_after_step(self):
        """
        WHAT: Verify weights change after optimizer.step().
        """
        from tinytorch.foundation.tensor import Tensor
        from tinytorch.foundation.layers import Linear
        from tinytorch.foundation.optimizers import SGD
        from tinytorch.foundation.autograd import enable_autograd

        enable_autograd()

        layer = Linear(2, 1)
        initial_weights = layer.weight.data.copy()

        optimizer = SGD(layer.parameters(), lr=0.1)

        # Forward
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = layer(x)
        loss = y.sum()

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        # Weights should have changed
        assert not np.allclose(layer.weight.data, initial_weights), (
            "Weights didn't change after optimizer.step().\n"
            "This means the model cannot learn."
        )

    def test_loss_decreases(self):
        """
        WHAT: Verify loss decreases over training iterations.
        """
        from tinytorch.foundation.tensor import Tensor
        from tinytorch.foundation.layers import Linear
        from tinytorch.foundation.optimizers import SGD
        from tinytorch.foundation.autograd import enable_autograd

        enable_autograd()

        # Simple linear regression
        layer = Linear(1, 1)
        # Use smaller learning rate to prevent gradient explosion
        optimizer = SGD(layer.parameters(), lr=0.01)

        # Target: y = 2x
        x = Tensor([[1.0], [2.0], [3.0]])
        target = Tensor([[2.0], [4.0], [6.0]])

        losses = []
        for _ in range(10):
            optimizer.zero_grad()

            pred = layer(x)
            diff = pred - target
            loss = (diff * diff).sum()
            losses.append(float(loss.data))

            loss.backward()
            optimizer.step()

        # Loss should generally decrease
        assert losses[-1] < losses[0], (
            f"Loss didn't decrease!\n"
            f"  Initial: {losses[0]:.4f}\n"
            f"  Final: {losses[-1]:.4f}\n"
            "Check learning rate and gradient computation."
        )

    def test_gradient_accumulation_scales_correctly(self):
        """
        WHAT: Verify gradient accumulation scales updates correctly.
        """
        from tinytorch.foundation.tensor import Tensor
        from tinytorch.foundation.layers import Linear
        from tinytorch.foundation.losses import MSELoss
        from tinytorch.foundation.optimizers import SGD
        from tinytorch.foundation.training import Trainer
        from tinytorch.foundation.autograd import enable_autograd

        enable_autograd()

        def build_model():
            layer = Linear(1, 1)
            layer.weight.data = np.array([[1.0]], dtype=np.float32)
            layer.bias.data = np.array([0.0], dtype=np.float32)
            return layer

        x = Tensor([[1.0]])
        y = Tensor([[2.0]])
        two_batches = [(x, y), (x, y)]
        one_batch = [(x, y)]

        model_accum = build_model()
        trainer_accum = Trainer(
            model_accum, SGD(model_accum.parameters(), lr=0.1), MSELoss()
        )
        trainer_accum.train_epoch(two_batches, accumulation_steps=2)

        model_single = build_model()
        trainer_single = Trainer(
            model_single, SGD(model_single.parameters(), lr=0.1), MSELoss()
        )
        trainer_single.train_epoch(one_batch, accumulation_steps=1)

        assert np.allclose(model_accum.weight.data, model_single.weight.data), (
            "Gradient accumulation did not scale updates correctly.\n"
            "Accumulated update should match a single large batch."
        )
        assert np.allclose(
            model_accum.bias.data, model_single.bias.data
        ), "Bias update did not match between accumulation and single batch."


class TestTrainingUtilities:
    """Test training helper functions."""

    def test_zero_grad_clears_gradients(self):
        """
        WHAT: Verify zero_grad() clears gradients.
        """
        from tinytorch.foundation.tensor import Tensor
        from tinytorch.foundation.layers import Linear
        from tinytorch.foundation.optimizers import SGD
        from tinytorch.foundation.autograd import enable_autograd

        enable_autograd()

        layer = Linear(2, 1)
        optimizer = SGD(layer.parameters(), lr=0.1)

        # First backward
        x = Tensor([[1.0, 1.0]])
        y = layer(x)
        y.sum().backward()

        # Clear gradients
        optimizer.zero_grad()

        # Gradients should be cleared
        for param in layer.parameters():
            if param.grad is not None:
                assert np.allclose(
                    param.grad, 0
                ), "zero_grad() should clear all gradients to 0"

    def test_evaluate_handles_generator_dataloader(self):
        """
        WHAT: Ensure evaluate works with generator dataloaders.
        """
        from tinytorch.foundation.tensor import Tensor
        from tinytorch.foundation.training import Trainer

        class DummyOptimizer:
            def __init__(self):
                self.lr = 0.1

            def step(self):
                pass

            def zero_grad(self):
                pass

        class ConstantModel:
            def __init__(self, value):
                self.value = value
                self.training = True

            def forward(self, inputs):
                return Tensor([self.value])

            def parameters(self):
                return []

        class ConstantLoss:
            def forward(self, predictions, targets):
                return Tensor(1.0)

        def data_generator():
            for _ in range(3):
                yield Tensor([1.0]), Tensor([1.0])

        trainer = Trainer(ConstantModel(1.0), DummyOptimizer(), ConstantLoss())

        avg_loss, accuracy = trainer.evaluate(data_generator())
        assert np.allclose(
            avg_loss, 1.0
        ), "Average loss should be 1.0 for constant loss."
        assert np.allclose(
            accuracy, 0.0
        ), "Accuracy should be 0.0 for non class outputs."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
