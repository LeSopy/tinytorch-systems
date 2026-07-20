"""
Module 06: Autograd - Core Functionality Tests
Tests automatic differentiation and computational graphs

"""

import numpy as np
import pytest

from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.autograd import enable_autograd, no_grad


class TestBasicOperations:
    """Test basic operations with gradient computation."""

    def test_addition_gradient(self):
        """Test gradient computation for addition."""

        x = Tensor(np.array([2.0]), requires_grad=True)
        y = Tensor(np.array([3.0]), requires_grad=True)

        z = x + y

        assert np.array_equal(z.data, [5.0])

        if hasattr(z, "backward"):
            z.backward()

            # d(x+y)/dx = 1, d(x+y)/dy = 1
            assert np.array_equal(x.grad, [1.0])
            assert np.array_equal(y.grad, [1.0])

    def test_multiplication_gradient(self):
        """Test gradient computation for multiplication."""

        x = Tensor(np.array([3.0]), requires_grad=True)
        y = Tensor(np.array([4.0]), requires_grad=True)

        z = x * y

        assert np.array_equal(z.data, [12.0])

        if hasattr(z, "backward"):
            z.backward()

            # d(x*y)/dx = y, d(x*y)/dy = x
            assert np.array_equal(x.grad, [4.0])
            assert np.array_equal(y.grad, [3.0])

    def test_power_gradient(self):
        """Test gradient computation for power operation."""

        x = Tensor(np.array([3.0]), requires_grad=True)

        # z = x²
        z = x**2

        assert np.array_equal(z.data, [9.0])

        if hasattr(z, "backward"):
            z.backward()

            # d(x²)/dx = 2x = 2*3 = 6
            assert np.array_equal(x.grad, [6.0])


class TestChainRule:
    """Test chain rule application."""

    def test_simple_chain_rule(self):
        """Test chain rule with simple composition."""

        x = Tensor(np.array([2.0]), requires_grad=True)

        # z = (x + 1)²
        y = x + 1  # y = 3
        z = y * y  # z = 9

        if hasattr(z, "backward"):
            z.backward()

            # dz/dx = dz/dy * dy/dx = 2y * 1 = 2*3 = 6
            assert np.array_equal(x.grad, [6.0])

    def test_complex_chain_rule(self):
        """Test chain rule with more complex composition."""

        x = Tensor(np.array([2.0]), requires_grad=True)

        # z = (x²)² = x⁴
        y = x * x  # y = x²
        z = y * y  # z = (x²)²

        if hasattr(z, "backward"):
            z.backward()

            # dz/dx = 4x³ = 4 * 2³ = 32
            assert np.array_equal(x.grad, [32.0])

    def test_multiple_variable_chain(self):
        """Test chain rule with multiple variables."""

        x = Tensor(np.array([2.0]), requires_grad=True)
        y = Tensor(np.array([3.0]), requires_grad=True)

        # z = (x + y)²
        u = x + y  # u = 5
        z = u * u  # z = 25

        if hasattr(z, "backward"):
            z.backward()

            # dz/dx = dz/du * du/dx = 2u * 1 = 2*5 = 10
            # dz/dy = dz/du * du/dy = 2u * 1 = 2*5 = 10
            assert np.array_equal(x.grad, [10.0])
            assert np.array_equal(y.grad, [10.0])


class TestComputationGraph:
    """Test computation graph construction and traversal."""

    def test_graph_construction(self):
        """Test that computation graph is built correctly."""

        x = Tensor(np.array([1.0]), requires_grad=True)
        y = x + 1
        z = y * 2

        # Each operation should create new nodes
        assert isinstance(y, Tensor)
        assert isinstance(z, Tensor)

        # Should track computation history
        if hasattr(z, "grad_fn") or hasattr(z, "_backward_fn"):
            assert True  # Has some form of backward tracking

    def test_graph_backward_traversal(self):
        """Test backward pass traverses graph correctly."""

        x = Tensor(np.array([2.0]), requires_grad=True)
        y = Tensor(np.array([3.0]), requires_grad=True)

        # Build computation graph
        u = x * y  # u = 6
        v = u + x  # v = 8
        w = v * 2  # w = 16

        if hasattr(w, "backward"):
            w.backward()

            # dw/dx = dw/dv * (dv/du * du/dx + dv/dx) = 2 * (y + 1) = 2 * 4 = 8
            # dw/dy = dw/dv * dv/du * du/dy = 2 * 1 * x = 2 * 2 = 4
            assert np.array_equal(x.grad, [8.0])
            assert np.array_equal(y.grad, [4.0])

    def test_graph_memory_management(self):
        """Test computation graph doesn't cause memory leaks."""

        # Create many operations
        x = Tensor(np.array([1.0]), requires_grad=True)
        result = x

        for i in range(100):
            result = result * 1.01  # Small multiplications

        if hasattr(result, "backward"):
            result.backward()

            # Should complete without memory issues
            assert x.grad is not None
            assert x.grad.size == 1


class TestGradientAccumulation:
    """Test gradient accumulation and zeroing."""

    def test_gradient_accumulation(self):
        """Test gradients accumulate across multiple backward passes."""

        x = Tensor(np.array([1.0]), requires_grad=True)

        # First computation
        y1 = x * 2
        if hasattr(y1, "backward"):
            y1.backward()
            first_grad = x.grad.copy() if x.grad is not None else None

            # Second computation (gradients should accumulate)
            y2 = x * 3
            y2.backward()

            if first_grad is not None and x.grad is not None:
                # Gradient should be sum: 2 + 3 = 5
                assert np.array_equal(x.grad, [5.0])

    def test_gradient_zeroing(self):
        """Test gradient zeroing functionality."""

        x = Tensor(np.array([1.0]), requires_grad=True)

        # Compute gradient
        y = x * 5
        if hasattr(y, "backward"):
            y.backward()

            if x.grad is not None:
                assert np.array_equal(x.grad, [5.0])

                # Zero gradients
                if hasattr(x, "zero_grad"):
                    x.zero_grad()
                    assert x.grad is None or np.array_equal(x.grad, [0.0])


class TestAutogradUtilities:
    """Test autograd utility functions."""

    def test_no_grad_context(self):
        """Test no_grad context manager."""

        x = Tensor(np.array([1.0]), requires_grad=True)

        with no_grad():
            y = x * 2

            # Operations in no_grad should not require gradients
            assert not y.requires_grad

    def test_detach_operation(self):
        """Test detaching variables from computation graph."""

        x = Tensor(np.array([2.0]), requires_grad=True)
        y = x * 3

        if hasattr(y, "detach"):
            z = y.detach()

            # Detached variable should not require gradients
            assert not z.requires_grad
            assert np.array_equal(z.data, y.data)
