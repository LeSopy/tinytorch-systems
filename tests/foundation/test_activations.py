"""
Module 02: Activations - Core Functionality Tests
==================================================
"""

import numpy as np
rng = np.random.default_rng(7)
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.foundation.activations import ReLU, Sigmoid, Tanh, Softmax
from tinytorch.foundation.tensor import Tensor


class TestReLUActivation:
    """
    Test ReLU (Rectified Linear Unit) activation.

    CONCEPT: ReLU(x) = max(0, x)
    """

    def test_relu_forward(self):
        """
        Verify ReLU outputs max(0, x) for each element.
        """
        relu = ReLU()
        x = Tensor(np.array([-2, -1, 0, 1, 2]))
        output = relu(x)

        expected = np.array([0, 0, 0, 1, 2])
        assert np.array_equal(output.data, expected), (
            f"ReLU output wrong.\n"
            f"  Input: {x.data}\n"
            f"  Expected: {expected} (negative -> 0, positive -> unchanged)\n"
            f"  Got: {output.data}"
        )

    def test_relu_gradient_property(self):
        """
        Verify ReLU gradient is 1 for x>0, 0 for x<=0.
        """
        relu = ReLU()
        x = Tensor(np.array([-1, 0, 1, 2]))
        output = relu(x)

        # Where output > 0, gradient passes through (=1)
        # Where output = 0, gradient is blocked (=0)
        gradient_mask = output.data > 0
        expected_mask = np.array([False, False, True, True])
        assert np.array_equal(gradient_mask, expected_mask), (
            "ReLU gradient mask is wrong.\n"
            "Gradient should flow (True) only where output > 0."
        )

    def test_relu_large_values(self):
        """
        Verify ReLU handles extreme values correctly.
        """
        relu = ReLU()
        x = Tensor(np.array([-1000, 1000]))
        output = relu(x)

        expected = np.array([0, 1000])
        assert np.array_equal(output.data, expected), (
            "ReLU failed on extreme values.\n"
            f"  Input: {x.data}\n"
            f"  Expected: {expected}\n"
            f"  Got: {output.data}"
        )


class TestSigmoidActivation:
    """
    Test Sigmoid activation function.
    """

    def test_sigmoid_forward(self):
        """
        Verify sigmoid outputs values between 0 and 1.
        """
        sigmoid = Sigmoid()
        x = Tensor(np.array([0, 1, -1]))
        output = sigmoid(x)

        # Sigmoid(0) = 0.5
        assert np.isclose(output.data[0], 0.5, atol=1e-6), (
            f"Sigmoid(0) should be 0.5, got {output.data[0]}"
        )

        # All outputs must be in (0, 1)
        assert np.all(output.data > 0) and np.all(output.data < 1), (
            f"Sigmoid outputs must be in (0, 1).\n"
            f"  Got: {output.data}\n"
            "This is essential for probability interpretation."
        )

    def test_sigmoid_symmetry(self):
        """
         Verify sigma(-x) = 1 - sigma(x) (point symmetry around 0.5).
        """
        sigmoid = Sigmoid()
        x = 2.0

        pos_out = sigmoid(Tensor([x]))
        neg_out = sigmoid(Tensor([-x]))

        expected = 1 - pos_out.data[0]
        assert np.isclose(neg_out.data[0], expected, atol=1e-6), (
            f"Sigmoid symmetry broken: sigma(-x) should equal 1 - sigma(x)\n"
            f"  sigma({x}) = {pos_out.data[0]}\n"
            f"  sigma({-x}) = {neg_out.data[0]}\n"
            f"  1 - sigma({x}) = {expected}"
        )

    def test_sigmoid_derivative_property(self):
        """
        Verify sigma'(x) = sigma(x) * (1 - sigma(x)).
        """
        sigmoid = Sigmoid()
        x = Tensor(np.array([0, 1, -1]))
        output = sigmoid(x)

        # Derivative = sigma(x) * (1 - sigma(x))
        derivative = output.data * (1 - output.data)

        # At x=0: sigma(0)=0.5, so derivative = 0.5 * 0.5 = 0.25
        assert np.isclose(derivative[0], 0.25, atol=1e-6), (
            f"Sigmoid derivative at x=0 should be 0.25.\n"
            f"  sigma(0) = {output.data[0]}\n"
            f"  sigma'(0) = sigma(0) * (1-sigma(0)) = {derivative[0]}"
        )


class TestTanhActivation:
    """
    Test Tanh (hyperbolic tangent) activation.
    """

    def test_tanh_forward(self):
        """
        WHAT: Verify tanh outputs values between -1 and 1.
        """
        tanh = Tanh()
        x = Tensor(np.array([0, 1, -1]))
        output = tanh(x)

        assert np.isclose(output.data[0], 0, atol=1e-6), (
            f"tanh(0) should be 0, got {output.data[0]}"
        )

        assert np.all(output.data > -1) and np.all(output.data < 1), (
            f"tanh outputs must be in (-1, 1).\n"
            f"  Got: {output.data}"
        )

    def test_tanh_antisymmetry(self):
        """
        WHAT: Verify tanh(-x) = -tanh(x) (odd function).
        """
        tanh = Tanh()
        x = 1.5

        pos_out = tanh(Tensor([x]))
        neg_out = tanh(Tensor([-x]))

        assert np.isclose(neg_out.data[0], -pos_out.data[0], atol=1e-6), (
            f"tanh antisymmetry broken: tanh(-x) should equal -tanh(x)\n"
            f"  tanh({x}) = {pos_out.data[0]}\n"
            f"  tanh({-x}) = {neg_out.data[0]}\n"
            f"  -tanh({x}) = {-pos_out.data[0]}"
        )

    def test_tanh_range(self):
        """
        Verify tanh saturates to +/-1 for extreme inputs.
        """
        tanh = Tanh()
        x = Tensor(np.array([-10, -5, 0, 5, 10]))
        output = tanh(x)

        assert output.data[0] < -0.99, "tanh(-10) should be near -1"
        assert output.data[4] > 0.99, "tanh(10) should be near 1"
        assert np.isclose(output.data[2], 0, atol=1e-6), "tanh(0) should be 0"


class TestSoftmaxActivation:
    """
    Test Softmax activation function.
    """

    def test_softmax_forward(self):
        """
        Verify softmax outputs sum to 1 and are positive.
        """
        softmax = Softmax()
        x = Tensor(np.array([1, 2, 3]))
        output = softmax(x)

        assert np.isclose(np.sum(output.data), 1.0, atol=1e-6), (
            f"Softmax outputs must sum to 1.\n"
            f"  Input: {x.data}\n"
            f"  Output: {output.data}\n"
            f"  Sum: {np.sum(output.data)}"
        )

        assert np.all(output.data > 0), (
            f"Softmax outputs must all be positive.\n"
            f"  Got: {output.data}"
        )

    def test_softmax_properties(self):
        """
        Verify softmax(x + c) = softmax(x) (shift invariance).
        """
        softmax = Softmax()

        x = Tensor(np.array([1, 2, 3]))
        x_shifted = Tensor(np.array([11, 12, 13]))  # x + 10

        out1 = softmax(x)
        out2 = softmax(x_shifted)

        assert np.allclose(out1.data, out2.data, atol=1e-6), (
            f"Softmax should be shift-invariant.\n"
            f"  softmax([1,2,3]) = {out1.data}\n"
            f"  softmax([11,12,13]) = {out2.data}\n"
            "These should be identical."
        )

    def test_softmax_numerical_stability(self):
        """
        Verify softmax handles large values without overflow.
        """
        softmax = Softmax()

        # These values would overflow with naive exp()
        x = Tensor(np.array([1000, 1001, 1002]))
        output = softmax(x)

        assert np.isclose(np.sum(output.data), 1.0, atol=1e-6), (
            "Softmax failed with large values - likely overflow."
        )
        assert np.all(np.isfinite(output.data)), (
            f"Softmax produced NaN/Inf with large values.\n"
            f"  Input: {x.data}\n"
            f"  Output: {output.data}\n"
            "Use the stable formula: exp(x - max(x))."
        )


class TestActivationComposition:
    """
    Test activation functions working together.
    """

    def test_activation_chaining(self):
        """
        Verify activations can be chained together.
        """
        relu = ReLU()
        sigmoid = Sigmoid()

        x = Tensor(np.array([-2, -1, 0, 1, 2]))

        # Chain: x -> ReLU -> Sigmoid
        h = relu(x)      # [-2,-1,0,1,2] -> [0,0,0,1,2]
        output = sigmoid(h)  # -> [0.5,0.5,0.5,0.73,0.88]

        assert output.shape == x.shape
        assert np.all(output.data >= 0) and np.all(output.data <= 1), (
            "Chained activation output should be in sigmoid range [0,1]."
        )

    def test_activation_with_batch_data(self):
        """
        Verify activations handle batch dimensions.
        """
        # Batch of 4 samples, 3 features each
        x = Tensor(rng.standard_normal((4, 3)))

        for name, activation in [("ReLU", ReLU()), ("Sigmoid", Sigmoid()), ("Tanh", Tanh())]:
            output = activation(x)
            assert output.shape == x.shape, (
                f"{name} changed shape!\n"
                f"  Input: {x.shape}\n"
                f"  Output: {output.shape}\n"
                "Activations should preserve shape."
            )

    def test_activation_zero_preservation(self):
        """
        Test how different activations handle zero input.
        """
        zero_input = Tensor(np.array([0.0]))

        relu = ReLU()
        assert relu(zero_input).data[0] == 0.0, "ReLU(0) should be 0"

        sigmoid = Sigmoid()
        assert np.isclose(sigmoid(zero_input).data[0], 0.5, atol=1e-6), (
            "Sigmoid(0) should be 0.5"
        )

        tanh = Tanh()
        assert np.isclose(tanh(zero_input).data[0], 0.0, atol=1e-6), (
            "Tanh(0) should be 0"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
