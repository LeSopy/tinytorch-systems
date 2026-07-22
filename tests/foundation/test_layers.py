"""
Module 03: Layers - Core Functionality Tests
=============================================
"""

import numpy as np

rng = np.random.default_rng(7)
import pytest


from tinytorch.foundation.layers import Layer
from tinytorch.foundation.tensor import Tensor


class TestLayerBaseClass:
    """
    Test the Layer base class.
    """

    def test_layer_creation(self):
        """
        WHAT: Verify Layer base class can be instantiated.
        """
        layer = Layer()
        assert layer is not None

    def test_layer_interface(self):
        """
        WHAT: Verify Layer has the required interface.
        """
        layer = Layer()

        assert hasattr(layer, "forward"), (
            "Layer must have forward() method.\n"
            "This is where the computation happens."
        )

        assert callable(layer), (
            "Layer must be callable (implement __call__).\n"
            "This allows: output = layer(input)"
        )

    def test_layer_inheritance(self):
        """
        WHAT: Verify custom layers can inherit from Layer.
        """

        class IdentityLayer(Layer):
            """A layer that returns its input unchanged."""

            def forward(self, x):
                return x

        layer = IdentityLayer()
        x = Tensor(np.array([1, 2, 3]))
        output = layer(x)

        assert isinstance(output, Tensor)
        assert np.array_equal(
            output.data, x.data
        ), "Identity layer should return input unchanged."


class TestParameterManagement:
    """
    Test how layers manage learnable parameters.
    """

    def test_layer_with_parameters(self):
        """
        WHAT: Verify layers can store trainable parameters.
        """

        class ParameterLayer(Layer):
            def __init__(self, input_size, output_size):
                self.weights = Tensor(rng.standard_normal((input_size, output_size)))
                self.bias = Tensor(np.zeros(output_size))

            def forward(self, x):
                return Tensor(x.data @ self.weights.data + self.bias.data)

        layer = ParameterLayer(5, 3)

        assert hasattr(layer, "weights"), "Layer should store weights"
        assert hasattr(layer, "bias"), "Layer should store bias"
        assert layer.weights.shape == (
            5,
            3,
        ), f"Weights shape wrong: expected (5, 3), got {layer.weights.shape}"

    def test_parameter_initialization(self):
        """
        WHAT: Verify weights are initialized properly.
        """

        class XavierLayer(Layer):
            def __init__(self, size):
                # Xavier initialization
                limit = np.sqrt(6.0 / (size + size))
                self.weights = Tensor(rng.uniform(-limit, limit, (size, size)))

            def forward(self, x):
                return Tensor(x.data @ self.weights.data)

        layer = XavierLayer(10)

        weights_std = np.std(layer.weights.data)
        assert 0.1 < weights_std < 1.0, (
            f"Weight initialization looks wrong.\n"
            f"  std = {weights_std}\n"
            "For Xavier with size=10, expect std ≈ 0.32"
        )

    def test_parameter_shapes(self):
        """
        WHAT: Verify parameter shapes match layer configuration.
        """

        class LinearLayer(Layer):
            def __init__(self, in_features, out_features):
                self.in_features = in_features
                self.out_features = out_features
                self.weights = Tensor(rng.standard_normal((in_features, out_features)))
                self.bias = Tensor(np.zeros(out_features))

            def forward(self, x):
                return Tensor(x.data @ self.weights.data + self.bias.data)

        layer = LinearLayer(128, 64)

        assert layer.weights.shape == (128, 64), (
            f"Weights shape wrong.\n"
            f"  Expected: (128, 64)\n"
            f"  Got: {layer.weights.shape}"
        )
        assert layer.bias.shape == (64,)

        # Test with batch input
        x = Tensor(rng.standard_normal((16, 128)))
        output = layer(x)
        assert output.shape == (16, 64), (
            f"Output shape wrong.\n"
            f"  Input: (16, 128)\n"
            f"  Expected output: (16, 64)\n"
            f"  Got: {output.shape}"
        )


class TestLinearTransformations:
    """
    Test linear transformation layers (y = Wx + b).
    """

    def test_matrix_multiplication_layer(self):
        """
        WHAT: Verify matrix multiplication works correctly.
        """

        class MatMulLayer(Layer):
            def __init__(self, weight_matrix):
                self.weights = Tensor(weight_matrix)

            def forward(self, x):
                return Tensor(x.data @ self.weights.data)

        W = np.array([[1, 2], [3, 4]])  # 2x2
        layer = MatMulLayer(W)

        x = Tensor(np.array([[1, 0], [0, 1]]))  # Identity matrix
        output = layer(x)

        # I @ W = W
        expected = np.array([[1, 2], [3, 4]])
        assert np.array_equal(output.data, expected), (
            f"Matrix multiplication failed.\n"
            f"  I @ W should equal W\n"
            f"  Expected: {expected}\n"
            f"  Got: {output.data}"
        )

    def test_affine_transformation(self):
        """
        WHAT: Verify affine transformation y = Wx + b.
        """

        class AffineLayer(Layer):
            def __init__(self, weights, bias):
                self.weights = Tensor(weights)
                self.bias = Tensor(bias)

            def forward(self, x):
                return Tensor(x.data @ self.weights.data + self.bias.data)

        W = np.array([[1, 0], [0, 1]])  # Identity
        b = np.array([10, 20])  # Offset

        layer = AffineLayer(W, b)
        x = Tensor(np.array([[1, 2]]))
        output = layer(x)

        # [1, 2] @ I + [10, 20] = [11, 22]
        expected = np.array([[11, 22]])
        assert np.array_equal(output.data, expected), (
            f"Affine transformation failed.\n"
            f"  x @ W + b\n"
            f"  [1,2] @ I + [10,20] = [11,22]\n"
            f"  Got: {output.data}"
        )

    def test_batch_processing(self):
        """
        WHAT: Verify layer processes batches correctly.
        """

        class ScaleLayer(Layer):
            def __init__(self):
                self.weights = Tensor(np.array([[2, 0], [0, 3]]))

            def forward(self, x):
                return Tensor(x.data @ self.weights.data)

        layer = ScaleLayer()

        # 3 samples, 2 features each
        x = Tensor(np.array([[1, 1], [2, 2], [3, 3]]))
        output = layer(x)

        expected = np.array([[2, 3], [4, 6], [6, 9]])
        assert np.array_equal(output.data, expected)
        assert output.shape == (3, 2), (
            f"Batch output shape wrong.\n"
            f"  Input: 3 samples\n"
            f"  Expected: (3, 2)\n"
            f"  Got: {output.shape}"
        )


class TestLayerComposition:
    """
    Test composing multiple layers into networks.
    """

    def test_layer_chaining(self):
        """
        WHAT: Verify layers can be chained together.
        """

        class ScaleLayer(Layer):
            def __init__(self, scale):
                self.scale = scale

            def forward(self, x):
                return Tensor(x.data * self.scale)

        class AddLayer(Layer):
            def __init__(self, offset):
                self.offset = offset

            def forward(self, x):
                return Tensor(x.data + self.offset)

        layer1 = ScaleLayer(2)
        layer2 = AddLayer(10)

        x = Tensor(np.array([1, 2, 3]))
        h = layer1(x)  # [2, 4, 6]
        output = layer2(h)  # [12, 14, 16]

        expected = np.array([12, 14, 16])
        assert np.array_equal(output.data, expected), (
            f"Layer chaining failed.\n"
            f"  x = [1, 2, 3]\n"
            f"  → scale by 2 → [2, 4, 6]\n"
            f"  → add 10 → [12, 14, 16]\n"
            f"  Got: {output.data}"
        )

    def test_sequential_layer_composition(self):
        """
        WHAT: Verify Sequential container works.
        """

        class Sequential(Layer):
            def __init__(self, layers):
                self.layers = layers

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class LinearLayer(Layer):
            def __init__(self, weights):
                self.weights = Tensor(weights)

            def forward(self, x):
                return Tensor(x.data @ self.weights.data)

        # 2-layer network
        layer1 = LinearLayer(np.array([[1, 2], [3, 4]]))  # 2→2
        layer2 = LinearLayer(np.array([[1], [1]]))  # 2→1

        network = Sequential([layer1, layer2])

        x = Tensor(np.array([[1, 1]]))
        output = network(x)

        # [1,1] @ [[1,2],[3,4]] = [4, 6]
        # [4,6] @ [[1],[1]] = [10]
        expected = np.array([[10]])
        assert np.array_equal(output.data, expected), (
            f"Sequential composition failed.\n"
            f"  Step 1: [1,1] @ [[1,2],[3,4]] = [4,6]\n"
            f"  Step 2: [4,6] @ [[1],[1]] = [10]\n"
            f"  Got: {output.data}"
        )


class TestLayerUtilities:
    """
    Test utility functions for layers.
    """

    def test_layer_parameter_count(self):
        """
        WHAT: Verify we can count layer parameters.
        """

        class CountableLayer(Layer):
            def __init__(self, in_features, out_features):
                self.weights = Tensor(rng.standard_normal((in_features, out_features)))
                self.bias = Tensor(np.zeros(out_features))

            def parameter_count(self):
                return self.weights.data.size + self.bias.data.size

            def forward(self, x):
                return Tensor(x.data @ self.weights.data + self.bias.data)

        layer = CountableLayer(10, 5)

        # 10*5 weights + 5 biases = 55 parameters
        expected_count = 10 * 5 + 5
        if hasattr(layer, "parameter_count"):
            assert layer.parameter_count() == expected_count, (
                f"Parameter count wrong.\n"
                f"  Linear(10, 5): 10*5 + 5 = 55\n"
                f"  Got: {layer.parameter_count()}"
            )

    def test_layer_output_shape_inference(self):
        """
        WHAT: Verify we can predict output shape.
        """

        class ShapeInferenceLayer(Layer):
            def __init__(self, out_features):
                self.out_features = out_features

            def forward(self, x):
                batch_size = x.shape[0]
                return Tensor(rng.standard_normal((batch_size, self.out_features)))

            def output_shape(self, input_shape):
                return (input_shape[0], self.out_features)

        layer = ShapeInferenceLayer(20)

        if hasattr(layer, "output_shape"):
            out_shape = layer.output_shape((32, 10))
            assert out_shape == (32, 20), (
                f"Shape inference wrong.\n"
                f"  Input: (32, 10)\n"
                f"  Layer out_features: 20\n"
                f"  Expected output: (32, 20)\n"
                f"  Got: {out_shape}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
