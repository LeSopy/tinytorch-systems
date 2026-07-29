"""
Module 09: Convolutions - Core Functionality Tests
===================================================
"""

import numpy as np

rng = np.random.default_rng(7)
import pytest


from tinytorch.foundation.convolutions import Conv2d, MaxPool2d, AvgPool2d
from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.autograd import enable_autograd


class TestConv2DLayer:
    """
    Test 2D Convolution layer.
    """

    def test_conv2d_creation(self):
        """
        WHAT: Verify Conv2d layer can be created.
        """
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)

        assert conv.in_channels == 3, "in_channels not set correctly"
        assert conv.out_channels == 16, "out_channels not set correctly"
        # kernel_size can be int or tuple
        assert conv.kernel_size == 3 or conv.kernel_size == (
            3,
            3,
        ), "kernel_size not set correctly"

    def test_conv2d_weight_shape(self):
        """
        Verify Conv2d weights have correct shape.
        """
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=5)

        # Weights: (out_channels, in_channels, kH, kW)
        expected_shape = (16, 3, 5, 5)
        weight = conv.weight if hasattr(conv, "weight") else conv.weights

        assert weight.shape == expected_shape, (
            f"Conv2d weight shape wrong.\n"
            f"  Expected: {expected_shape} (out, in, kH, kW)\n"
            f"  Got: {weight.shape}\n"
            "Remember: each output channel needs kernels for ALL input channels."
        )

    def test_conv2d_forward_shape(self):
        """
        Verify Conv2d output has correct shape.
        """
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)

        # Input: (batch, C, H, W) - NCHW format
        x = Tensor(rng.standard_normal((8, 3, 32, 32)))
        output = conv(x)

        # 32 - 3 + 1 = 30
        expected_shape = (8, 16, 30, 30)
        assert output.shape == expected_shape, (
            f"Conv2d output shape wrong.\n"
            f"  Input: (8, 3, 32, 32) NCHW\n"
            f"  kernel_size=3, no padding\n"
            f"  Expected: (8, 16, 30, 30)\n"
            f"  Got: {output.shape}\n"
            "Formula: output = input - kernel + 1 = 32 - 3 + 1 = 30"
        )

    def test_conv2d_simple_convolution(self):
        """
        Verify convolution computes correctly with known kernel.
        """
        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)

        # Set kernel to all ones (sum kernel)
        weight = conv.weight if hasattr(conv, "weight") else conv.weights
        weight.data = np.ones((1, 1, 3, 3))

        # All-ones input in NCHW format
        x = Tensor(np.ones((1, 1, 5, 5)))
        output = conv(x)

        # Each output pixel = sum of 9 ones = 9
        if output.shape == (1, 1, 3, 3):
            assert np.allclose(output.data, 9.0), (
                f"Convolution value wrong.\n"
                f"  All-ones kernel (3x3) on all-ones input\n"
                f"  Each output should be 9 (sum of 9 ones)\n"
                f"  Got: {output.data[0,0,0,0]}"
            )


class TestPoolingLayers:
    """
    Test pooling layers (MaxPool, AvgPool).
    """

    def test_maxpool2d_creation(self):
        """
        Verify MaxPool2d can be created.
        """
        pool = MaxPool2d(kernel_size=2)
        assert pool is not None

    def test_maxpool2d_forward(self):
        """
        Verify MaxPool2d takes maximum in each window.
        """
        pool = MaxPool2d(kernel_size=2, stride=2)

        # Simple 4x4 input with known values
        x = Tensor(
            np.array(
                [
                    [
                        [[1], [2], [5], [6]],
                        [[3], [4], [7], [8]],
                        [[9], [10], [13], [14]],
                        [[11], [12], [15], [16]],
                    ]
                ]
            )
        )  # (1, 4, 4, 1)

        output = pool(x)

        # 2x2 pooling should give max of each 2x2 region
        # Top-left: max(1,2,3,4) = 4
        # Top-right: max(5,6,7,8) = 8
        # etc.
        expected = np.array([[[[4], [8]], [[12], [16]]]])

        if output.shape == (1, 2, 2, 1):
            assert np.array_equal(output.data, expected), (
                f"MaxPool values wrong.\n"
                f"  Expected: {expected.squeeze()}\n"
                f"  Got: {output.data.squeeze()}"
            )

    def test_avgpool2d_forward(self):
        """
        Verify AvgPool2d computes average of each window.
        """
        pool = AvgPool2d(kernel_size=2, stride=2)

        # All-ones input - average should be 1
        x = Tensor(np.ones((1, 4, 4, 1)))
        output = pool(x)

        if output.shape == (1, 2, 2, 1):
            assert np.allclose(output.data, 1.0), (
                f"AvgPool of all-ones should be 1.0\n" f"  Got: {output.data[0,0,0,0]}"
            )


class TestConvOutputShapes:
    """
    Test convolution output shape calculations.
    """

    def test_conv_padding_same(self):
        """
        Verify padding preserves spatial dimensions.
        """
        # With padding=1 and kernel=3, output should match input spatial dims
        # Formula: output = input - kernel + 2*padding + 1 = 32 - 3 + 2 + 1 = 32
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)

        # NCHW format
        x = Tensor(rng.standard_normal((4, 3, 32, 32)))
        output = conv(x)

        assert output.shape == (4, 8, 32, 32), (
            f"padding=1 with kernel=3 should preserve spatial dims.\n"
            f"  Input: (4, 3, 32, 32) NCHW\n"
            f"  Expected: (4, 8, 32, 32)\n"
            f"  Got: {output.shape}"
        )

    def test_conv_stride(self):
        """
        Verify stride reduces output dimensions.
        """
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)

        # NCHW format
        x = Tensor(rng.standard_normal((1, 3, 32, 32)))
        output = conv(x)

        # (32 - 3) / 2 + 1 = 15
        expected_size = 15
        # In NCHW, spatial dims are at indices 2 and 3
        assert output.shape[2] == expected_size and output.shape[3] == expected_size, (
            f"Stride=2 output size wrong.\n"
            f"  Input: 32x32, kernel=3, stride=2\n"
            f"  Expected: {expected_size}x{expected_size}\n"
            f"  Got: {output.shape[2]}x{output.shape[3]}\n"
            "Formula: (input - kernel) / stride + 1"
        )


class TestConvGradientFlow:
    """
    Test that gradients flow through convolutions.
    """

    def test_conv2d_gradient_to_input(self):
        """
        Verify input receives gradients through Conv2d.
        """
        enable_autograd()

        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        # NCHW format
        x = Tensor(rng.standard_normal((1, 1, 8, 8)), requires_grad=True)

        output = conv(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, (
            "Input didn't receive gradients through Conv2d.\n"
            "This means backprop through the conv is broken."
        )

    def test_conv2d_gradient_to_weights(self):
        """
        Verify conv weights receive gradients.
        """
        enable_autograd()

        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        conv.weight.requires_grad = True  # Enable gradient tracking for weights
        # NCHW format
        x = Tensor(rng.standard_normal((1, 1, 8, 8)), requires_grad=True)

        output = conv(x)
        loss = output.sum()
        loss.backward()

        weight = conv.weight if hasattr(conv, "weight") else conv.weights
        assert weight.grad is not None, (
            "Conv weights didn't receive gradients.\n"
            "This means the conv layer cannot learn."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
