__all__ = [
    "rng",
    "INT8_MIN_VALUE",
    "INT8_MAX_VALUE",
    "INT8_RANGE",
    "EPSILON",
    "BYTES_PER_FLOAT",
    "BYTES_PER_INT8",
    "MB_TO_BYTES",
    "quantize_int8",
    "dequantize_int8",
    "QuantizedLinear",
    "quantize_model",
    "Quantizer",
]

import numpy as np
from typing import Tuple, Dict, List, Optional

rng = np.random.default_rng(7)

# Import dependancies from other modules
from ..foundation.tensor import Tensor
from ..foundation.layers import Linear, Sequential
from ..foundation.activations import ReLU

# Constants for INT8 quantization
INT8_MIN_VALUE = -128
INT8_MAX_VALUE = 127
INT8_RANGE = 256  # Number of possible INT* values (-128 to 127 inclusive)
EPSILON = 1e-8  # Small value for numrical stability (constant tensor detection)

# Constants for memory calculations
BYTES_PER_FLOAT = 4  # Standard float32 size in bytes
BYTES_PER_INT8 = 1  # INT8 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

if __name__ == "__main__":
    print("Quantization module imports complete")


def quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, int]:
    """
    Quantize FP32 tensor to INT8 using asymmetric (min-max) quantization.
    """

    data_tensor = tensor.data

    # Step1 : Find dynamic range
    min_val = float(np.min(data_tensor))
    max_val = float(np.max(data_tensor))

    # Step 2: Handle edge case (constant tensor)
    # All elements have the same value c, so there is no range to map. We encode
    # every element as q=0 and choose zero_point/scale so dequantization recovers
    # the constant via (0 - zero_point) * scale = c.
    if abs(max_val - min_val < EPSILON):
        c = min_val
        if abs(c) <= INT8_MAX_VALUE:
            # |c| fits the INT8 range: scale=1.0; zero_point=-round(c)
            scale = 1.0
            zero_point = int(np.round(-c))
        else:
            # |c| exceeds the INT8 range. Keeping scale=1.0 would force
            # zero_point = -c, which np.clip would saturate to +-128 and
            # silently corrupt the value. Use zero_point = +-1 and scale = |c|
            # instead, so (0 - zero_point) * scale = c holds without clamping.
            zero_point = -1 if c > 0 else 1
            scale = abs(c)
        quantized_data = np.zeros_like(data_tensor, dtype=np.int8)
        return Tensor(quantized_data), scale, zero_point

    # Step 3 : Calculate scale and zero_point for standard quantization
    # Map [min_val, max_val] to [INT8_MIN_VALUE, INT8_MAX_VALUE]
    scale = (max_val - min_val) / (INT8_RANGE - 1)
    zero_point = int(np.round(INT8_MIN_VALUE - (min_val / scale)))

    # Clamp zero_point to INT8 range
    zero_point = int(np.clip(zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE))

    # Step 4 : Apply quantization formula: q = (x / scale) + zero_point
    quantized_data = np.round(data_tensor / scale + zero_point)

    # Clamp to INT8 range and convert to int8
    quantized_data = (np.clip(quantized_data, INT8_MIN_VALUE, INT8_MAX_VALUE)).astype(
        np.int8
    )

    return Tensor(quantized_data), scale, zero_point


def dequantize_int8(q_tensor: Tensor, scale: float, zero_point: int) -> Tensor:
    """
    Dequantize INT8 tensor back to FP32.
    """

    dequantized_data = (q_tensor.data.astype(np.float32) - zero_point) * scale
    return Tensor(dequantized_data)


class QuantizedLinear:
    """Quantized version of Linear layer using INT8 arithmetic."""

    def __init__(self, linear_layer: Linear):
        """Create quantized version of existing linear layer."""
        self.original_layer = linear_layer

        # Quantized weights
        self.q_weight, self.weight_scale, self.weight_zero_point = quantize_int8(
            linear_layer.weight
        )

        # Quantize bias if it exists
        if linear_layer.bias is not None:
            self.q_bias, self.bias_scale, self.bias_zero_point = quantize_int8(
                linear_layer.bias
            )
        else:
            self.q_bias = None
            self.bias_scale = None
            self.bias_zero_point = None

        # Store input quantization parameters (set during calibration)
        self.input_scale = None
        self.input_zero_point = None

    def calibrate(self, sample_inputs: List[Tensor]):
        """Calibrate input quantization parameters using sample data"""

        all_values = []
        for inp in sample_inputs:
            all_values.extend(inp.data.flatten())

        all_values = np.array(all_values)

        # Calculate input quantization parameters
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))

        if max_val - min_val < EPSILON:
            self.input_scale = 1
            self.input_zero_point = 0
        else:
            self.input_scale = (max_val - min_val) / (INT8_RANGE - 1)
            self.input_zero_point = int(
                np.round(INT8_MIN_VALUE - (min_val / self.input_scale))
            )
            self.input_zero_point = np.clip(
                self.input_zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized computation"""

        # 1. Handle uncalibrated input: use static params if calibrated, else dynamically quantize
        if self.input_scale is None or self.input_zero_point is None:
            q_x, input_scale, input_zero_point = quantize_int8(x)
            q_x_data = q_x.data
        else:
            input_scale = self.input_scale
            input_zero_point = self.input_zero_point
            # Quantize x using saved static calibration parameters
            scaled_x = x.data / input_scale + input_zero_point
            q_x_data = np.clip(
                np.round(scaled_x), INT8_MIN_VALUE, INT8_MAX_VALUE
            ).astype(np.int8)

        # 2. Subtract zero points and cast to int32 to avoid overflow during accumulation
        x_int32 = q_x_data.astype(np.int32) - input_zero_point
        w_int32 = self.q_weight.data.astype(np.int32) - self.weight_zero_point

        # 3. Perform integer matrix multiplication (INT8 x INT8 -> INT32)
        int32_gemm = np.matmul(x_int32, w_int32)

        # 4. Rescale INT32 accumulated output back to FP32
        effective_scale = input_scale * self.weight_scale
        output_fp32 = int32_gemm.astype(np.float32) * effective_scale

        # 5. Add bias if present
        if self.q_bias is not None:
            bias_fp32 = dequantize_int8(
                self.q_bias, self.bias_scale, self.bias_zero_point
            )
            output_fp32 = output_fp32 + bias_fp32.data

        return Tensor(output_fp32)

    def __call__(self, x: Tensor):
        """Allows the quantized linear layer to be called like a function."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return quantized parameters."""
        params = [self.q_weight]
        if self.q_bias is not None:
            params.append(self.q_bias)
        return params

    def memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage in bytes."""

        # Original FP32 usage
        original_weight_bytes = self.original_layer.weight.data.size * BYTES_PER_FLOAT
        original_bias_bytes = 0
        if self.original_layer.bias is not None:
            original_bias_bytes = self.original_layer.bias.data.size * BYTES_PER_FLOAT

        # Quantized INT8 usage
        quantized_weight_bytes = self.q_weight.data.size * BYTES_PER_INT8
        quantized_bias_bytes = 0
        if self.q_bias is not None:
            quantized_bias_bytes = self.q_bias.data.size * BYTES_PER_INT8

        # Add overhead for scales and zero points (small)
        # 2 floats: one scale for weights, one scale for bias (if present)
        overhead_bytes = BYTES_PER_FLOAT * 2

        quantized_total = quantized_weight_bytes + quantized_bias_bytes + overhead_bytes
        original_total = original_weight_bytes + original_bias_bytes

        return {
            "original_bytes": original_total,
            "quantized_bytes": quantized_total,
            "compression_ratio": (
                original_total / quantized_total if quantized_total > 0 else 1.0
            ),
        }


def _collect_layer_inputs(
    model, layer_index: int, calibration_data: List[Tensor], max_samples: int = 10
) -> List[Tensor]:
    """
    Forward calibration data through preceding layers to collect inputs for a specific layer.- Use model.layers[j].forward(x) to pass through each preceding layer
    """

    sample_inputs = []
    for data in calibration_data[:max_samples]:
        x = data
        for j in range(layer_index):
            x = model.layers[j].forward(x)
        sample_inputs.append(x)
    return sample_inputs


def _quantize_single_layer(
    layer: Linear, calibration_inputs: Optional[List[Tensor]] = None
) -> QuantizedLinear:
    """
    Quantize a single Linear layer and optionally calibrate it.
    """

    quantized_layer = QuantizedLinear(layer)

    if calibration_inputs is not None:
        quantized_layer.calibrate(calibration_inputs)

    return quantized_layer


def quantize_model(model, calibration_data: Optional[List[Tensor]] = None) -> None:
    """
    Quantize all Linear layers in a model in-place.
    """

    if hasattr(model, "layers"):
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                # Collect calibration inputs if data provided
                cal_inputs = None
                if calibration_data is not None:
                    cal_inputs = _collect_layer_inputs(model, i, calibration_data)

                # Replace with quantized version
                model.layers[i] = _quantize_single_layer(layer, cal_inputs)

    elif isinstance(model, Linear):
        raise ValueError(
            f"Cannot quantize single Linear layer in-place\n"
            f"  ❌ quantize_model() modifies models in-place, but a single layer has no container to modify\n"
            f"  💡 In-place modification requires a container (like Sequential) that holds layer references\n"
            f"  🔧 Use QuantizedLinear directly: quantized_layer = QuantizedLinear(your_linear_layer)"
        )

    else:
        raise ValueError(
            f"Unsupported model type for quantization: {type(model).__name__}\n"
            f"  ❌ quantize_model() expects a model with .layers attribute (like Sequential)\n"
            f"  💡 The function iterates through model.layers to find and replace Linear layers\n"
            f"  🔧 Wrap your layers in Sequential: model = Sequential(layer1, activation, layer2)"
        )


class Quantizer:
    """
    Complete quantization system for milestone use.
    """

    @staticmethod
    def quantize_tensor(tensor: Tensor) -> Tuple[Tensor, float, int]:
        """Quantize FP32 tensor to INT8. Delegates to quantize_int8()."""
        return quantize_int8(tensor)

    @staticmethod
    def dequantize_tensor(q_tensor: Tensor, scale: float, zero_point: int) -> Tensor:
        """Dequantize INT8 tensor back to FP32. Delegates to dequantize_int8()."""
        return dequantize_int8(q_tensor, scale, zero_point)

    @staticmethod
    def quantize_model(
        model, calibration_data: Optional[List[Tensor]] = None
    ) -> Dict[str, any]:
        """
        Quantize all Linear layers in a model and return stats.
        """
        quantized_layers = {}
        original_size = 0
        total_elements = 0
        param_idx = 0

        # Iterate through model parameters
        for layer in model.layers:
            for param in layer.parameters():
                param_size = param.data.nbytes
                original_size += param_size
                total_elements += param.data.size

                # Quantize parameter using the standalone function
                q_param, scale, zp = quantize_int8(param)

                quantized_layers[f"param_{param_idx}"] = {
                    "quantized": q_param,
                    "scale": scale,
                    "zero_point": zp,
                    "original_shape": param.data.shape,
                }
                param_idx += 1

        # INT8 uses 1 byte per element
        quantized_size = total_elements

        return {
            "quantized_layers": quantized_layers,
            "original_size_mb": original_size / MB_TO_BYTES,
            "quantized_size_mb": quantized_size / MB_TO_BYTES,
            "compression_ratio": (
                original_size / quantized_size if quantized_size > 0 else 1.0
            ),
        }

    @staticmethod
    def compare_models(original_model, quantized_info: Dict) -> Dict[str, float]:
        """Compare memory usage between original and quantized models."""
        return {
            "original_mb": quantized_info["original_size_mb"],
            "quantized_mb": quantized_info["quantized_size_mb"],
            "compression_ratio": quantized_info["compression_ratio"],
            "memory_saved_mb": quantized_info["original_size_mb"]
            - quantized_info["quantized_size_mb"],
        }
