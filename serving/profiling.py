__all__ = [
    "rng",
    "BYTES_PER_FLOAT32",
    "KB_TO_BYTES",
    "MB_TO_BYTES",
    "Profiler",
    "quick_profile",
    "analyze_weight_distribution",
]

import sys
import os
import time
import numpy as np
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict
import gc

from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.layers import Linear
from tinytorch.architecture.convolutions import Conv2d

BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024


class Profiler:
    """
    Dynamic, hardware-aware ML model profiler supporting layer-wise shape tracing,
    true backward pass measurement, and modern Transformer layer metrics.
    """

    def __init__(self, peak_gflops: float = 100.0, peak_memory_bw_gbs: float = 50.0):
        """
        Initialize profiler with configurable hardware ceilings for roofline analysis.
        """
        self.measurements = {}
        self.operation_counts = defaultdict(int)
        self.peak_gflops = peak_gflops
        self.peak_memory_bw_gbs = peak_memory_bw_gbs

    def __enter__(self):
        self._context_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self._context_start) * 1000

    def count_parameters(self, model) -> Tuple[int, int]:
        """
        Counts total and trainable parameters across model structures.
        """
        params = []
        if hasattr(model, "parameters") and callable(model.parameters):
            params = list(model.parameters())
        elif hasattr(model, "layers"):
            params = [
                p
                for layer in model.layers
                if hasattr(layer, "parameters")
                for p in layer.parameters()
            ]
        elif hasattr(model, "weight"):
            params = [model.weight]
            if hasattr(model, "bias") and model.bias is not None:
                params.append(model.bias)

        total_params = sum(
            p.data.size if hasattr(p, "data") else getattr(p, "size", 0) for p in params
        )
        trainable_params = sum(
            p.data.size if hasattr(p, "data") else getattr(p, "size", 0)
            for p in params
            if getattr(p, "requires_grad", True)
        )

        return total_params, trainable_params

    def count_layer_flops(
        self, layer: Any, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]
    ) -> int:
        """
        Computes dynamic FLOPs given layer type, input shape, and output shape.
        """
        layer_name = layer.__class__.__name__

        if layer_name == "Linear":
            in_features = input_shape[-1]
            out_features = output_shape[-1]
            batch_elements = int(np.prod(input_shape[:-1]))
            return batch_elements * in_features * out_features * 2

        elif layer_name == "Conv2d":
            if not hasattr(layer, "kernel_size"):
                return 0
            batch_size = input_shape[0]
            out_channels, out_h, out_w = (
                output_shape[1],
                output_shape[2],
                output_shape[3],
            )
            in_channels = layer.in_channels
            kh = kw = (
                layer.kernel_size
                if isinstance(layer.kernel_size, int)
                else layer.kernel_size[0]
            )
            return (
                batch_size
                * (out_h * out_w)
                * out_channels
                * (2 * kh * kw * in_channels)
            )

        elif "Attention" in layer_name or layer_name in [
            "MultiHeadAttention",
            "SelfAttention",
        ]:
            batch_size, seq_len, embed_dim = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
            )
            qkv_proj_flops = 3 * (batch_size * 2 * seq_len * (embed_dim**2))
            attn_score_flops = batch_size * 2 * (seq_len**2) * embed_dim
            softmax_flops = batch_size * seq_len * seq_len * 5
            attn_out_flops = batch_size * 2 * (seq_len**2) * embed_dim
            out_proj_flops = batch_size * 2 * seq_len * (embed_dim**2)
            return (
                qkv_proj_flops
                + attn_score_flops
                + softmax_flops
                + attn_out_flops
                + out_proj_flops
            )

        elif layer_name in ["LayerNorm", "BatchNorm2d"]:
            return int(np.prod(input_shape)) * 5

        elif layer_name in ["ReLU", "GELU", "SiLU", "Sigmoid", "Tanh"]:
            ops = 8 if layer_name in ["GELU", "SiLU", "Sigmoid", "Tanh"] else 1
            return int(np.prod(input_shape)) * ops

        return int(np.prod(input_shape))

    def profile_forward_pass(self, model: Any, input_tensor: Tensor) -> Dict[str, Any]:
        """
        Dynamic forward pass execution profiling with layer-wise shape propagation.
        """
        layer_profiles = []
        total_flops = 0

        # Dynamic layer trace
        def trace_layer(layer: Any, curr_tensor: Tensor) -> Tensor:
            nonlocal total_flops
            inp_shape = curr_tensor.shape
            try:
                out_tensor = layer.forward(curr_tensor)
                out_shape = out_tensor.shape
            except Exception:
                print(f"Could not infer shape for {layer}: {e}")
                out_tensor = curr_tensor
                out_shape = inp_shape

            flops = self.count_layer_flops(layer, inp_shape, out_shape)
            params, _ = self.count_parameters(layer)

            layer_profiles.append(
                {
                    "layer_name": layer.__class__.__name__,
                    "input_shape": inp_shape,
                    "output_shape": out_shape,
                    "parameters": params,
                    "flops": flops,
                }
            )
            total_flops += flops
            return out_tensor

        # Trace layers sequentially or as a single block
        curr = input_tensor
        if hasattr(model, "layers"):
            for layer in model.layers:
                curr = trace_layer(layer, curr)
        else:
            trace_layer(model, curr)

        # Timed execution & peak dynamic memory trace
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        start_time = time.perf_counter()
        fwd_output = model.forward(input_tensor)
        end_time = time.perf_counter()

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latency_ms = (end_time - start_time) * 1000.0
        peak_memory_mb = max((peak_mem - start_mem) / MB_TO_BYTES, 0.001)

        total_params, trainable_params = self.count_parameters(model)
        param_memory_mb = (total_params * BYTES_PER_FLOAT32) / MB_TO_BYTES
        activation_memory_mb = (
            input_tensor.data.nbytes + fwd_output.data.nbytes
        ) / MB_TO_BYTES

        derived = self._compute_derived_metrics(total_flops, latency_ms, peak_memory_mb)
        bottleneck = self._analyze_bottleneck(
            derived["gflops_per_second"], derived["memory_bandwidth_mbs"]
        )

        return {
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "flops": total_flops,
            "latency_ms": latency_ms,
            "parameter_memory_mb": param_memory_mb,
            "activation_memory_mb": activation_memory_mb,
            "peak_memory_mb": max(
                peak_memory_mb, param_memory_mb + activation_memory_mb
            ),
            "layer_profiles": layer_profiles,
            **derived,
            **bottleneck,
        }

    def profile_backward_pass(
        self, model: Any, input_tensor: Tensor, loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Executes actual backpropagation to measure backward latency and memory.
        """
        fwd_profile = self.profile_forward_pass(model, input_tensor)

        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        fwd_output = model.forward(input_tensor)
        loss = loss_fn(fwd_output) if loss_fn is not None else fwd_output.sum()

        bwd_start = time.perf_counter()
        if hasattr(loss, "backward") and callable(loss.backward):
            loss.backward()
        bwd_end = time.perf_counter()

        _, bwd_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        bwd_latency_ms = (bwd_end - bwd_start) * 1000.0
        bwd_flops = fwd_profile["flops"] * 2
        total_flops = fwd_profile["flops"] + bwd_flops
        total_latency_ms = fwd_profile["latency_ms"] + bwd_latency_ms
        gradient_memory_mb = fwd_profile["parameter_memory_mb"]
        peak_bwd_memory_mb = (bwd_peak_mem - start_mem) / MB_TO_BYTES

        return {
            "forward_flops": fwd_profile["flops"],
            "forward_latency_ms": fwd_profile["latency_ms"],
            "backward_flops": bwd_flops,
            "backward_latency_ms": bwd_latency_ms,
            "gradient_memory_mb": gradient_memory_mb,
            "total_flops": total_flops,
            "total_latency_ms": total_latency_ms,
            "total_memory_mb": fwd_profile["parameter_memory_mb"]
            + fwd_profile["activation_memory_mb"]
            + gradient_memory_mb,
            "peak_backward_memory_mb": max(peak_bwd_memory_mb, gradient_memory_mb),
            "total_gflops_per_second": (total_flops / 1e9)
            / max(total_latency_ms / 1000.0, 1e-6),
            "optimizer_memory_estimates": {
                "sgd": 0.0,
                "adam": gradient_memory_mb * 2.0,
                "adamw": gradient_memory_mb * 2.0,
            },
            "bottleneck": fwd_profile["bottleneck"],
        }

    def measure_latency(
        self, model: Any, input_tensor: Tensor, warmup: int = 10, iterations: int = 100
    ) -> float:
        """
        Measures robust median execution latency over warmup and timed runs.
        """
        for _ in range(warmup):
            _ = model.forward(input_tensor)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model.forward(input_tensor)
            times.append((time.perf_counter() - start) * 1000.0)

        return float(np.median(times))

    def _compute_derived_metrics(
        self, flops: int, latency_ms: float, peak_memory_mb: float
    ) -> Dict[str, float]:
        latency_sec = max(latency_ms / 1000.0, 1e-6)
        gflops_per_second = (flops / 1e9) / latency_sec
        memory_bandwidth_mbs = peak_memory_mb / latency_sec

        return {
            "gflops_per_second": gflops_per_second,
            "memory_bandwidth_mbs": memory_bandwidth_mbs,
            "computational_efficiency": min(gflops_per_second / self.peak_gflops, 1.0),
        }

    def _analyze_bottleneck(
        self, gflops_per_second: float, memory_bandwidth_mbs: float
    ) -> Dict[str, Any]:
        memory_bw_gbs = memory_bandwidth_mbs / 1024.0
        arithmetic_intensity = gflops_per_second / max(memory_bw_gbs, 1e-6)
        knee_point = self.peak_gflops / max(self.peak_memory_bw_gbs, 1e-6)
        is_memory_bound = arithmetic_intensity < knee_point

        return {
            "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
            "is_memory_bound": is_memory_bound,
            "is_compute_bound": not is_memory_bound,
            "bottleneck": "memory" if is_memory_bound else "compute",
        }


def quick_profile(
    model: Any, input_tensor: Tensor, profiler: Optional[Profiler] = None
):
    if profiler is None:
        profiler = Profiler()

    profile = profiler.profile_forward_pass(model, input_tensor)

    print("🧪 Quick Profile Results:")
    print(
        f"   Parameters: {profile['parameters']:,} (Trainable: {profile['trainable_parameters']:,})"
    )
    print(f"   FLOPs: {profile['flops']:,}")
    print(f"   Latency: {profile['latency_ms']:.2f} ms")
    print(f"   Peak Memory: {profile['peak_memory_mb']:.2f} MB")
    print(
        f"   Arithmetic Intensity: {profile['arithmetic_intensity_flops_per_byte']:.2f} FLOP/Byte"
    )
    print(f"   Bottleneck: {profile['bottleneck']}")
    print(f"   Efficiency: {profile['computational_efficiency']*100:.1f}%")

    return profile


def analyze_weight_distribution(
    model: Any, percentiles: List[int] = [10, 25, 50, 75, 90]
):
    weights = []
    if hasattr(model, "parameters") and callable(model.parameters):
        for param in model.parameters():
            weights.extend(param.data.flatten().tolist())
    elif hasattr(model, "weight"):
        weights.extend(model.weight.data.flatten().tolist())
    else:
        return {"error": "No weights found"}

    weights = np.array(weights)
    abs_weights = np.abs(weights)

    stats = {
        "total_weights": len(weights),
        "mean": float(np.mean(abs_weights)),
        "std": float(np.std(abs_weights)),
        "min": float(np.min(abs_weights)),
        "max": float(np.max(abs_weights)),
    }

    for p in percentiles:
        stats[f"percentile_{p}"] = float(np.percentile(abs_weights, p))

    for threshold in [0.001, 0.01, 0.1]:
        below = np.sum(abs_weights < threshold) / len(weights) * 100
        stats[f'below_threshold_{str(threshold).replace(".", "")}'] = float(below)

    return stats
