__all__ = [
    "rng",
    "BYTES_PER_FLOAT32",
    "MB_TO_BYTES",
    "measure_sparsity",
    "magnitude_prune",
    "structured_prune",
    "low_rank_approximate",
    "KnowledgeDistillation",
    "Compressor",
    "verify_pruning_works",
]

import numpy as np

rng = np.random.default_rng(7)
import copy
import time
from typing import Dict, Tuple, List, Any, Optional

from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.layers import Linear, Sequential
from tinytorch.foundation.activations import ReLU

BYTES_PER_FLOAT32 = 4
MB_TO_BYTES = 1024 * 1024


def measure_sparsity(model) -> float:
    """Calculate the percentage of zero in a model"""

    total_params = 0
    zero_params = 0
    for param in model.parameters():
        if len(param.shape) > 1:
            total_params += param.size
            zero_params += np.sum(param.data == 0)

    if total_params == 0.0:
        return 0.0

    return (zero_params / total_params) * 100.0


def magnitude_prune(model, sparsity=0.9):
    """Remove weights with smaller magnitudes to achieve target sparsity."""

    # Collect all weights (excluding biases)
    all_weights = []
    weight_params = []

    for param in model.parameters():
        # Skip biases (typically 1D)
        if len(param.shape) > 1:
            all_weights.extend(param.data.flatten())
            weight_params.append(param)

    if not all_weights:
        return model

    # Calculate magnitude threshold
    magnitudes = np.abs(all_weights)
    threshold = np.percentile(magnitudes, sparsity * 100)

    # Apply pruning to each weight parameter
    for param in weight_params:
        mask = np.abs(param.data) >= threshold
        param.data = param.data * mask

    return model


def structured_prune(model, prune_ratio=0.5):
    """Remove entire channels/neurons based on L2 norm importance."""

    # All Linear layers have .weight attribute
    for layer in model.layers:
        if isinstance(layer, Linear):
            weight = layer.weight.data

            # Calculate L2 norm for each output channel (column)
            channel_norms = np.linalg.norm(weight, axis=0)

            # Find channels to prune (lowest importance)
            num_channels = weight.shape[1]
            num_to_prune = int(num_channels * prune_ratio)

            if num_to_prune > 0:
                # Get indices of channels to prune (smallest norms)
                prune_indices = np.argpartition(channel_norms, num_to_prune)[
                    :num_to_prune
                ]

                # Zero out entire channels
                weight[:, prune_indices] = 0

                # Also zero corresponding bias elements if bias exists
                if layer.bias is not None:
                    layer.bias.data[prune_indices] = 0

    return model


def low_rank_approximate(weight_matrix, rank_ratio=0.5):
    """Approximate weight matrix using low-rank decomposition (SVD)."""

    m, n = weight_matrix.shape

    # Perform SVD
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)

    # Determine target rank
    max_rank = min(m, n)
    target_rank = max(1, int(rank_ratio * max_rank))

    # Truncate to target rank
    U_truncated = U[:, :target_rank]
    S_truncated = S[:target_rank]
    V_truncated = V[:target_rank, :]

    return U_truncated, S_truncated, V_truncated


class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.

    Train a smaller student model to mimic a larger teacher model.
    """

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        """
        Initialize knowledge distillation.
        """

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Calculate combined distillation loss."""

        # Extract numpy arrays from Tensors
        # student_logits and teacher_logits are always Tensors from forward passes
        student_logits = student_logits.data
        teacher_logits = teacher_logits.data

        # true_labels might be numpy array or Tensor
        if isinstance(true_labels, Tensor):
            true_labels = true_labels.data

        # Soften distributions with temperature
        student_soft = self._softmax(student_logits / self.temperature)
        teacher_soft = self._softmax(teacher_logits / self.temperature)

        # Soft target loss (KL divergence)
        soft_loss = self._kl_divergence(student_soft, teacher_soft)

        # Hard target loss (cross-entropy)
        student_hard = self._softmax(student_logits)
        hard_loss = self._cross_entropy(student_hard, true_labels)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss

    def _softmax(self, logits):
        """Compute softmax with numerical stability."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _kl_divergence(self, p, q):
        """Compute KL divergence between distributions."""
        return np.sum(p * np.log(p / (q + 1e-8) + 1e-8))

    def _cross_entropy(self, predictions, labels):
        """Compute cross-entropy loss."""
        # Simple implementation for integer labels
        if labels.ndim == 1:
            return -np.mean(np.log(predictions[np.arange(len(labels)), labels] + 1e-8))
        else:
            return -np.mean(np.sum(labels * np.log(predictions + 1e-8), axis=1))


class Compressor:
    """
    Complete compression system for milestone use.
    """

    @staticmethod
    def measure_sparsity(model) -> float:
        """Measure the sparsity of a model (returns fraction 0-1)."""
        # Delegate to standalone function and convert percentage to fraction
        return measure_sparsity(model) / 100.0

    @staticmethod
    def magnitude_prune(model, sparsity=0.5):
        """Prune model weights by magnitude. Delegates to standalone function."""
        return magnitude_prune(model, sparsity)

    @staticmethod
    def structured_prune(model, prune_ratio=0.5):
        """Prune entire neurons/channels. Delegates to standalone function."""
        return structured_prune(model, prune_ratio)

    @staticmethod
    def compress_model(model, compression_config: Dict[str, Any]):
        """
        Apply complete compression pipeline to a model.
        """
        stats = {"original_sparsity": Compressor.measure_sparsity(model)}

        # Apply magnitude pruning
        if "magnitude_sparsity" in compression_config:
            model = Compressor.magnitude_prune(
                model, compression_config["magnitude_sparsity"]
            )

        # Apply structured pruning
        if "structured_prune_ratio" in compression_config:
            model = Compressor.structured_prune(
                model, compression_config["structured_prune_ratio"]
            )

        stats["final_sparsity"] = Compressor.measure_sparsity(model)
        stats["compression_ratio"] = (
            1.0 / (1.0 - stats["final_sparsity"])
            if stats["final_sparsity"] < 1.0
            else float("inf")
        )

        return model, stats


def verify_pruning_works(model, target_sparsity=0.8):
    """
    Verify pruning actually creates zeros using real zero counting.
    """
    print(" Verifying pruning sparsity with actual zero counting...")

    # Count actual zeros in model parameters
    zeros = sum(np.sum(p.data == 0) for p in model.parameters())
    total = sum(p.data.size for p in model.parameters())
    sparsity = zeros / total
    memory_bytes = sum(p.data.nbytes for p in model.parameters())

    # Display results
    print(f"   Total parameters: {total:,}")
    print(f"   Zero parameters: {zeros:,}")
    print(f"   Active parameters: {total - zeros:,}")
    print(f"   Sparsity achieved: {sparsity*100:.1f}%")
    print(
        f"   Memory footprint: {memory_bytes / MB_TO_BYTES:.2f} MB (unchanged with dense storage)"
    )

    # Verify target met (allow 15% tolerance for structured pruning variations)
    verified = abs(sparsity - target_sparsity) < 0.15
    status = "✓" if verified else "✗"
    print(f"   {status} Meets {target_sparsity*100:.0f}% sparsity target")

    assert verified, f"Sparsity target not met: {sparsity:.2f} vs {target_sparsity:.2f}"

    print(f"\n VERIFIED: {sparsity*100:.1f}% sparsity achieved")
    print(f" Memory saved: 0 MB (dense numpy arrays)")
    print(
        f" LEARNING: Compute savings ~{sparsity*100:.1f}% (skip zero multiplications)"
    )
    print(
        f"   In production: Use sparse formats (scipy.sparse.csr_matrix) for memory savings"
    )

    return {
        "sparsity": sparsity,
        "zeros": zeros,
        "total": total,
        "active": total - zeros,
        "memory_mb": memory_bytes / MB_TO_BYTES,
        "verified": verified,
    }
