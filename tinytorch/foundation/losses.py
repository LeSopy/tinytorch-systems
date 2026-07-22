__all__ = [
    "rng",
    "EPSILON",
    "log_softmax",
    "MSELoss",
    "CrossEntropyLoss",
    "BinaryCrossEntropyLoss",
]


import numpy as np

rng = np.random.default_rng(7)
from typing import Optional

# Import from my-tinyTorch package (previous modules must be completed and exported)
from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.activations import ReLU
from tinytorch.foundation.layers import Linear

# Constants for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log-softmax with numerical stability."""

    # Step 1: Find max along dimension for numerical stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)

    # Step 2: Subtract max to prevent overflow
    shifted = x.data - max_vals

    # Step 3: Compute log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # Step 4: Return log_softmax = input - max - log_sum_exp
    result = x.data - max_vals - log_sum_exp

    return Tensor(result)


class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        """Initialize MSE loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean squared error between predictions and targets.
        """

        # Step 1: Compute element-wise difference
        diff = predictions.data - targets.data

        # Step 2: Square the differences
        squared_diff = diff**2

        # Step 3: Take mean across all elements
        mse = np.mean(squared_diff)

        return Tensor(mse)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (placeholder — gradient computation is separate).
        """
        pass


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def __init__(self):
        """Initialize cross-entropy loss function."""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss between logits and target class indices.
        """

        # Step 1: Compute log-softmax for numerical stability
        log_probs = log_softmax(logits, dim=-1)

        # Step 2: Select log-probabilities for correct classes
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        # Select correct class log-probabilities using advanced indexing
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        # Step 3: Return negative mean (cross-entropy is negative log-likelihood)
        cross_entropy = -np.mean(selected_log_probs)

        return Tensor(cross_entropy)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(logits, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (placeholder — gradient computation is separate).
        """
        pass


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize binary cross-entropy loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute binary cross-entropy loss.
        """

        # Step 1: Clamp predictions to avoid numerical issues with log(0) and log(1)
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        # Step 2: Compute binary cross-entropy
        # BCE = -(targets * log(preds) + (1-targets) * log(1-preds))
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)

        bce_per_sample = -(
            targets.data * log_preds + (1 - targets.data) * log_one_minus_preds
        )

        # Step 3: Return mean across all samples
        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (placeholder — gradient computation is separate).
        """
        pass
