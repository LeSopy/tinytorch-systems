
# Export 
__all__ = ['rng', 'TOLERANCE', 'Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax']

import numpy as np
rng = np.random.default_rng(7)
from typing import Optional

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tinytorch.foundation.tensor import Tensor

# Constants for numerical comparisons
TOLERANCE = 1e-10  # Small tolerance for floating-point comparisons in tests

# Export only activation classes
__all__ = ['Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax']

class Sigmoid:
    """
    Sigmoid activation: σ(x) = 1/(1 + e^(-x))
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply sigmoid activation element-wise.
        """
        
        x_data = x.data
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.where(
                x_data >= 0,
                1.0 / (1.0 + np.exp(-x_data)),
                np.exp(x_data) / (1.0 + np.exp(x_data)),
            )
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

class ReLU:
    """
    ReLU activation: f(x) = max(0, x)
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU activation element-wise.
        """
        # Apply ReLU: max(0, x)
        result = np.maximum(0, x.data)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

class Tanh:
    """
    Tanh activation: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply tanh activation element-wise.
        """
        # Apply tanh using NumPy
        result = np.tanh(x.data)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

class GELU:
    """
    GELU activation: f(x) = x * Φ(x) ≈ x * Sigmoid(1.702 * x)
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply GELU activation element-wise.
        """
        return Sigmoid()(x * 1.702) * x

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

class Softmax:
    """
    Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Apply softmax activation along specified dimension.
        """
        # Numerical stability: subtract max to prevent overflow
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max

        # Compute exponentials
        exp_values = np.exp(x_shifted)

        # Sum along dimension
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return Tensor(result)

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x, dim)
