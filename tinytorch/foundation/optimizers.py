__all__ = [
    "rng",
    "DEFAULT_LEARNING_RATE_SGD",
    "DEFAULT_LEARNING_RATE_ADAM",
    "DEFAULT_MOMENTUM",
    "DEFAULT_BETA1",
    "DEFAULT_BETA2",
    "DEFAULT_EPS",
    "DEFAULT_WEIGHT_DECAY_ADAMW",
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
]

import numpy as np

rng = np.random.default_rng(7)
from typing import List, Union, Optional, Dict, Any

# Import Tensor from Module 01 (now with gradient support from Module 06)
from tinytorch.foundation.tensor import Tensor

# Enable autograd to add gradient tracking to Tensor
# This module depends on Module 06 (Autograd) being available
from tinytorch.foundation.autograd import enable_autograd

enable_autograd()

# Constants for optimizer defaults
DEFAULT_LEARNING_RATE_SGD = 0.01  # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001  # Default learning rate for Adam/AdamW
DEFAULT_MOMENTUM = 0.9  # Default momentum for SGD
DEFAULT_BETA1 = 0.9  # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999  # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8  # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01  # Default weight decay for AdamW


class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, params: List[Tensor]):
        """
        Initialize optimizer with parameters to optimize.
        """

        # Validate and store parameters
        if not isinstance(params, list):
            params = list(params)

        # Store parameters
        self.params = params

        # Ensure parameters participate in autograd once it is enabled
        for param in self.params:
            if isinstance(param, Tensor):
                param.requires_grad = True
                param.grad = None
        self.step_count = 0  # For algorithms that need step counting

    def zero_grad(self):
        """
        Clear gradients from all parameters.

        """

        for param in self.params:
            param.grad = None

    def step(self):
        """
        Update parameters based on gradients.

        This is abstract - each optimizer implements its own update rule.
        """
        raise NotImplementedError(
            f"Abstract method step() not implemented\n"
            f"  ❌ {self.__class__.__name__} inherits from Optimizer but doesn't define step()\n"
            f"  💡 Each optimizer must implement its own update rule (SGD, Adam, etc.)\n"
            f"  🔧 Override step() in your optimizer subclass:\n"
            f"      def step(self):\n"
            f"          for param in self.params:\n"
            f"              if param.grad is not None:\n"
            f"                  param.data -= self.lr * param.grad.data"
        )


class _ExtractGradientMixin:
    """Mixin added to Optimizer for gradient extraction."""

    def _extract_gradient(self, param: Tensor) -> np.ndarray:
        """
        Extract gradient data as a NumPy array from a parameter.
        """

        grad = param.grad
        if isinstance(grad, Tensor):
            return grad.data
        else:
            return grad


# Attach _extract_gradient to Optimizer so all subclasses inherit it
Optimizer._extract_gradient = _ExtractGradientMixin._extract_gradient


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_SGD,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """
        Initialize SGD optimizer.

        """

        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize momentum buffers (created lazily)
        self.momentum_buffers = [None for _ in self.params]

    def has_momentum(self) -> bool:
        """
        Check if this optimizer uses momentum.
        """
        return self.momentum > 0

    def get_momentum_state(self) -> Optional[List]:
        """
        Get momentum buffers for checkpointing.
        """
        if not self.has_momentum():
            return None
        return [
            buf.copy() if buf is not None else None for buf in self.momentum_buffers
        ]

    def set_momentum_state(self, state: Optional[List]) -> None:
        """
        Restore momentum buffers from checkpointing.
        """
        if state is None or not self.has_momentum():
            return

        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"Momentum state length mismatch\n"
                f"  State has {len(state)} buffers, but optimizer has {len(self.momentum_buffers)} parameters\n"
                f"  Checkpoint was saved with a different model architecture or parameter count\n"
                f"  Ensure you're loading state into an optimizer with the same number of parameters:\n"
                f"   # Check parameter counts match before restoring\n"
                f"   assert len(saved_state) == len(optimizer.params)"
            )

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffers[i] = buf.copy()

    def step(self):
        """
        Perform SGD update step with momentum.

        """

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            # Update momentum buffer
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    # Initialize momentum buffer
                    self.momentum_buffers[i] = np.zeros_like(param.data)

                # Update momentum: v = momentum * v_prev + grad
                self.momentum_buffers[i] = (
                    self.momentum * self.momentum_buffers[i] + grad_data
                )
                grad_data = self.momentum_buffers[i]

            # Update parameter: param = param - lr * grad
            param.data = param.data - self.lr * grad_data

        # Increment step counter
        self.step_count += 1


class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_ADAM,
        betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2),
        eps: float = DEFAULT_EPS,
        weight_decay: float = 0.0,
    ):
        """
        Initialize Adam optimizer.
        """

        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (created lazily)
        self.m_buffers = [None for _ in self.params]  # First moment (mean)
        self.v_buffers = [None for _ in self.params]  # Second moment (variance)


class _AdamUpdateMomentsMixin:
    """Mixin added to Adam for moment updates."""

    def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
        """
        Update first and second moment estimates with bias correction.
        """

        # Initialize buffers if needed
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(grad_data)
            self.v_buffers[i] = np.zeros_like(grad_data)

        # Update biased first moment estimate
        self.m_buffers[i] = (
            self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
        )

        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (
            grad_data**2
        )

        # Compute bias correction
        bias_correction1 = 1 - self.beta1**self.step_count
        bias_correction2 = 1 - self.beta2**self.step_count

        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2

        return m_hat, v_hat


# Attach _update_moments to Adam
Adam._update_moments = _AdamUpdateMomentsMixin._update_moments


class _AdamStepMixin:
    """Mixin added to Adam for step method."""

    def step(self):
        """
        Perform Adam update step by composing helpers.
        """

        # Increment step counter first (needed for bias correction)
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            # Update moments and get bias-corrected estimates
            m_hat, v_hat = self._update_moments(i, grad_data)

            # Update parameter
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Attach step to Adam
Adam.step = _AdamStepMixin.step


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_ADAM,
        betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2),
        eps: float = DEFAULT_EPS,
        weight_decay: float = DEFAULT_WEIGHT_DECAY_ADAMW,
    ):
        """
        Initialize AdamW optimizer.
        """

        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (same as Adam)
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]


class _AdamWUpdateMomentsMixin:
    """Mixin added to AdamW for moment updates."""

    def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
        """
        Update first and second moment estimates with bias correction for AdamW.
        """

        # Initialize buffers if needed
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(grad_data)
            self.v_buffers[i] = np.zeros_like(grad_data)

        # Update biased first moment estimate
        self.m_buffers[i] = (
            self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
        )

        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (
            grad_data**2
        )

        # Compute bias correction
        bias_correction1 = 1 - self.beta1**self.step_count
        bias_correction2 = 1 - self.beta2**self.step_count

        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2

        return m_hat, v_hat


# Attach _update_moments to AdamW
AdamW._update_moments = _AdamWUpdateMomentsMixin._update_moments


class _AdamWStepMixin:
    """Mixin added to AdamW for step method."""

    def step(self):
        """
        Perform AdamW update step by composing helpers with decoupled weight decay.
        """

        # Increment step counter first
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Update moments using PURE gradients (no weight decay mixed in)
            m_hat, v_hat = self._update_moments(i, grad_data)

            # Apply gradient-based update
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Apply decoupled weight decay (separate from gradient update)
            if self.weight_decay != 0:
                param.data = param.data * (1 - self.lr * self.weight_decay)


# Attach step to AdamW
AdamW.step = _AdamWStepMixin.step
