__all__ = [
    "rng",
    "EPSILON",
    "Function",
    "AddBackward",
    "MulBackward",
    "PowBackward",
    "SubBackward",
    "DivBackward",
    "MatmulBackward",
    "TransposeBackward",
    "PermuteBackward",
    "SliceBackward",
    "ReshapeBackward",
    "SumBackward",
    "ReLUBackward",
    "SigmoidBackward",
    "TanhBackward",
    "SoftmaxBackward",
    "GELUBackward",
    "MSEBackward",
    "BCEBackward",
    "CrossEntropyBackward",
    "is_grad_enabled",
    "no_grad",
    "enable_autograd",
]


import numpy as np

rng = np.random.default_rng(7)
from typing import Optional, List, Tuple
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.foundation.tensor import Tensor

# Constants for numerical differentiation
EPSILON = 1e-7  # Small perturbation for numerical gradient computation


class Function:
    """
    Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class and implement the apply() method.

    ```
    """

    def __init__(self, *tensors):
        """
        Initialize function with input tensors.

        Args:
            *tensors: Input tensors that will be saved for backward pass
        """
        self.saved_tensors = tensors

    def apply(self, grad_output):
        """
        Compute gradients for inputs.

        """
        raise NotImplementedError("Each Function must implement apply() method")


def _reduce_broadcast_grad(grad, original_shape):
    """
    Reduce gradient to match original tensor shape after broadcasting.
    """

    # Step 1: Remove leading dimensions that weren't in original tensor
    # Example: grad (32, 128) with original (128,) → sum over axis 0
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)

    # Step 2: Collapse dimensions where original had size 1
    # Example: grad (10, 5) with original (10, 1) → sum over axis 1 with keepdims
    for i in range(len(original_shape)):
        if original_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class AddBackward(Function):
    """
    Gradient computation for tensor addition.

    """

    def apply(self, grad_output):
        """
        Compute gradients for addition.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) for the two inputs
        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        # Gradient for second input
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b


class MulBackward(Function):
    """
    Gradient computation for tensor multiplication.

    """

    def apply(self, grad_output):
        """
        Compute gradients for multiplication.
        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output * b
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        # Gradient for second input: grad_output * a
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b


class PowBackward(Function):
    """
    Gradient computation for tensor multiplication.

    """

    def __init__(self, tensor, other):

        super().__init__(tensor)
        self.other = other

    def apply(self, grad_output):
        """
        Compute gradients for multiplication.

        """

        (tensor,) = self.saved_tensors
        grad_tensor = None

        # Gradient for first input: grad_output * b
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            if isinstance(self.other, (int, float)):
                grad_tensor = (
                    grad_output * self.other * (tensor.data ** (self.other - 1))
                )

                # Gérer le broadcasting éventuel
                grad_tensor = _reduce_broadcast_grad(grad_tensor, tensor.data.shape)

        return grad_tensor


class SubBackward(Function):
    """
    Gradient computation for tensor subtraction.

    """

    def apply(self, grad_output):
        """
        Compute gradients for subtraction.

        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output  # ∂(a-b)/∂a = 1
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output  # ∂(a-b)/∂b = -1 (note the negative!)
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b


class DivBackward(Function):
    """
    Gradient computation for tensor division.
    """

    def apply(self, grad_output):
        """
        Compute gradients for division using quotient rule.
        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            # ∂(a/b)/∂a = 1/b
            if isinstance(b, Tensor):
                grad_a = grad_output / b.data
            else:
                grad_a = grad_output / b
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        if isinstance(b, Tensor) and b.requires_grad:
            # ∂(a/b)/∂b = -a/b²
            grad_b = -grad_output * a.data / (b.data**2)
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b


class MatmulBackward(Function):
    """
    Gradient computation for matrix multiplication.

    """

    def apply(self, grad_output):
        """
        Compute gradients for matrix multiplication.

        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output @ b.T
        if isinstance(a, Tensor) and a.requires_grad:
            if b.data.ndim >= 2:
                # Batched: transpose only the last two dims
                b_T = np.swapaxes(b.data, -2, -1)
                grad_a = np.matmul(grad_output, b_T)
            else:
                # 1D b: A(m,k) @ b(k,) -> out(m,)
                # grad_A = outer(grad_output, b): (m,) x (k,) -> (m, k)
                grad_a = np.outer(grad_output, b.data)

        # Gradient for second input: a.T @ grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            if a.data.ndim >= 2:
                # Batched: transpose only the last two dims
                a_T = np.swapaxes(a.data, -2, -1)
                grad_b = np.matmul(a_T, grad_output)
            else:
                # 1D a: a(k,) @ B(k,n) -> out(n,)
                # grad_B = outer(a, grad_output): (k,) x (n,) -> (k, n)
                grad_b = np.outer(a.data, grad_output)

        return grad_a, grad_b


class TransposeBackward(Function):
    """
    Gradient computation for transpose operation.
    """

    def __init__(self, tensor, dim0, dim1):

        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        """
        Compute gradient for transpose.

        """

        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Transpose gradient using the same dims
            if self.dim0 is None and self.dim1 is None:
                # Default: transpose last two dimensions
                if grad_output.ndim < 2:
                    grad_x = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    grad_x = np.transpose(grad_output, axes)
            else:
                # Specific dimensions: swap them back
                axes = list(range(grad_output.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_x = np.transpose(grad_output, axes)

        return (grad_x,)


class PermuteBackward(Function):
    """
    Gradient computation for arbitrary axis permutation (general transpose).
    """

    def __init__(self, tensor, axes):

        super().__init__(tensor)
        self.axes = axes
        # Compute inverse permutation: if axes[i] = j, then inverse_axes[j] = i
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_output):
        """
        Compute gradient for permutation.

        The gradient is permuted back using the inverse permutation.
        """

        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Permute gradient back to original axis order
            grad_x = np.transpose(grad_output, self.inverse_axes)

        return (grad_x,)


class SliceBackward(Function):
    """
    Gradient computation for tensor slicing/indexing operations.

    """

    def __init__(self, tensor, key):

        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_output):
        """
        Compute gradient for slicing operation.

            Tuple with single gradient for input tensor
        """

        (tensor,) = self.saved_tensors
        grad_input = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Create gradient array with same shape as original tensor
            grad_input = np.zeros(self.original_shape, dtype=np.float32)

            # Place gradients back into the sliced positions
            # This is the inverse of the forward slicing operation
            grad_input[self.key] = grad_output

        return (grad_input,)


class ReshapeBackward(Function):
    """
    Gradient computation for reshape operation.
    """

    def __init__(self, tensor, original_shape):
        """
        Args:
            tensor: Input tensor
            original_shape: Shape before reshape
        """
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        """
        Compute gradient for reshape.

        """

        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Reshape gradient back to original shape
            grad_x = grad_output.reshape(self.original_shape)

        return (grad_x,)


class SumBackward(Function):
    """
    Gradient computation for tensor sum.
    """

    def __init__(self, tensor, axis=None, keepdims=False):
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def apply(self, grad_output):
        """
        Compute gradients for sum operation.
        """

        (tensor,) = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # For axis-reduced sums, expand grad_output back along the summed
            # axis before broadcasting, so each row/column gets its own gradient.
            if self.axis is not None and not self.keepdims:
                grad_output = np.expand_dims(grad_output, axis=self.axis)
            return (np.ones_like(tensor.data) * grad_output,)
        return (None,)


class ReLUBackward(Function):
    """
    Gradient computation for ReLU activation.

    ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    """

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """
        Compute gradient for ReLU.
        """

        (tensor,) = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # ReLU gradient: 1 if x > 0, else 0
            relu_grad = (tensor.data > 0).astype(np.float32)
            return (grad_output * relu_grad,)
        return (None,)


class SigmoidBackward(Function):
    """
    Gradient computation for sigmoid activation.
    """

    def __init__(self, input_tensor, output_tensor):
        """
        Initialize with both input and output.

        Args:
            input_tensor: Original input to sigmoid
            output_tensor: Output of sigmoid (saves recomputation)
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        """
        Compute gradient for sigmoid.
        """

        (tensor,) = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # σ'(x) = σ(x) * (1 - σ(x))
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return (grad_output * sigmoid_grad,)
        return (None,)


class TanhBackward(Function):
    """
    Gradient computation for tanh activation.
    """

    def __init__(self, input_tensor, output_tensor):
        """
        Initialize with both input and output.

        Args:
            input_tensor: Original input to tanh
            output_tensor: Output of tanh (saves recomputation)
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        """
        Compute gradient for tanh.
        """

        (tensor,) = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # tanh'(x) = 1 - tanh(x)²
            tanh_grad = 1 - self.output_data**2
            return (grad_output * tanh_grad,)
        return (None,)


class SoftmaxBackward(Function):
    """
    Gradient computation for softmax activation.
    """

    def __init__(self, input_tensor, output_tensor, dim=-1):
        """
        Initialize with input, output, and dimension.

        Args:
            input_tensor: Original input to softmax
            output_tensor: Output of softmax (needed for gradient)
            dim: Dimension along which softmax was applied
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self, grad_output):
        """
        Compute gradient for softmax.
        """

        (tensor,) = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Compute sum(grad_output * softmax) along the softmax dimension
            sum_term = np.sum(
                grad_output * self.output_data, axis=self.dim, keepdims=True
            )

            # Softmax gradient: softmax * (grad_output - sum_term)
            grad_x = self.output_data * (grad_output - sum_term)

            return (grad_x,)
        return (None,)


class GELUBackward(Function):
    """
    Gradient computation for GELU activation.
    """

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """
        Compute gradient for GELU.
        """

        (tensor,) = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data
            # GELU derivative using the sigmoid approximation (matches forward):
            # forward: gelu(x) = x * sigmoid(1.702 * x)
            # d/dx [x * sig(1.702x)] = sig(1.702x) + x * 1.702 * sig(1.702x) * (1 - sig(1.702x))
            sig = 1.0 / (1.0 + np.exp(-1.702 * x))
            gelu_grad = sig + x * 1.702 * sig * (1.0 - sig)

            return (grad_output * gelu_grad,)
        return (None,)


class MSEBackward(Function):
    """
    Gradient computation for Mean Squared Error Loss.
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """
        Compute gradient for MSE loss.
        """

        (predictions,) = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            # Gradient: 2 * (predictions - targets) / N
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples

            return (grad * grad_output,)
        return (None,)


class BCEBackward(Function):
    """
    Gradient computation for Binary Cross-Entropy Loss.
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """
        Compute gradient for BCE loss.
        """

        (predictions,) = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = EPSILON
            p = np.clip(predictions.data, eps, 1 - eps)
            y = self.targets_data

            # Gradient: (p - y) / (p * (1-p) * N)
            grad = (p - y) / (p * (1 - p) * self.num_samples)

            return (grad * grad_output,)
        return (None,)


def _stable_softmax(logits_data):
    """
    Compute softmax probabilities with numerical stability.
    """

    max_logits = np.max(logits_data, axis=1, keepdims=True)
    exp_logits = np.exp(logits_data - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _one_hot_encode(targets, batch_size, num_classes):
    """
    Convert class indices to one-hot vectors.
    """

    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), targets] = 1.0
    return one_hot


class CrossEntropyBackward(Function):
    """
    Gradient computation for Cross-Entropy Loss.

    CrossEntropy: L = -mean(log_softmax(logits)[targets])
    """

    def __init__(self, logits, targets):
        """Initialize with logits and target class indices."""
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_output):
        """
        Compute gradient for cross-entropy loss.

        Uses helper functions for each sub-computation:
        - _stable_softmax(): Converts logits to probabilities (numerically stable)
        - _one_hot_encode(): Converts target indices to one-hot vectors
        """

        (logits,) = self.saved_tensors

        if isinstance(logits, Tensor) and logits.requires_grad:
            softmax = _stable_softmax(logits.data)
            one_hot = _one_hot_encode(
                self.targets_data, self.batch_size, self.num_classes
            )

            # Gradient: (softmax - one_hot) / batch_size
            grad = (softmax - one_hot) / self.batch_size

            return (grad * grad_output,)
        return (None,)


# ===== Global Gradient Tracking Flag =====
# Why this exists: During inference or parameter updates, we don't need to build
# computation graphs. Skipping graph construction saves memory and time.
# This matches PyTorch's torch.no_grad() behavior.
_GRAD_TRACKING_ENABLED = True


def is_grad_enabled():
    """Check if gradient tracking is currently enabled.

    Returns True when operations should build computation graphs,
    False when inside a no_grad() context.
    """
    return _GRAD_TRACKING_ENABLED


class no_grad:
    """
    Context manager that disables gradient tracking.
    """

    def __enter__(self):
        """Save previous state and disable gradient tracking."""
        global _GRAD_TRACKING_ENABLED
        # Save previous state so nested contexts restore correctly
        self._prev_state = _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous gradient tracking state."""
        global _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = self._prev_state
        return False  # Don't suppress exceptions


def enable_autograd(quiet=False):
    """
    Enable gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.
    Call this once to activate gradients globally.

    """

    # Note: hasattr() is LEGITIMATE here because:
    # 1. This is a runtime monkey-patch system (meta-programming)
    # 2. We're checking if a class has been dynamically modified
    # 3. _autograd_enabled is a marker attribute we add at runtime
    # This is the CORRECT use of hasattr() for dynamic class modification
    if hasattr(Tensor, "_autograd_enabled"):
        # Silently return if already enabled - no need to warn
        return

    # ===== STEP 1: Add gradient infrastructure to Tensor =====
    # Store original __init__ to extend it
    _original_init = Tensor.__init__

    def gradient_aware_init(self, data, requires_grad=False):
        """Extended Tensor init that supports gradient tracking."""
        _original_init(self, data)
        self.requires_grad = requires_grad
        self.grad = None

    # Replace __init__ with gradient-aware version
    Tensor.__init__ = gradient_aware_init

    # Store original operations
    # These are guaranteed to exist from Module 01 (Tensor class)
    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_pow = Tensor.__pow__
    _original_div = Tensor.__truediv__
    _original_getitem = Tensor.__getitem__

    # These methods are also guaranteed from Module 01 - trust Single Tensor Class
    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    # Helper to safely check requires_grad (handles tensors created before enable_autograd)
    def _get_requires_grad(tensor):
        """Safely get requires_grad, defaulting to False for pre-autograd tensors.

        Also returns False when inside a no_grad() context, since we should
        not build computation graphs there.
        """
        if not _GRAD_TRACKING_ENABLED:
            return False
        return (
            getattr(tensor, "requires_grad", False)
            if isinstance(tensor, Tensor)
            else False
        )

    def _ensure_grad_attrs(tensor):
        """Ensure tensor has gradient attributes (for tensors created before enable_autograd)."""
        if isinstance(tensor, Tensor):
            if not hasattr(tensor, "requires_grad"):
                tensor.requires_grad = False
            if not hasattr(tensor, "grad"):
                tensor.grad = None

    # Enhanced operations that track gradients
    def tracked_add(self, other):
        """
        Addition with gradient tracking.

        Enhances the original __add__ method to build computation graphs
        when requires_grad=True for any input.
        """
        # Ensure self has gradient attributes
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_add(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)

        return result

    def tracked_radd(self, other):
        """Scalar-left addition with gradient tracking."""
        return tracked_add(self, other)

    def tracked_mul(self, other):
        """
        Multiplication with gradient tracking.

        Enhances the original __mul__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        result = _original_mul(self, other)
        _ensure_grad_attrs(result)

        # Pass other_tensor (always a Tensor) so MulBackward can call .data on it.
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other_tensor)

        return result

    def tracked_pow(self, other):
        """
        Multiplication with gradient tracking.

        Enhances the original __pow__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)
        result = _original_pow(self, other)
        _ensure_grad_attrs(result)

        # Pass other_tensor (always a Tensor) so PowBackward can call .data on it.
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = PowBackward(self, other)

        return result

    def tracked_rmul(self, other):
        """Scalar-left multiplication with gradient tracking."""
        return tracked_mul(self, other)

    def tracked_matmul(self, other):
        """
        Matrix multiplication with gradient tracking.

        Enhances the original matmul method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)

        # Call original matmul from Module 01
        result = _original_matmul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = MatmulBackward(self, other)

        return result

    def tracked_transpose(self, dim0=None, dim1=None):
        """
        Transpose with gradient tracking.

        Enhances the original transpose method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original transpose from Module 01
        result = _original_transpose(self, dim0, dim1)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = TransposeBackward(self, dim0, dim1)

        return result

    def tracked_reshape(self, *shape):
        """
        Reshape with gradient tracking.

        Enhances the original reshape method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)
        original_shape = self.shape

        # Call original reshape from Module 01
        result = _original_reshape(self, *shape)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = ReshapeBackward(self, original_shape)

        return result

    def tracked_sub(self, other):
        """
        Subtraction with gradient tracking.

        Enhances the original __sub__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_sub(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = SubBackward(self, other)

        return result

    def tracked_rsub(self, other):
        """Scalar-left subtraction with gradient tracking."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return tracked_sub(other, self)

    def tracked_div(self, other):
        """
        Division with gradient tracking.

        Enhances the original __truediv__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_div(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = DivBackward(self, other)

        return result

    def tracked_rdiv(self, other):
        """Scalar-left division with gradient tracking."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return tracked_div(other, self)

    def tracked_getitem(self, key):
        """
        Indexing/slicing with gradient tracking.

        Enhances the original __getitem__ method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original __getitem__ from Module 01
        result = _original_getitem(self, key)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SliceBackward(self, key)

        return result

    def sum_op(self, axis=None, keepdims=False):
        """
        Sum operation with gradient tracking.

        Creates a new sum method that builds computation graphs
        when requires_grad=True.
        """
        _ensure_grad_attrs(self)

        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SumBackward(self, axis=axis, keepdims=keepdims)

        return result

    def backward(self, gradient=None, retain_graph=False):
        """
        Compute gradients via backpropagation.

        This is the key method that makes training possible!
        It implements reverse-mode automatic differentiation.
        """

        # Ensure gradient attributes exist
        _ensure_grad_attrs(self)

        # Only compute gradients if required
        if not _get_requires_grad(self):
            return

        # Initialize gradient if not provided (for scalar outputs)
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(
                    f"backward() called on non-scalar tensor without gradient argument.\n"
                    f"  Tensor shape: {self.shape}\n"
                    f"  Issue: For non-scalar outputs, you must provide the gradient from the next layer.\n"
                    f"  Fix: Call backward(gradient) with the gradient tensor from the loss function."
                )

        # Initialize or accumulate gradient
        def accumulate_grad(tensor, grad):
            _ensure_grad_attrs(tensor)
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)

            if grad.shape != tensor.grad.shape:
                # Step 1: Remove extra leading dimensions added during forward pass
                # Example: gradient (batch_size, features) → self.grad (features,)
                while grad.ndim > tensor.grad.ndim:
                    grad = grad.sum(axis=0)

                # Step 2: Sum over dimensions that were size-1 in original tensor
                # Example: bias with shape (1,) broadcast to (batch_size,) during forward
                for i in range(grad.ndim):
                    if tensor.grad.shape[i] == 1 and grad.shape[i] != 1:
                        grad = grad.sum(axis=i, keepdims=True)

            tensor.grad += grad

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                grad_fn = getattr(v, "_grad_fn", None)
                if grad_fn is not None:
                    for p in grad_fn.saved_tensors:
                        if isinstance(p, Tensor):
                            build_topo(p)
                topo.append(v)

        build_topo(self)

        # Set the seed gradient for the root node
        accumulate_grad(self, gradient)

        # 3. Traverse the list in reverse topological order (backpropagation)
        for node in reversed(topo):
            grad_fn = getattr(node, "_grad_fn", None)
            if grad_fn is not None and node.grad is not None:
                # Compute gradients for inputs of this operation
                grads = grad_fn.apply(node.grad)

                # Normalize single-output gradients (e.g., PowBackward returning a single array)
                if not isinstance(grads, (tuple, list)):
                    grads = (grads,)

                # Propagate gradients to parent tensors
                for parent, grad in zip(grad_fn.saved_tensors, grads):
                    if (
                        isinstance(parent, Tensor)
                        and parent.requires_grad
                        and grad is not None
                    ):
                        accumulate_grad(parent, grad)

        # 4. Release the computation graph if retain_graph is False (memory cleanup)
        if not retain_graph:
            for node in topo:
                node._grad_fn = None

    def zero_grad(self):
        """
        Reset gradients to zero.

        Call this before each backward pass to prevent gradient accumulation
        from previous iterations.
        """
        self.grad = None

    # Install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__radd__ = tracked_radd
    Tensor.__sub__ = tracked_sub
    Tensor.__rsub__ = tracked_rsub
    Tensor.__mul__ = tracked_mul
    Tensor.__pow__ = tracked_pow
    Tensor.__rmul__ = tracked_rmul
    Tensor.__truediv__ = tracked_div
    Tensor.__rtruediv__ = tracked_rdiv
    Tensor.__getitem__ = tracked_getitem
    Tensor.matmul = tracked_matmul
    Tensor.transpose = tracked_transpose
    Tensor.reshape = tracked_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    # Patch activations and losses to track gradients
    try:
        from tinytorch.foundation.activations import Sigmoid, ReLU, Softmax, GELU, Tanh
        from tinytorch.foundation.losses import (
            BinaryCrossEntropyLoss,
            MSELoss,
            CrossEntropyLoss,
        )

        # Store original methods
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_tanh_forward = Tanh.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward

        def tracked_sigmoid_forward(self, x):
            """Sigmoid with gradient tracking."""
            result_data = 1.0 / (1.0 + np.exp(-x.data))
            result = Tensor(result_data)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SigmoidBackward(x, result)

            return result

        def tracked_tanh_forward(self, x):
            """Tanh with gradient tracking."""
            result_data = np.tanh(x.data)
            result = Tensor(result_data)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = TanhBackward(x, result)

            return result

        def tracked_relu_forward(self, x):
            """ReLU with gradient tracking."""
            result_data = np.maximum(0, x.data)
            result = Tensor(result_data)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)

            return result

        def tracked_softmax_forward(self, x, dim=-1):
            """Softmax with gradient tracking."""
            # Call original forward to get result using Tensor operations
            result = _original_softmax_forward(self, x, dim=dim)

            # Attach the correct gradient function
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SoftmaxBackward(x, result, dim)

            return result

        def tracked_gelu_forward(self, x):
            """GELU with gradient tracking."""
            # Call original forward to get result
            result = _original_gelu_forward(self, x)

            # Attach the correct gradient function
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = GELUBackward(x)

            return result

        def tracked_bce_forward(self, predictions, targets):
            """Binary cross-entropy with gradient tracking."""
            # Compute BCE loss
            eps = EPSILON
            clamped_preds = np.clip(predictions.data, eps, 1 - eps)
            log_preds = np.log(clamped_preds)
            log_one_minus_preds = np.log(1 - clamped_preds)
            bce_per_sample = -(
                targets.data * log_preds + (1 - targets.data) * log_one_minus_preds
            )
            bce_loss = np.mean(bce_per_sample)

            result = Tensor(bce_loss)

            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = BCEBackward(predictions, targets)

            return result

        def tracked_mse_forward(self, predictions, targets):
            """MSE loss with gradient tracking."""
            # Compute MSE loss
            diff = predictions.data - targets.data
            squared_diff = diff**2
            mse = np.mean(squared_diff)

            result = Tensor(mse)

            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = MSEBackward(predictions, targets)

            return result

        def tracked_ce_forward(self, logits, targets):
            """Cross-entropy loss with gradient tracking."""
            from tinytorch.foundation.losses import log_softmax

            # Compute log-softmax for numerical stability
            log_probs = log_softmax(logits, dim=-1)

            # Select log-probabilities for correct classes
            batch_size = logits.shape[0]
            target_indices = targets.data.astype(int)
            selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

            # Return negative mean
            ce_loss = -np.mean(selected_log_probs)

            result = Tensor(ce_loss)

            if _GRAD_TRACKING_ENABLED and logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits, targets)

            return result

        # Install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        Tanh.forward = tracked_tanh_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward
        CrossEntropyLoss.forward = tracked_ce_forward

    except ImportError:
        # Activations/losses not yet available (happens during module development)
        pass

    # Mark as enabled
    Tensor._autograd_enabled = True

    if not quiet:
        print("✅ Autograd enabled! Tensors now track gradients.")
        print("   - Operations build computation graphs")
        print("   - backward() computes gradients")
        print("   - requires_grad=True enables tracking")


# Auto-enable when module is imported
# Always quiet to avoid cluttering user imports
import os

enable_autograd(quiet=True)
