

__all__ = ['rng', 'BYTES_PER_FLOAT32', 'KB_TO_BYTES', 'MB_TO_BYTES', 'Tensor']

# %% ../../modules/01_tensor/tensor.ipynb #9f43abe6
import numpy as np
rng = np.random.default_rng(7)

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% ../../modules/01_tensor/tensor.ipynb #d6b380df
class Tensor:
    

    def __init__(self, data):
        """Create a new tensor from data.

        TODO: Initialize a Tensor by wrapping data in a NumPy array and setting attributes.

        APPROACH:
        1. Convert data to NumPy array with dtype=float32
        2. Store the array as self.data
        3. Set self.shape from the array's shape
        4. Set self.size from the array's size
        5. Set self.dtype from the array's dtype
        EXAMPLE:
        >>> t = Tensor([1, 2, 3])
        >>> print(t.shape)
        (3,)
        >>> print(t.size)
        3

        HINT: Use np.array(data, dtype=np.float32) to convert data to NumPy array
        """
        ### BEGIN 
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            data = np.stack([t.data for t in data])
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        ### END

    def __repr__(self):
        """String representation of tensor for debugging."""
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """Human-readable string representation."""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data

    def memory_footprint(self):
        """Calculate exact memory usage in bytes.

        Systems Concept: Understanding memory footprint is fundamental to ML systems.
        Before running any operation, engineers should know how much memory it requires.

        Returns:
            int: Memory usage in bytes (e.g., 1000x1000 float32 = 4MB)
        """
        return self.data.nbytes

    @property
    def ndim(self):
        """Number of tensor dimensions (0=scalar, 1=vector, 2=matrix, ...)."""
        return len(self.shape)

    def numel(self):
        """Return total number of elements (PyTorch-compatible)."""
        return self.size

    def contiguous(self):
        """Return a contiguous copy of the tensor data (PyTorch-compatible)."""
        return Tensor(np.ascontiguousarray(self.data))

    def view(self, *shape):
        """Alias for reshape, matching PyTorch's Tensor.view() for contiguous tensors."""
        return self.reshape(*shape)

    def masked_fill(self, mask, value):
        """Fill positions where mask is True with value, matching PyTorch's masked_fill.

        Used in transformer attention to set padding positions to -inf before softmax.

        Args:
            mask:  A Tensor or numpy array of booleans, same shape as self (or broadcastable).
            value: Scalar fill value (e.g. float('-inf') for attention masking).

        Returns:
            New Tensor with masked positions replaced by value.
        """
        mask_array = mask.data.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
        result = self.data.copy()
        result[mask_array] = value
        return Tensor(result)

    def __add__(self, other):
        """Add two tensors element-wise with broadcasting support.


        APPROACH:
        1. Check if other is a Tensor (use isinstance)
        2. If Tensor: add self.data + other.data
        3. If scalar: add self.data + other (broadcasting)
        4. Wrap result in new Tensor
        """
        ### BEGIN 
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)
        ### END 

    def __radd__(self, other):
        """Support natural scalar arithmetic: scalar + tensor."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract two tensors element-wise."""

        ### BEGIN 
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)
        ### END 

    def __rsub__(self, other):
        """Support natural scalar arithmetic: scalar - tensor."""
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        return Tensor(other - self.data)

    def __mul__(self, other):
        """Multiply two tensors element-wise (NOT matrix multiplication). """

        ### BEGIN 
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)
        ### END 

    def __rmul__(self, other):
        """Support natural scalar arithmetic: scalar * tensor."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide two tensors element-wise."""

        ### BEGIN 
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)
        ### END 

    def __rtruediv__(self, other):
        """Support natural scalar arithmetic: scalar / tensor."""
        if isinstance(other, Tensor):
            return Tensor(other.data / self.data)
        return Tensor(other / self.data)

    def _validate_matmul_shapes(self, other):
        """Validate that two tensors are compatible for matrix multiplication."""

        ### BEGIN 
        if not isinstance(other, Tensor):
            raise TypeError(
                f"Matrix multiplication requires Tensor, got {type(other).__name__}\n"
                f"  ❌ Cannot perform: Tensor @ {type(other).__name__}\n"
                f"  💡 Matrix multiplication (@) only works between two Tensors\n"
                f"  🔧 Wrap your data: Tensor({other}) @ other_tensor"
            )
        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError(
                f"Matrix multiplication requires at least 1D tensors\n"
                f"  ❌ Got shapes: {self.shape} @ {other.shape}\n"
                f"  💡 Scalars (0D tensors) cannot be matrix-multiplied; use * for element-wise\n"
                f"  🔧 Reshape scalar to 1D: tensor.reshape(1) or use tensor * scalar"
            )
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}\n"
                    f"  ❌ Inner dimensions don't match: {self.shape[-1]} vs {other.shape[-2]}\n"
                    f"  💡 For A @ B, A's last dim must equal B's second-to-last dim\n"
                    f"  🔧 Try: other.transpose() to get shape {other.shape[::-1]}, or reshape self"
                )
        ### END 

    def matmul(self, other):
        """Matrix multiplication of two tensors."""

        ### BEGIN 
        self._validate_matmul_shapes(other)

        # This is intentionally slower than np.matmul to demonstrate the value of vectorization

        a = self.data
        b = other.data

        # Handle 2D matrices with explicit loops (educational)
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)

            # Each output element is a dot product of a row from A and a column from B
            for i in range(M):
                for j in range(N):
                    # Dot product of row i from A with column j from B
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            
            result_data = np.matmul(a, b)

        return Tensor(result_data)
        ### END 

    def __matmul__(self, other):
        """Enable @ operator for matrix multiplication."""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing operations on Tensors."""

        ### BEGIN 
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)
        ### END 

    def reshape(self, *shape):
        """Reshape tensor to new dimensions."""

        ### BEGIN 
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError(
                    f"Cannot reshape {self.shape} with multiple unknown dimensions\n"
                    f"  ❌ Found {new_shape.count(-1)} dimensions set to -1 in {new_shape}\n"
                    f"  💡 Only one dimension can be inferred; others must be specified\n"
                    f"  🔧 Replace all but one -1 with explicit sizes (total elements: {self.size})"
                )
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            if self.size % known_size != 0:
                raise ValueError(
                    f"Cannot infer -1 dimension: {self.size} elements is not "
                    f"divisible by the known dimensions product {known_size}\n"
                    f"  ❌ {self.size} % {known_size} = {self.size % known_size}\n"
                    f"  💡 The -1 dimension must be a whole number"
                )
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}\n"
                f"  ❌ Element count mismatch: {self.size} elements vs {target_size} elements\n"
                f"  💡 Reshape preserves data, so total elements must stay the same\n"
                f"  🔧 Use -1 to infer a dimension: reshape(-1, {new_shape[-1] if len(new_shape) > 0 else 1}) lets NumPy calculate"
            )
        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)
        ### END 

    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions.

        TODO: Swap tensor dimensions (default: swap last two dimensions).

        APPROACH:
        1. If no dims specified: swap last two dimensions (most common case)
        2. For 1D tensors: return copy (no transpose needed)
        3. If both dims specified: swap those specific dimensions
        4. Use np.transpose with axes list to perform the swap
        5. Return result wrapped in new Tensor

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3
        >>> transposed = t.transpose()
        >>> print(transposed.data)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]  # 3×2

        HINTS:
        - Create axes list: [0, 1, 2, ...] then swap positions
        - For default: axes[-2], axes[-1] = axes[-1], axes[-2]
        - Use np.transpose(self.data, axes)
        """
        ### BEGIN 
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                provided = f"dim0={dim0}" if dim1 is None else f"dim1={dim1}"
                missing = "dim1" if dim1 is None else "dim0"
                raise ValueError(
                    f"Transpose requires both dimensions to be specified\n"
                    f"  ❌ Got {provided}, but {missing} is None\n"
                    f"  💡 Either provide both dims or neither (default swaps last two)\n"
                    f"  🔧 Use transpose({dim0 if dim0 is not None else 0}, {dim1 if dim1 is not None else 1}) or just transpose()"
                )
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)
        ### END 

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis."""

        ### BEGIN 
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
        ### END 

    def mean(self, axis=None, keepdims=False):
        """Compute mean of tensor along specified axis."""

        ### BEGIN 
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
        ### END 

    def max(self, axis=None, keepdims=False):
        """Find maximum values along specified axis. """

        ### BEGIN 
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
        ### END 
