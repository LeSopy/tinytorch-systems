
__all__ = ['rng', 'Dataset', 'TensorDataset', 'DataLoader', 'RandomHorizontalFlip', 'RandomCrop', 'Compose']


# Essential imports for data loading
import random
import sys
import time
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import numpy as np
rng = np.random.default_rng(7)

# Import real Tensor class from tinytorch package
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tinytorch.foundation.tensor import Tensor

class Dataset(ABC):
    """
    Abstract base class for all datasets.

    """

    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        This method must be implemented by all subclasses to enable
        len(dataset) calls and batch size calculations.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Return the sample at the given index.

        Args:
            idx: Index of the sample to retrieve (0 <= idx < len(dataset))

        Returns:
            The sample at index idx. Format depends on the dataset implementation.
            Could be (data, label) tuple, single tensor, etc.
        """
        pass

class TensorDataset(Dataset):
    """
    Dataset wrapping tensors for supervised learning.

    Each sample is a tuple of tensors from the same index across all input tensors.
    All tensors must have the same size in their first dimension.
    """

    def __init__(self, *tensors):
        """
        Create dataset from multiple tensors.

        Args:
            *tensors: Variable number of Tensor objects

        All tensors must have the same size in their first dimension.
        """
        
        assert len(tensors) > 0, "Must provide at least one tensor"

        # Store all tensors
        self.tensors = tensors

        # Validate all tensors have same first dimension
        first_size = len(tensors[0].data)  # Size of first dimension
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f"Tensor size mismatch in TensorDataset\n"
                    f"  ❌ Tensor 0 has {first_size} samples, but Tensor {i} has {len(tensor.data)} samples\n"
                    f"  💡 All tensors must have the same size in their first dimension (the sample dimension)\n"
                    f"  🔧 Check your data: features.shape[0] should equal labels.shape[0]\n"
                    f"     Example fix: labels = labels[:{first_size}] or features = features[:{len(tensor.data)}]"
                )
        

    def __len__(self) -> int:
        """
        Return number of samples (size of first dimension).

        """
        
        return len(self.tensors[0].data)
        

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """
        Return tuple of tensor slices at given index.
        """
        
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Return tuple of slices from all tensors
        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)
        

class DataLoader:
    """
    Data loader with batching and shuffling support.

    Wraps a dataset to provide batched iteration with optional shuffling.
    Essential for efficient training with mini-batch gradient descent.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        """
        Create DataLoader for batched iteration.

        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        

    def __len__(self) -> int:
        """
        Return number of batches per epoch.
        """
        
        # Calculate number of complete batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        

    def __iter__(self) -> Iterator:
        """
        Return iterator over batches.
        """
        
        # Create list of indices
        indices = list(range(len(self.dataset)))

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # Collate batch - convert list of tuples to tuple of tensors
            yield self._collate_batch(batch)
        

    def _collate_batch(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        """
        Collate individual samples into batch tensors.

        """
        
        if len(batch) == 0:
            return ()

        # Determine number of tensors per sample
        num_tensors = len(batch[0])

        # Group tensors by position
        batched_tensors = []
        for tensor_idx in range(num_tensors):
            # Extract all tensors at this position
            tensor_list = [sample[tensor_idx].data for sample in batch]

            # Stack into batch tensor
            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))

        return tuple(batched_tensors)
      


class RandomHorizontalFlip:
    """
    Randomly flip images horizontally with given probability.

    A simple but effective augmentation for most image datasets.
    Flipping is appropriate when horizontal orientation doesn't change class
    (cats, dogs, cars - not digits or text!).

    Args:
        p: Probability of flipping (default: 0.5)
    """

    def __init__(self, p=0.5):
        """
        Initialize RandomHorizontalFlip.
        """
        
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"Invalid flip probability: {p}\n"
                f"  ❌ p must be between 0.0 and 1.0\n"
                f"  💡 p is the probability of flipping the image horizontally (p=0.5 means 50% chance)\n"
                f"  🔧 Common values: p=0.0 (never flip), p=0.5 (standard), p=1.0 (always flip)"
            )
        self.p = p
        

    def __call__(self, x):
        """
        Apply random horizontal flip to input.
        """
        
        if np.random.random() < self.p:
            is_tensor = isinstance(x, Tensor)
            data = x.data if is_tensor else x

            # Determine width axis for HW/CHW/HWC.
            # Convention (matching _pad_image and RandomCrop): check shape[0]
            # for channels-first (C, H, W) where C <= 4. This is the standard
            # TinyTorch/PyTorch NCHW convention for 3D image tensors.
            if data.ndim == 2:
                # (H, W)
                axis = -1
            elif data.ndim == 3:
                if data.shape[0] <= 4:
                    # Channels-first: (C, H, W) — flip width (last axis)
                    axis = -1
                else:
                    # Channels-last: (H, W, C) — flip width (second-to-last)
                    axis = -2
            else:
                raise ValueError(
                    f"RandomHorizontalFlip requires at least 2D input\n"
                    f"  ❌ Got {data.ndim}D input with shape {data.shape}\n"
                    f"  💡 Images need at least height and width dimensions (H, W) to flip horizontally\n"
                    f"  🔧 Reshape your data: x.reshape(height, width) or x.reshape(1, height, width)"
                )

            flipped = np.flip(data, axis=axis).copy()
            return Tensor(flipped) if is_tensor else flipped
        return x
        

def _pad_image(data, padding):
    """
    Detect image format and apply zero-padding to spatial dimensions only.
    """
    
    if data.ndim == 2:
        # (H, W) format — pad both axes
        return np.pad(data, padding, mode='constant', constant_values=0)
    elif data.ndim == 3:
        if data.shape[0] <= 4:
            # Channels-first: (C, H, W) — pad only H and W
            return np.pad(data,
                          ((0, 0), (padding, padding), (padding, padding)),
                          mode='constant', constant_values=0)
        else:
            # Channels-last: (H, W, C) — pad only H and W
            return np.pad(data,
                          ((padding, padding), (padding, padding), (0, 0)),
                          mode='constant', constant_values=0)
    else:
        raise ValueError(
            f"RandomCrop requires 2D or 3D input\n"
            f"  ❌ Got {data.ndim}D input with shape {data.shape}\n"
            f"  💡 Expected formats: (H, W) for grayscale, (C, H, W) or (H, W, C) for color images\n"
            f"  🔧 Reshape your data:\n"
            f"     - For single grayscale image: x.reshape(height, width)\n"
            f"     - For single color image: x.reshape(channels, height, width)"
        )
    

def _random_crop_region(padded_h, padded_w, target_h, target_w):
    """
    Sample a random (top, left) position for cropping.
    """
    
    top = rng.integers(0, padded_h - target_h + 1)
    left = rng.integers(0, padded_w - target_w + 1)
    return top, left
    

class RandomCrop:
    """
    Randomly crop image after padding.

    This is the standard augmentation for CIFAR-10:
    1. Pad image by `padding` pixels on each side
    2. Randomly crop back to original size

    This simulates small translations in the image, forcing the model
    to recognize objects regardless of their exact position.

    Args:
        size: Output crop size (int for square, or tuple (H, W))
        padding: Pixels to pad on each side before cropping (default: 4)
    """

    def __init__(self, size, padding=4):
        """
        Initialize RandomCrop.
        """
        
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
        

    def __call__(self, x):
        """
        Apply random crop after padding.

        Composes _pad_image and _random_crop_region to perform:
        1. Pad the image with zeros on spatial dimensions
        2. Sample a random crop position
        3. Extract the crop and return
        """
        
        is_tensor = isinstance(x, Tensor)
        data = x.data if is_tensor else x

        target_h, target_w = self.size

        # Step 1: Pad the image (handles format detection internally)
        padded = _pad_image(data, self.padding)

        # Step 2: Determine padded spatial dims and sample crop position
        if data.ndim == 2:
            padded_h, padded_w = padded.shape
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[top:top + target_h, left:left + target_w]
        elif data.shape[0] <= 4:
            # Channels-first: (C, H, W)
            padded_h, padded_w = padded.shape[1], padded.shape[2]
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[:, top:top + target_h, left:left + target_w]
        else:
            # Channels-last: (H, W, C)
            padded_h, padded_w = padded.shape[0], padded.shape[1]
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[top:top + target_h, left:left + target_w, :]

        return Tensor(cropped) if is_tensor else cropped
        

class Compose:
    """
    Compose multiple transforms into a pipeline.

    Applies transforms in sequence, passing output of each
    as input to the next.

    Args:
        transforms: List of transform callables
    """

    def __init__(self, transforms):
        """
        Initialize Compose with list of transforms.
        """
        self.transforms = transforms

    def __call__(self, x):
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            x = transform(x)
        return x
