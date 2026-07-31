"""
CIFAR-10 CNN
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)
import argparse
import time

# Import TinyTorch components
from tinytorch import CrossEntropyLoss
from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.layers import Linear
from tinytorch.foundation.activations import ReLU, Softmax
from tinytorch.foundation.convolutions import (
    Conv2d,
    MaxPool2d,
    BatchNorm2d,
)
from tinytorch.foundation.optimizers import Adam
from tinytorch.foundation.dataloader import (
    DataLoader,
    Dataset,
)
from tinytorch.foundation.dataloader import (
    RandomHorizontalFlip,
    RandomCrop,
    Compose,
)

# Import dataset manager
from tinytorch.foundation.data_manager import DatasetManager


class CIFARDataset(Dataset):
    """Custom CIFAR-10 Dataset using Dataset interface from Module 05!

    with data augmentation support using transforms
    """

    def __init__(self, data, labels, transform=None):
        """Initialize with data, labels, and optional transforms."""
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        """Get a single sample -  Dataset interface!"""
        img = self.data[idx]

        # Apply augmentation if provided (training only!)
        if self.transform is not None:
            img = self.transform(img)
            # Convert back to numpy if it became a Tensor
            if isinstance(img, Tensor):
                img = img.data

        return Tensor(img), Tensor([self.labels[idx]])

    def __len__(self):
        """Return dataset size -  Dataset interface!"""
        return len(self.data)

    def get_num_classes(self):
        """Return number of classes."""
        return 10


# Training augmentation using  transforms from Module 05!
train_transforms = Compose(
    [
        RandomHorizontalFlip(p=0.5),
        RandomCrop(32, padding=4),
    ]
)


def flatten(x):
    """Flatten spatial features for dense layers -  implementation!"""
    batch_size = x.data.shape[0]
    return Tensor(x.data.reshape(batch_size, -1))


class CIFARCNN:
    """
    Convolutional Neural Network for CIFAR-10
    """

    def __init__(self):
        print(" Building CIFAR-10 CNN ")

        # Convolutional feature extractors -  spatial modules!
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.bn2 = BatchNorm2d(64)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        # Activation functions
        self.relu = ReLU()

        # Dense classification head
        # After conv1(32→30)→pool(15)→conv2(13)→pool(6): 64*6*6 = 2304 features
        self.fc1 = Linear(64 * 6 * 6, 256)
        self.fc2 = Linear(256, 10)

        # Training mode flag
        self._training = True

        # Calculate total parameters (including BatchNorm gamma/beta)
        conv1_params = 3 * 3 * 3 * 32 + 32
        bn1_params = 32 * 2  # gamma + beta
        conv2_params = 3 * 3 * 32 * 64 + 64
        bn2_params = 64 * 2
        fc1_params = 64 * 6 * 6 * 256 + 256  # Flattened→256
        fc2_params = 256 * 10 + 10  # 256→10 classes
        self.total_params = (
            conv1_params
            + bn1_params
            + conv2_params
            + bn2_params
            + fc1_params
            + fc2_params
        )

        print(f"   Conv1: 3→32 channels + BatchNorm ( modules!)")
        print(f"   Conv2: 32→64 channels + BatchNorm ( modules!)")
        print(f"   Dense: 2304→256→10 ( Linear classification)")
        print(f"   Total parameters: {self.total_params:,}")

    def train(self):
        """Set model to training mode."""
        self._training = True
        self.bn1.train()
        self.bn2.train()
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.bn1.eval()
        self.bn2.eval()
        return self

    def forward(self, x):
        """Forward pass through  CNN architecture."""
        # First conv block: Conv → BatchNorm → ReLU → Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten and classify
        x = flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

    def parameters(self):
        """Get all trainable parameters from  layers."""
        return [
            self.conv1.weight,
            self.conv1.bias,
            self.bn1.gamma,
            self.bn1.beta,
            self.conv2.weight,
            self.conv2.bias,
            self.bn2.gamma,
            self.bn2.beta,
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
        ]


# =============================================================================
# TRAINING LOOP -
# =============================================================================


def train_cifar_cnn(model, train_loader, epochs=3, learning_rate=0.001):
    """Train CNN using complete training system with DataLoader!"""
    print("\n🚀 Training CIFAR-10 CNN ")
    print(f"   Dataset: {len(train_loader.dataset)} color images")
    print(f"   Batch size: {train_loader.batch_size}")

    # Set model to training mode - BatchNorm uses batch statistics
    model.train()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\n   Epoch {epoch+1}/{epochs}:")
        epoch_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        # Use DataLoader to iterate through batches!
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            if batch_idx >= 100:  # Demo mode - limit batches
                break

            # Forward pass
            outputs = model(batch_data)

            outputs_np = np.array(
                outputs.data.data if hasattr(outputs.data, "data") else outputs.data
            )

            loss = criterion.forward(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy
            predictions = np.argmax(outputs_np, axis=1)
            correct += np.sum(predictions == batch_labels.data.flatten())
            total += len(batch_labels.data)

            epoch_loss += loss.data.item()
            batch_count += 1

            # Progress
            if (batch_idx + 1) % 20 == 0:
                acc = 100 * correct / total
                print(
                    f"   Batch {batch_idx+1}: "
                    f"Loss = {loss.data.item():.4f}, Accuracy = {acc:.1f}%"
                )

        # Epoch summary
        epoch_acc = 100 * correct / total
        avg_loss = epoch_loss / max(1, batch_count)
        print(
            f"   → Epoch Complete: Loss = {avg_loss:.4f}, "
            f"Accuracy = {epoch_acc:.1f}% (CNN + DataLoader!)"
        )

    return model


# =============================================================================
# TESTING - Evaluating  CNN on Unseen Images
# =============================================================================


def test_cifar_cnn(model, test_loader, class_names):
    """Test CNN on CIFAR-10 test set using DataLoader."""
    print("\n Testing CNN on Natural Images with DataLoader...")

    # Set model to evaluation mode - BatchNorm uses running statistics
    model.eval()
    print("  Model in eval mode: BatchNorm uses running statistics")

    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    # Test using DataLoader
    for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
        if batch_idx >= 20:  # Demo mode - limit batches
            break

        outputs = model(batch_data)

        outputs_np = np.array(
            outputs.data.data if hasattr(outputs.data, "data") else outputs.data
        )
        predictions = np.argmax(outputs_np, axis=1)
        batch_y = batch_labels.data.flatten()
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)

        # Per-class accuracy
        for j in range(len(batch_y)):
            label = int(batch_y[j])
            class_total[label] += 1
            if predictions[j] == label:
                class_correct[label] += 1

    # Results
    accuracy = 100 * correct / total
    print(f"\n   📊 Overall Test Accuracy: {accuracy:.2f}%")

    # Per-class performance
    print("\n   Per-Class Performance ( CNN's understanding):")
    print("   " + "─" * 50)
    print("   │ Class      │ Accuracy │ Visual               │")
    print("   ├────────────┼──────────┼──────────────────────┤")

    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            bar_length = int(class_acc / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   │ {class_name:10} │  {class_acc:5.1f}%  │ {bar} │")

    print("   " + "─" * 50)

    if accuracy >= 65:
        print("\n  CNN mastered natural image recognition!")
    elif accuracy >= 50:
        print("\n  Good progress! CNN is learning visual features!")
    else:
        print("\n  CNN is still learning... (normal for demo mode)")

    return accuracy


# =============================================================================
# SYSTEMS ANALYSIS - Understanding the Engineering Trade-offs
# =============================================================================


def analyze_cnn_systems(model, batch_size=32):
    """Analyze CNN from an ML systems perspective."""
    print("\n SYSTEMS ANALYSIS of  CNN Implementation:")

    print(f"\n   Model Architecture:")
    print(f"   • Convolutional layers: 2 (3→32→64 channels)")
    print(f"   • Pooling layers: 2 (2×2 max pooling)")
    print(f"   • Dense layers: 2 (2304→256→10)")
    print(f"   • Total parameters: {model.total_params:,}")

    print(f"\n   Computational Complexity:")
    print(f"   • Conv1: 32×30×30×(3×3×3) = 777,600 ops")
    print(f"   • Conv2: 64×13×13×(3×3×32) = 3,093,504 ops")
    print(f"   • Dense: 2,304×256 + 256×10 = 592,384 ops")
    print(f"   • Total: ~4.5M ops per image")

    # Memory profiling table - quantitative systems thinking
    params_mem = model.total_params * 4 / 1024  # KB
    activations_mem = 500  # Peak activations ~500KB per image
    batch_mem = batch_size * 32 * 32 * 3 * 4 / 1024  # Input batch in KB
    total_mem = params_mem + activations_mem + batch_mem

    print(f"\n   MEMORY PROFILING - Where  RAM Goes:")
    print(f"   ┌────────────────────────┬──────────────┬─────────────┐")
    print(f"   │ Component              │ Memory (KB)  │ Percentage  │")
    print(f"   ├────────────────────────┼──────────────┼─────────────┤")
    print(
        f"   │ Parameters (weights)   │ {params_mem:10.1f}   │ {100*params_mem/total_mem:5.1f}%      │"
    )
    print(
        f"   │ Activations (forward)  │ {activations_mem:10.1f}   │ {100*activations_mem/total_mem:5.1f}%      │"
    )
    print(
        f"   │ Batch data ({batch_size} imgs)   │ {batch_mem:10.1f}   │ {100*batch_mem/total_mem:5.1f}%      │"
    )
    print(f"   ├────────────────────────┼──────────────┼─────────────┤")
    print(f"   │ TOTAL per batch        │ {total_mem:10.1f}   │ 100.0%      │")
    print(f"   └────────────────────────┴──────────────┴─────────────┘")
    print(f"\n   KEY INSIGHT: Activations dominate! This is why gradient checkpointing")
    print(
        f"      trades compute (recompute activations) for memory (don't store them)."
    )


# =============================================================================
# MAIN - Orchestrating Complete ML System
# =============================================================================


def main():
    """Demonstrate CIFAR-10 CNN"""

    parser = argparse.ArgumentParser(description="CIFAR-10 CNN")
    parser.add_argument(
        "--test-only", action="store_true", help="Test architecture only"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs (demo mode)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    parser.add_argument(
        "--quick-test", action="store_true", help="Use small subset for testing"
    )
    args = parser.parse_args()

    print("   CIFAR-10 CNN - Natural Image Recognition with  Convolution Modules!")
    print("   Historical significance: CNNs revolutionized computer vision")
    print("    achievement: Spatial feature extraction on real photos")
    print("   Components used:  Conv2d + MaxPool2d + complete system")

    # Class names
    class_names = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Step 1: Load CIFAR-10
    print("\n Loading CIFAR-10 dataset...")
    data_manager = DatasetManager()

    (train_data, train_labels), (test_data, test_labels) = data_manager.get_cifar10()
    print(f"✅ Loaded {len(train_data)} training, {len(test_data)} test images")

    if args.quick_test:
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        test_data = test_data[:500]
        test_labels = test_labels[:500]
        print("   (Using subset for quick testing)")

    # Step 2: Create Datasets and DataLoaders using  Module 05!
    print("\n Creating  Dataset and DataLoader (Module 05)...", train_data[0, 0])

    # Training with augmentation -  transforms!
    train_dataset = CIFARDataset(train_data, train_labels, transform=train_transforms)
    # Testing without augmentation - we want consistent evaluation
    test_dataset = CIFARDataset(test_data, test_labels, transform=None)

    #  DataLoader handles batching and shuffling!
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    print(f"\n   ✅ Data Augmentation: RandomFlip + RandomCrop (training only)")

    # Step 3: Build CNN
    model = CIFARCNN()

    if args.test_only:
        print("\n ARCHITECTURE TEST MODE")
        # Create minimal test data for fast architecture validation
        print("   Using minimal dataset for optimization testing framework...")
        test_data_mini = rng.standard_normal((2, 3, 32, 32)).astype(
            np.float32
        )  # Just 2 samples
        test_labels_mini = np.array([0, 1], dtype=np.int64)  # 2 labels

        # Create minimal dataset and dataloader
        mini_dataset = CIFARDataset(test_data_mini, test_labels_mini)
        mini_loader = DataLoader(
            mini_dataset, batch_size=1, shuffle=False
        )  # Batch size 1

        # Test with single sample from minimal DataLoader
        for batch_data, batch_labels in mini_loader:
            test_output = model(batch_data)
            print(f"✅ Forward pass successful! Shape: {test_output.data.shape}")
            print("✅  CNN + DataLoader work together!")
            break
        return

    # Step 4: Train using  DataLoader
    start_time = time.time()
    model = train_cifar_cnn(model, train_loader, epochs=args.epochs)
    train_time = time.time() - start_time

    # Step 5: Test using  DataLoader
    accuracy = test_cifar_cnn(model, test_loader, class_names)

    # Step 5: Analysis
    analyze_cnn_systems(model, batch_size=args.batch_size)

    print(f"\n⏱  Training time: {train_time:.1f} seconds")
    print(f"   Images/sec: {len(train_dataset) * args.epochs / train_time:.0f}")

    print("\n SUCCESS! CIFAR-10 CNN  Complete!")
    print("\n🎓 What we Accomplished:")
    print("   •  Conv2d extracts spatial features from natural images")
    print("   •  MaxPool2d reduces dimensions while preserving information")
    print("   •  DataLoader efficiently batches and shuffles data")
    print("   •  CNN achieves real accuracy on complex photos")
    print("   •  complete ML system works end-to-end!")


if __name__ == "__main__":
    main()
