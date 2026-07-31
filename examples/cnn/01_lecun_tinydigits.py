"""
LeNet Part 1: TinyDigits
==========================

"""

import sys
import os
import time
import pickle
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box

# Import TinyTorch components
from tinytorch import Tensor, SGD, CrossEntropyLoss
from tinytorch.foundation.convolutions import Conv2d, MaxPool2d
from tinytorch.foundation.layers import Linear, ReLU
from tinytorch.foundation.dataloader import DataLoader, TensorDataset

console = Console()

# ============================================================================
#  DATA LOADING
# ============================================================================


def load_digits_dataset():
    """
    Load the TinyDigits dataset (8×8 curated digits).
    """
    # Load from TinyDigits dataset (shipped with TinyTorch)
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / "datasets" / "tinydigits" / "train.pkl"
    test_path = project_root / "datasets" / "tinydigits" / "test.pkl"

    if not train_path.exists() or not test_path.exists():
        console.print(f"[red]✗ TinyDigits dataset not found![/red]")
        console.print(f"[yellow]Expected location: {train_path.parent}[/yellow]")
        console.print(
            "[yellow]Run: python3 datasets/tinydigits/create_tinydigits.py[/yellow]"
        )
        sys.exit(1)

    # Load training data
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    train_images = train_data["images"]  # (1000, 8, 8)
    train_labels = train_data["labels"]  # (1000,)

    # Load test data
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    test_images = test_data["images"]  # (200, 8, 8)
    test_labels = test_data["labels"]  # (200,)

    # CNN expects (batch, channels, height, width)
    # Add channel dimension: (N, 8, 8) → (N, 1, 8, 8)
    train_images = train_images[:, np.newaxis, :, :]  # (1000, 1, 8, 8)
    test_images = test_images[:, np.newaxis, :, :]  # (200, 1, 8, 8)

    return (
        Tensor(train_images.astype(np.float32)),
        Tensor(train_labels.astype(np.int64)),
        Tensor(test_images.astype(np.float32)),
        Tensor(test_labels.astype(np.int64)),
    )


# ============================================================================
#  NETWORK ARCHITECTURE
# ============================================================================


class SimpleCNN:
    """
    Simple Convolutional Neural Network for digit classification.
    """

    def __init__(self):
        # Convolutional layers
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        # After conv(3×3) and pool(2×2): 8×8 → 6×6 → 3×3
        # Flattened size: 8 channels × 3 × 3 = 72
        self.fc = Linear(in_features=72, out_features=10)

        # Set requires_grad for all parameters
        self.conv1.weight.requires_grad = True
        self.conv1.bias.requires_grad = True
        self.fc.weight.requires_grad = True
        self.fc.bias.requires_grad = True

        self.params = [self.conv1.weight, self.conv1.bias, self.fc.weight, self.fc.bias]

    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)

    def forward(self, x):
        # Conv + ReLU + Pool
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        # Flatten: (batch, 8, 3, 3) → (batch, 72)
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)

        # Final classification
        out = self.fc.forward(out)
        return out

    def parameters(self):
        return self.params


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================


def train_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch."""
    total_loss = 0.0
    n_batches = 0

    for batch_images, batch_labels in dataloader:
        # Forward pass
        logits = model(batch_images)
        loss = criterion.forward(logits, batch_labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.data.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_accuracy(model, images, labels):
    """Evaluate model accuracy on a dataset."""
    logits = model(images)

    predictions = np.argmax(logits.data, axis=1)

    accuracy = 100.0 * np.mean(predictions == labels.data)
    avg_loss = np.mean((predictions - labels.data) ** 2)
    return accuracy, avg_loss


def press_enter_to_continue():
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            console.input("\n[yellow]Press Enter to continue...[/yellow] ")
        except EOFError:
            pass
        console.print()


# ============================================================================
# MAIN EXAMPLE DEMONSTRATION
# ============================================================================


def train_cnn():
    """Main training loop following 5-Act structure."""

    # Load data
    console.print("[bold] Loading Handwritten Digits Dataset...[/bold]")
    train_images, train_labels, test_images, test_labels = load_digits_dataset()

    console.print(f"  Training samples: [cyan]{len(train_images.data)}[/cyan]")
    console.print(f"  Test samples: [cyan]{len(test_images.data)}[/cyan]")
    console.print(
        f"  Image shape: [cyan]{train_images.data[0].shape}[/cyan] (1 channel, 8×8 pixels)"
    )
    console.print(f"  Classes: [cyan]10[/cyan] (digits 0-9)")

    # Show training data structure
    console.print(f"\n  [dim]Sample digit values (first image, top-left 3×3):[/dim]")
    sample = train_images.data[0, 0, :3, :3]
    for row in sample:
        console.print(f"    {' '.join(f'{val:.2f}' for val in row)}")

    press_enter_to_continue()

    # Create model
    console.print("\n Building Convolutional Neural Network...")
    model = SimpleCNN()

    # Count parameters
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    conv_params = np.prod(model.conv1.weight.shape) + np.prod(model.conv1.bias.shape)
    fc_params = np.prod(model.fc.weight.shape) + np.prod(model.fc.bias.shape)

    console.print(f"  ✓ Conv layer: [cyan]{conv_params}[/cyan] parameters")
    console.print(f"  ✓ FC layer: [cyan]{fc_params}[/cyan] parameters")
    console.print(f"  ✓ Total: [bold cyan]{total_params}[/bold cyan] parameters")

    # Hyperparameters
    console.print("\n[bold]  Training Configuration:[/bold]")
    epochs = 50
    batch_size = 32
    learning_rate = 0.01

    config_table = Table(show_header=False, box=None)
    config_table.add_row("Epochs:", f"[cyan]{epochs}[/cyan]")
    config_table.add_row("Batch size:", f"[cyan]{batch_size}[/cyan]")
    config_table.add_row("Learning rate:", f"[cyan]{learning_rate}[/cyan]")
    config_table.add_row("Optimizer:", "[cyan]SGD[/cyan]")
    config_table.add_row("Loss:", "[cyan]CrossEntropyLoss[/cyan]")
    console.print(config_table)

    # Create optimizer and loss
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # Create dataloader
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    press_enter_to_continue()

    console.print("[bold] Training CNN on Handwritten Digits...[/bold]\n")

    # Before training
    initial_acc, initial_loss = evaluate_accuracy(model, test_images, test_labels)
    console.print(f"[yellow]Before training:[/yellow] Accuracy = {initial_acc:.1f}%\n")

    # Training loop
    history = {
        "train_loss": [],
        "test_accuracy": [],
        "train_accuracy": [],  # Track training accuracy to detect overfitting
    }
    start_time = time.time()

    # Use Live display with spinner for real-time feedback
    with Live(console=console, refresh_per_second=10) as live:
        for epoch in range(epochs):
            # Update spinner before training
            spinner_text = Text()
            spinner_text.append("⠋ ", style="cyan")
            spinner_text.append(f"Epoch {epoch+1:3d}/{epochs}  Training...")
            live.update(spinner_text)

            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer)

            # Evaluate on both train and test
            train_acc, _ = evaluate_accuracy(model, train_images, train_labels)
            test_acc, _ = evaluate_accuracy(model, test_images, test_labels)

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["test_accuracy"].append(test_acc)

            if (epoch + 1) % 5 == 0:  # Print every 5 epochs
                gap = train_acc - test_acc
                gap_indicator = "⚠️" if gap > 10 else "✓"
                live.console.print(
                    f"Epoch {epoch+1:3d}/{epochs}  "
                    f"Loss: {train_loss:.4f}  "
                    f"Train: {train_acc:.1f}%  "
                    f"Test: {test_acc:.1f}%  "
                    f"{gap_indicator} Gap: {gap:.1f}%"
                )

    training_time = time.time() - start_time

    press_enter_to_continue()

    console.print("[bold] The Results:[/bold]\n")

    final_train_acc = history["train_accuracy"][-1]
    final_test_acc = history["test_accuracy"][-1]
    final_loss = history["train_loss"][-1]
    overfitting_gap = final_train_acc - final_test_acc

    table = Table(title="Training Outcome", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=20)
    table.add_column("Status", style="magenta", width=20)

    table.add_row(
        "Train Accuracy",
        f"{final_train_acc:.1f}%",
        f"↑ +{final_train_acc - initial_acc:.1f}%",
    )
    table.add_row(
        "Test Accuracy",
        f"{final_test_acc:.1f}%",
        f"↑ +{final_test_acc - initial_acc:.1f}%",
    )
    table.add_row(
        "Overfitting Gap",
        f"{overfitting_gap:.1f}%",
        "✓ Healthy" if overfitting_gap < 10 else "⚠️ Overfitting",
    )
    table.add_row("Training Time", f"{training_time*1000:.0f}ms", "—")

    console.print(table)

    press_enter_to_continue()

    # Sample predictions
    console.print("[bold] Sample Predictions:[/bold]")
    sample_images = Tensor(test_images.data[:10])  # First 10 test samples
    logits = model(sample_images)
    predictions = np.argmax(logits.data, axis=1)

    samples_table = Table(show_header=True, box=box.SIMPLE)
    samples_table.add_column("True", style="cyan", justify="center")
    samples_table.add_column("Pred", style="green", justify="center")
    samples_table.add_column("Result", justify="center")

    for i in range(10):
        true_label = int(test_labels.data[i])
        pred_label = int(predictions[i])
        result = "✓" if true_label == pred_label else "✗"
        style = "green" if true_label == pred_label else "red"
        samples_table.add_row(
            str(true_label), str(pred_label), f"[{style}]{result}[/{style}]"
        )

    console.print(samples_table)

    # Key insights
    console.print("\n[bold] Key Insights:[/bold]")
    console.print(f"  • CNNs preserve spatial structure")
    console.print(f"  • Conv layers detect local patterns (edges → digits)")
    console.print(f"  • Pooling provides translation invariance")
    console.print(f"  • {total_params} params vs ~5,000 for MLP with similar accuracy!")

    press_enter_to_continue()


if __name__ == "__main__":
    train_cnn()
