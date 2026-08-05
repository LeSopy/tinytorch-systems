#!/usr/bin/env python3
"""
TinyTorch Dataset Manager
========================
"""

import os
import sys
import urllib.request
import tarfile
import pickle
import gzip
import shutil
import numpy as np

rng = np.random.default_rng(7)
from pathlib import Path

# Add project root for TinyTorch imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Dataset size information (approximate)
DATASET_INFO = {
    "cifar10": {
        "name": "CIFAR-10",
        "download_size_mb": 170,
        "extracted_size_mb": 180,
        "description": "60,000 natural images (32x32 RGB) in 10 classes",
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    },
}


class DatasetManager:
    """Handles all dataset logistics for TinyTorch milestone examples."""

    def __init__(self, data_dir=None, auto_confirm=False):
        """
        Initialize the dataset manager.

        Args:
            data_dir: Directory to store datasets (default: milestones/datasets/)
            auto_confirm: If True, skip download confirmation prompts
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "datasets"
        else:
            self.data_dir = Path(data_dir)

        self.auto_confirm = auto_confirm

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)

    def check_disk_space(self, required_mb):
        """Check if sufficient disk space is available.

        Returns:
            tuple: (available_mb, has_space)
        """
        try:
            total, used, free = shutil.disk_usage(self.data_dir)
            available_mb = free / (1024 * 1024)
            return available_mb, available_mb >= required_mb
        except Exception:
            # If we can't check, assume we have space
            return -1, True

    def confirm_download(self, dataset_key):
        """Display dataset info and confirm download with user.

        Returns:
            bool: True if user confirms, False otherwise
        """
        if dataset_key not in DATASET_INFO:
            return True  # Unknown dataset, proceed anyway

        info = DATASET_INFO[dataset_key]
        required_mb = info["extracted_size_mb"]
        available_mb, has_space = self.check_disk_space(required_mb)

        # Print dataset information
        print("\n" + "=" * 60)
        print(f"  This milestone requires the {info['name']} dataset")
        print("=" * 60)
        print()
        print("  Dataset Info:")
        print("  +" + "-" * 18 + "+" + "-" * 38 + "+")
        print(f"  | {'Name':<16} | {info['name']:<36} |")
        print(f"  | {'Description':<16} | {info['description']:<36} |")
        print(
            f"  | {'Download Size':<16} | {'~' + str(info['download_size_mb']) + ' MB':<36} |"
        )
        print(f"  | {'Location':<16} | {str(self.data_dir)[:36]:<36} |")
        print("  +" + "-" * 18 + "+" + "-" * 38 + "+")
        print()
        print("  System Check:")
        if available_mb > 0:
            print(f"    * Available space: {available_mb:.1f} MB")
        else:
            print(f"    * Available space: (unable to check)")
        print(f"    * Required space:  ~{required_mb} MB")

        if not has_space:
            print(f"    * Status: INSUFFICIENT SPACE")
            print()
            print(
                f"  Please free up at least {required_mb - available_mb:.0f} MB and try again."
            )
            print("=" * 60)
            return False
        else:
            print(f"    * Status: Ready to download")
        print()

        # Auto-confirm mode (for CI/testing)
        if self.auto_confirm:
            print("  [Auto-confirm enabled, proceeding with download]")
            print("=" * 60)
            return True

        # Ask user
        try:
            response = input(f"  Download {info['name']}? [Y/n]: ").strip().lower()
            print("=" * 60)
            return response in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            print("\n  Download cancelled.")
            print("=" * 60)
            return False

    def download_with_progress(self, url, filename):
        """Download with visual progress bar."""

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)

                # Visual progress bar
                bar_length = 30
                filled = int(bar_length * percent / 100)
                bar = "█" * filled + "░" * (bar_length - filled)

                print(
                    f"\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                    end="",
                    flush=True,
                )

        print(f"📥 Downloading {Path(filename).name}...")
        urllib.request.urlretrieve(url, filename, progress_hook)
        print("\n✅ Download complete!")

    def get_cifar10(self):
        """Download and prepare CIFAR-10 dataset for CNN milestone."""
        cifar_dir = self.data_dir / "cifar-10"
        cifar_dir.mkdir(exist_ok=True)

        # Check if already downloaded
        data_file = cifar_dir / "cifar-10-python.tar.gz"
        extracted_dir = cifar_dir / "cifar-10-batches-py"

        if not extracted_dir.exists():
            # Download if needed
            if not data_file.exists():
                # Confirm download with user
                if not self.confirm_download("cifar10"):
                    raise RuntimeError("CIFAR-10 download cancelled by user")

                url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                self.download_with_progress(url, data_file)

            # Extract (filter='data' prevents path traversal; Python 3.12+)
            print("📦 Extracting CIFAR-10...")
            with tarfile.open(data_file, "r:gz") as tar:
                tar.extractall(cifar_dir, filter="data")
            print("✅ Extraction complete!")
        else:
            print(f"✅ CIFAR-10 already downloaded at {cifar_dir}")

        # Load data from pickle files
        train_data, train_labels = [], []
        for i in range(1, 6):
            batch_file = extracted_dir / f"data_batch_{i}"
            with open(batch_file, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                train_data.append(batch[b"data"])
                train_labels.extend(batch[b"labels"])

        # Test data
        test_file = extracted_dir / "test_batch"
        with open(test_file, "rb") as f:
            test_batch = pickle.load(f, encoding="bytes")
            test_data = test_batch[b"data"]
            test_labels = test_batch[b"labels"]

        # Reshape to proper image format
        train_data = (
            np.vstack(train_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        )
        test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        train_labels = np.array(train_labels, dtype=np.int64)
        test_labels = np.array(test_labels, dtype=np.int64)

        print(
            f"📊 CIFAR-10 loaded: {len(train_data)} training, {len(test_data)} test images"
        )
        return (train_data, train_labels), (test_data, test_labels)


def main():
    """Test dataset manager functionality."""
    print(" Testing TinyTorch Dataset Manager")
    print("=" * 50)

    manager = DatasetManager()

    # Test dataset
    print("\n1. Testing Perceptron Data:")

    print("\n4. Testing CIFAR-10 (this may take a moment):")
    try:
        (train_X, train_y), (test_X, test_y) = manager.get_cifar10()
        print(f"   Shape: train_X={train_X.shape}, test_X={test_X.shape}")
    except Exception as e:
        print(f"   CIFAR-10 download failed: {e}")

    print("\n Dataset Manager test complete!")


if __name__ == "__main__":
    main()
