__all__ = [
    "rng",
    "DEFAULT_MAX_LR",
    "DEFAULT_MIN_LR",
    "DEFAULT_TOTAL_EPOCHS",
    "CosineSchedule",
    "clip_grad_norm",
    "Trainer",
    "trainer_init",
    "trainer_train_epoch",
    "trainer_evaluate",
    "trainer_save_checkpoint",
    "trainer_load_checkpoint",
]

import numpy as np

rng = np.random.default_rng(7)
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import sys
import os

# Import dependencies from other modules
from tinytorch.foundation.tensor import Tensor
from tinytorch.foundation.layers import Linear
from tinytorch.foundation.losses import MSELoss, CrossEntropyLoss
from tinytorch.foundation.optimizers import SGD, AdamW

# Enable autograd for gradient tracking (required for training)
from tinytorch.foundation.autograd import enable_autograd

enable_autograd()

# Constants for learning rate scheduling defaults
DEFAULT_MAX_LR = 0.1  # Default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01  # Default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS = 100  # Default total epochs for learning rate schedule


class CosineSchedule:
    """
    Cosine annealing learning rate schedule.
    """

    def __init__(
        self,
        max_lr: float = DEFAULT_MAX_LR,
        min_lr: float = DEFAULT_MIN_LR,
        total_epochs: int = DEFAULT_TOTAL_EPOCHS,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor


def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    """
    Clip gradients by global norm to prevent exploding gradients.
    """

    if not parameters:
        return 0.0

    # Collect all gradients and compute global norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            # Handle both Tensor gradients and numpy array gradients
            if isinstance(param.grad, np.ndarray):
                grad_data = param.grad
            else:
                # Trust that Tensor has .data attribute
                grad_data = param.grad.data
            total_norm += np.sum(grad_data**2)

    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                # Handle both Tensor gradients and numpy array gradients
                if isinstance(param.grad, np.ndarray):
                    param.grad = param.grad * clip_coef
                else:
                    # Trust that Tensor has .data attribute
                    param.grad.data = param.grad.data * clip_coef

    return float(total_norm)


class Trainer:
    """
    Complete training orchestrator for neural networks.
    """

    def _get_model_state(self):
        """Extract model parameters for checkpointing."""
        return {i: param.data.copy() for i, param in enumerate(self.model.parameters())}

    def _set_model_state(self, state):
        """Restore model parameters from checkpoint."""
        for i, param in enumerate(self.model.parameters()):
            if i in state:
                param.data = state[i].copy()

    def _get_optimizer_state(self):
        """Extract optimizer state for checkpointing."""
        state = {}
        state["lr"] = self.optimizer.lr
        if hasattr(self.optimizer, "has_momentum") and self.optimizer.has_momentum():
            momentum_state = self.optimizer.get_momentum_state()
            if momentum_state is not None:
                state["momentum_buffers"] = momentum_state
        return state

    def _set_optimizer_state(self, state):
        """Restore optimizer state from checkpoint."""
        if "lr" in state:
            self.optimizer.lr = state["lr"]
        if "momentum_buffers" in state:
            if (
                hasattr(self.optimizer, "has_momentum")
                and self.optimizer.has_momentum()
            ):
                self.optimizer.set_momentum_state(state["momentum_buffers"])

    def _get_scheduler_state(self):
        """Extract scheduler state for checkpointing."""
        if self.scheduler is None:
            return None
        return {
            "max_lr": getattr(self.scheduler, "max_lr", None),
            "min_lr": getattr(self.scheduler, "min_lr", None),
            "total_epochs": getattr(self.scheduler, "total_epochs", None),
        }

    def _set_scheduler_state(self, state):
        """Restore scheduler state from checkpoint."""
        if state is None or self.scheduler is None:
            return
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)


def trainer_init(self, model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None):
    """
    Initialize trainer with model and training components.
    """

    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.scheduler = scheduler
    self.grad_clip_norm = grad_clip_norm

    # Enable gradient tracking for all model parameters.
    # Layers (e.g. Linear) may be created without requires_grad=True,
    # so we set it explicitly here to ensure backward() populates param.grad.
    # Guard against raw numpy arrays passed by test stubs or non-Tensor params.
    for param in model.parameters():
        if isinstance(param, Tensor):
            param.requires_grad = True

    # Training state
    self.epoch = 0
    self.step = 0
    self.training_mode = True

    # History tracking
    self.history = {"train_loss": [], "eval_loss": [], "learning_rates": []}


Trainer.__init__ = trainer_init


def _trainer_process_batch(self, inputs, targets, accumulation_steps):
    """
    Process one batch: forward pass, loss computation, backward pass.
    """

    # Forward pass
    outputs = self.model.forward(inputs)
    loss = self.loss_fn.forward(outputs, targets)

    # Scale loss for accumulation
    scaled_loss = loss.data / accumulation_steps

    # Backward pass with scaled gradients
    scaled_gradient = np.ones_like(loss.data) / accumulation_steps
    loss.backward(scaled_gradient)

    return float(scaled_loss)


Trainer._process_batch = _trainer_process_batch


def _trainer_optimizer_update(self):
    """
    Clip gradients (if enabled) and step the optimizer.
    """

    if self.grad_clip_norm is not None:
        params = self.model.parameters()
        clip_grad_norm(params, self.grad_clip_norm)
    self.optimizer.step()
    self.optimizer.zero_grad()


Trainer._optimizer_update = _trainer_optimizer_update


def trainer_train_epoch(self, dataloader, accumulation_steps=1):
    """
    Train for one epoch through the dataset.
    """

    self.model.training = True
    self.training_mode = True

    # Update scheduler at the start of each epoch so LR is set before training begins
    if self.scheduler is not None:
        current_lr = self.scheduler.get_lr(self.epoch)
        self.optimizer.lr = current_lr
        self.history["learning_rates"].append(current_lr)

    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        accumulated_loss += self._process_batch(inputs, targets, accumulation_steps)

        # Update parameters every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            self._optimizer_update()
            total_loss += accumulated_loss
            accumulated_loss = 0.0
            num_batches += 1
            self.step += 1

    # Handle remaining accumulated gradients
    if accumulated_loss > 0:
        self._optimizer_update()
        total_loss += accumulated_loss
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    self.history["train_loss"].append(avg_loss)

    self.epoch += 1
    return avg_loss


Trainer.train_epoch = trainer_train_epoch


def trainer_evaluate(self, dataloader):
    """
    Evaluate model on dataset without updating parameters.

    """

    self.model.training = False
    self.training_mode = False

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for inputs, targets in dataloader:
        # Forward pass only
        outputs = self.model.forward(inputs)
        loss = self.loss_fn.forward(outputs, targets)

        total_loss += loss.data
        num_batches += 1

        # Calculate accuracy (for classification only).
        # outputs.data.shape[-1] > 1 distinguishes true multi-class (C logits)
        # from regression with a single output neuron (shape (N,1)), which would
        # otherwise enter this branch and produce argmax=0 for every sample.
        if len(outputs.data.shape) > 1 and outputs.data.shape[-1] > 1:  # Multi-class
            predictions = np.argmax(outputs.data, axis=1)
            if len(targets.data.shape) == 1:  # Integer targets
                correct += np.sum(predictions == targets.data)
            else:  # One-hot targets
                correct += np.sum(predictions == np.argmax(targets.data, axis=1))
            total += len(predictions)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    self.history["eval_loss"].append(avg_loss)

    return avg_loss, accuracy


Trainer.evaluate = trainer_evaluate


def trainer_save_checkpoint(self, path: str):
    """
    Save complete training state for resumption.
    """

    checkpoint = {
        "epoch": self.epoch,
        "step": self.step,
        "model_state": self._get_model_state(),
        "optimizer_state": self._get_optimizer_state(),
        "scheduler_state": self._get_scheduler_state(),
        "history": self.history,
        "training_mode": self.training_mode,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


Trainer.save_checkpoint = trainer_save_checkpoint


def trainer_load_checkpoint(self, path: str):
    """
    Load training state from checkpoint.
    """

    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    self.epoch = checkpoint["epoch"]
    self.step = checkpoint["step"]
    self.history = checkpoint["history"]
    self.training_mode = checkpoint["training_mode"]

    # Restore states
    if "model_state" in checkpoint:
        self._set_model_state(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        self._set_optimizer_state(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        self._set_scheduler_state(checkpoint["scheduler_state"])


Trainer.load_checkpoint = trainer_load_checkpoint
