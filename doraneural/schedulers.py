"""Learning rate schedulers for dynamic learning rate adjustment during training.

Provides standard step decay, cosine annealing, and linear warmup cosine schedules
that update the learning rate of any doraneural Optimizer object.
"""

import math
from typing import Optional
from .optimizers import Optimizer


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"Expected doraneural Optimizer, got {type(optimizer)}")
        self.optimizer: Optimizer = optimizer
        self.base_lr: float = float(optimizer.lr)
        self.last_epoch: int = last_epoch
        self.current_lr: float = self.base_lr

    def get_lr(self) -> float:
        """Compute the learning rate for current epoch."""
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> float:
        """Step the scheduler and update the optimizer's learning rate."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self.current_lr = self.get_lr()
        self.optimizer.lr = self.current_lr
        return self.current_lr


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs.

    Formula:
        lr = base_lr * (gamma ** (epoch // step_size))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch)
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size: int = step_size
        self.gamma: float = float(gamma)

    def get_lr(self) -> float:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return self.base_lr * factor


class CosineAnnealingLR(LRScheduler):
    """Cosine Annealing learning rate schedule.

    Formula:
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch)
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max: int = T_max
        self.eta_min: float = float(eta_min)

    def get_lr(self) -> float:
        if self.last_epoch >= self.T_max:
            return self.eta_min
        fraction = self.last_epoch / self.T_max
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * fraction))


class WarmupCosineLR(LRScheduler):
    """Linear warmup from 0 to peak LR, followed by Cosine Annealing decay.

    During warmup (epoch < warmup_epochs):
        lr = base_lr * (epoch + 1) / warmup_epochs
    During cosine decay (warmup_epochs <= epoch < total_epochs):
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * (epoch - warmup) / (total - warmup)))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch)
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if total_epochs <= warmup_epochs:
            raise ValueError(f"total_epochs ({total_epochs}) must be > warmup_epochs ({warmup_epochs})")

        self.warmup_epochs: int = warmup_epochs
        self.total_epochs: int = total_epochs
        self.eta_min: float = float(eta_min)

    def get_lr(self) -> float:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * float(self.last_epoch + 1) / float(self.warmup_epochs)

        if self.last_epoch >= self.total_epochs:
            return self.eta_min

        decay_epochs = self.total_epochs - self.warmup_epochs
        progress = float(self.last_epoch - self.warmup_epochs) / float(decay_epochs)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))
