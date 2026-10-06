"""Learning-rate schedules for the NumPy optimizers.

Schedulers update the learning rate of any doraneural Optimizer object.
"""

import math
from numbers import Integral
from typing import Optional

from .optimizers import Optimizer


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


class LRScheduler:
    """Base class for learning-rate schedulers."""

    requires_metric = False

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
    """Linear warmup from 0 to peak LR, followed by cosine decay.

    During warmup (epoch < warmup_epochs):
        lr = base_lr * (epoch + 1) / warmup_epochs
    During cosine decay (warmup_epochs <= epoch < total_epochs):
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * progress))
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
            return self.base_lr * float(self.last_epoch + 1) / float(self.warmup_epochs)

        if self.last_epoch >= self.total_epochs:
            return self.eta_min

        decay_epochs = self.total_epochs - self.warmup_epochs
        progress = float(self.last_epoch - self.warmup_epochs) / float(decay_epochs)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))


class ReduceLROnPlateau(LRScheduler):
    """Reduce the learning rate when a monitored validation metric stalls.

    The ``step`` method takes the current metric value. In ``mode="min"`` a
    lower value is better; in ``mode="max"`` a higher value is better. When
    used by ``Sequential.fit``, the scheduler automatically monitors val_loss.
    """

    requires_metric = True

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 0.1,
        patience: int = 5,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        mode: str = "min",
    ) -> None:
        super().__init__(optimizer)
        factor = _finite_float("factor", factor)
        threshold = _finite_float("threshold", threshold)
        min_lr = _finite_float("min_lr", min_lr)
        eps = _finite_float("eps", eps)
        if not (0.0 < factor < 1.0):
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        if not isinstance(patience, Integral) or isinstance(patience, bool) or patience < 0:
            raise ValueError(f"patience must be a non-negative integer, got {patience}")
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if not isinstance(cooldown, Integral) or isinstance(cooldown, bool) or cooldown < 0:
            raise ValueError(f"cooldown must be a non-negative integer, got {cooldown}")
        if min_lr < 0.0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")
        if min_lr > self.base_lr:
            raise ValueError(f"min_lr ({min_lr}) cannot exceed the initial learning rate ({self.base_lr})")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.factor = factor
        self.patience = int(patience)
        self.threshold = threshold
        self.cooldown = int(cooldown)
        self.min_lr = min_lr
        self.eps = eps
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.num_reductions = 0

    def get_lr(self) -> float:
        """Return the optimizer's current rate; updates are metric-driven."""
        return float(self.optimizer.lr)

    def step(self, metric: Optional[float] = None) -> float:
        """Observe a metric and reduce the rate after a configurable plateau."""
        if metric is None:
            raise ValueError("ReduceLROnPlateau.step requires a metric value")
        metric_value = _finite_float("metric", metric)

        self.last_epoch += 1
        if self.mode == "min":
            improved = metric_value < self.best - self.threshold
        else:
            improved = metric_value > self.best + self.threshold

        if improved:
            self.best = metric_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            old_lr = float(self.optimizer.lr)
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                self.optimizer.lr = new_lr
                self.num_reductions += 1
                self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self.current_lr = float(self.optimizer.lr)
        return self.current_lr
