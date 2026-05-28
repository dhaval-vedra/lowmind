"""
LowMind Learning Rate Schedulers

Available:
    StepLR               — decay every N steps by a factor
    MultiStepLR          — decay at specific milestones
    ExponentialLR        — exponential decay
    CosineAnnealingLR    — cosine annealing
    ReduceLROnPlateau    — reduce when metric stops improving
    LinearWarmupLR       — linear warmup then constant
    CyclicLR             — triangular cycling
"""
import math


class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.get_lr()
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self.optimizer.set_lr(lr)

    def get_last_lr(self):
        return self.optimizer.get_lr()


class StepLR(_LRScheduler):
    """
    Decays LR by *gamma* every *step_size* epochs.

    Args:
        optimizer:  Optimizer instance.
        step_size:  Decay period in epochs.
        gamma:      Multiplicative factor (default 0.1).

    Example::

        scheduler = lm.StepLR(optimizer, step_size=10, gamma=0.5)
        for epoch in range(epochs):
            train(...)
            scheduler.step()
    """

    def __init__(self, optimizer, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.optimizer.get_lr()
        return self.optimizer.get_lr() * self.gamma


class MultiStepLR(_LRScheduler):
    """
    Decays LR by *gamma* at each milestone epoch.

    Args:
        optimizer:   Optimizer instance.
        milestones:  List of epoch indices (sorted ascending).
        gamma:       Multiplicative factor (default 0.1).
    """

    def __init__(self, optimizer, milestones, gamma=0.1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            return self.optimizer.get_lr() * self.gamma
        return self.optimizer.get_lr()


class ExponentialLR(_LRScheduler):
    """Decays LR by *gamma* every epoch: lr = base_lr * gamma^epoch."""

    def __init__(self, optimizer, gamma):
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing between *eta_max* (base lr) and *eta_min* over *T_max* epochs.

    Args:
        optimizer: Optimizer instance.
        T_max:     Half-period of the cosine cycle.
        eta_min:   Minimum learning rate (default 0).
    """

    def __init__(self, optimizer, T_max, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        return (self.eta_min +
                (self.base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2)


class ReduceLROnPlateau:
    """
    Reduce LR when a metric has stopped improving.

    Args:
        optimizer:  Optimizer instance.
        mode:       'min' or 'max' (default 'min').
        factor:     Factor to reduce LR (default 0.1).
        patience:   Epochs with no improvement before reducing (default 10).
        min_lr:     Minimum LR (default 1e-8).
        threshold:  Min relative change to qualify as improvement (default 1e-4).
        verbose:    Print message on LR change (default False).

    Example::

        scheduler = lm.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        for epoch in range(epochs):
            val_loss = validate(...)
            scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 min_lr=1e-8, threshold=1e-4, verbose=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.verbose = verbose
        self._best = float('inf') if mode == 'min' else float('-inf')
        self._num_bad = 0

    def step(self, metric):
        if self.mode == 'min':
            is_better = metric < self._best * (1 - self.threshold)
        else:
            is_better = metric > self._best * (1 + self.threshold)

        if is_better:
            self._best = metric
            self._num_bad = 0
        else:
            self._num_bad += 1

        if self._num_bad >= self.patience:
            old_lr = self.optimizer.get_lr()
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.optimizer.set_lr(new_lr)
            self._num_bad = 0
            if self.verbose:
                print(f"ReduceLROnPlateau: lr {old_lr:.2e} → {new_lr:.2e}")


class LinearWarmupLR(_LRScheduler):
    """
    Linear warmup for *warmup_steps* then constant LR.

    Args:
        optimizer:     Optimizer instance.
        warmup_steps:  Number of warmup steps.
        target_lr:     Target LR after warmup (defaults to optimizer's current lr).
    """

    def __init__(self, optimizer, warmup_steps, target_lr=None):
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr or optimizer.get_lr()
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.target_lr * (self.last_epoch + 1) / self.warmup_steps
        return self.target_lr


class CyclicLR:
    """
    Triangular / cyclical learning rate.

    Args:
        optimizer:    Optimizer instance.
        base_lr:      Minimum LR.
        max_lr:       Maximum LR.
        step_size:    Half-cycle length in steps.
        mode:         'triangular', 'triangular2', or 'exp_range'.
        gamma:        Factor for exp_range mode (default 1.0).
    """

    def __init__(self, optimizer, base_lr, max_lr, step_size=2000,
                 mode='triangular', gamma=1.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self._step = 0

    def step(self):
        self._step += 1
        cycle = math.floor(1 + self._step / (2 * self.step_size))
        x = abs(self._step / self.step_size - 2 * cycle + 1)
        scale = max(0, 1 - x)

        if self.mode == 'triangular2':
            scale /= 2 ** (cycle - 1)
        elif self.mode == 'exp_range':
            scale *= self.gamma ** self._step

        lr = self.base_lr + (self.max_lr - self.base_lr) * scale
        self.optimizer.set_lr(lr)
