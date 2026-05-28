"""
LowMind Training Callbacks

EarlyStopping       — stop when val_loss stops improving
ModelCheckpoint     — save best model weights
LRSchedulerCallback — wrap any scheduler
History             — record metrics each epoch
"""
import numpy as np


class Callback:
    """Base callback interface."""

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, epoch, trainer, train_loss, val_loss=None):
        """Return True to stop training."""
        return False

    def on_train_end(self, trainer):
        pass


class EarlyStopping(Callback):
    """
    Stop training when validation loss stops improving.

    Args:
        patience:  Epochs to wait after last improvement (default 5).
        min_delta: Minimum improvement to count (default 0).
        mode:      'min' (lower is better) or 'max' (default 'min').
        verbose:   Print message when triggered (default True).

    Example::

        callbacks = [lm.EarlyStopping(patience=10, verbose=True)]
        trainer.fit(train_loader, val_loader, epochs=200, callbacks=callbacks)
    """

    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self._best = float('inf') if mode == 'min' else float('-inf')
        self._counter = 0

    def on_train_begin(self, trainer):
        self._best = float('inf') if self.mode == 'min' else float('-inf')
        self._counter = 0

    def on_epoch_end(self, epoch, trainer, train_loss, val_loss=None):
        metric = val_loss if val_loss is not None else train_loss
        if metric is None:
            return False

        if self.mode == 'min':
            improved = metric < self._best - self.min_delta
        else:
            improved = metric > self._best + self.min_delta

        if improved:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                if self.verbose:
                    print(f"EarlyStopping: no improvement for {self.patience} epochs (best={self._best:.4f})")
                return True
        return False


class ModelCheckpoint(Callback):
    """
    Save the model whenever validation loss improves.

    Args:
        filepath:   Path to save the model file.
        monitor:    Metric to monitor: 'val_loss' or 'train_loss' (default 'val_loss').
        mode:       'min' or 'max' (default 'min').
        verbose:    Print message on save (default True).
        save_best_only: Only save when improved (default True).

    Example::

        callbacks = [lm.ModelCheckpoint('best_model.lmz')]
        trainer.fit(train_loader, val_loader, epochs=100, callbacks=callbacks)
    """

    def __init__(self, filepath, monitor='val_loss', mode='min',
                 verbose=True, save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.save_best_only = save_best_only
        self._best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, trainer, train_loss, val_loss=None):
        metric = val_loss if self.monitor == 'val_loss' else train_loss
        if metric is None:
            return False

        improved = (metric < self._best) if self.mode == 'min' else (metric > self._best)
        if improved or not self.save_best_only:
            self._best = metric
            trainer.model.save(self.filepath)
            if self.verbose:
                print(f"ModelCheckpoint: saved model (epoch={epoch}, {self.monitor}={metric:.4f})")
        return False


class LRSchedulerCallback(Callback):
    """
    Wrap a scheduler to call .step() at the end of each epoch.

    Args:
        scheduler: A scheduler instance (StepLR, ReduceLROnPlateau, etc.).
        monitor:   For ReduceLROnPlateau — 'val_loss' or 'train_loss'.
    """

    def __init__(self, scheduler, monitor='val_loss'):
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch, trainer, train_loss, val_loss=None):
        if hasattr(self.scheduler, '_num_bad'):
            # ReduceLROnPlateau
            metric = val_loss if self.monitor == 'val_loss' else train_loss
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
        return False


class History(Callback):
    """
    Record per-epoch metrics into a dictionary.

    Access via `callback.history` after training.
    """

    def __init__(self):
        self.history = {}

    def on_epoch_end(self, epoch, trainer, train_loss, val_loss=None):
        self.history.setdefault('epoch', []).append(epoch)
        self.history.setdefault('train_loss', []).append(train_loss)
        if val_loss is not None:
            self.history.setdefault('val_loss', []).append(val_loss)
        return False
