"""
LowMind Trainer — production-ready training loop
Supports: callbacks, gradient clipping, gradient accumulation, mixed-precision hints
"""
import time
import numpy as np
from ..core.tensor import Tensor, clip_grad_norm
from ..core.no_grad import no_grad as _no_grad


class Trainer:
    """
    High-level, production-ready training loop for LowMind models.

    Features:
    - Automatic train / validation phases
    - Gradient clipping
    - Gradient accumulation (simulate large batches on small devices)
    - Callback system (EarlyStopping, ModelCheckpoint, LRScheduler, etc.)
    - Verbose logging

    Args:
        model:            A `Module` instance.
        optimizer:        Optimizer (SGD, Adam, etc.).
        loss_fn:          Loss function: (output, target) → scalar Tensor.
        callbacks:        List of Callback instances.
        clip_grad:        Max gradient norm (0 = off).
        grad_accum_steps: Accumulate gradients over N batches before stepping (default 1).
        verbose:          Print every N epochs (1 = every epoch, 0 = silent).

    Example::

        trainer = lm.Trainer(
            model=model,
            optimizer=lm.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2),
            loss_fn=lm.cross_entropy_loss,
            callbacks=[
                lm.EarlyStopping(patience=10),
                lm.ModelCheckpoint('/tmp/best.lmz'),
            ],
            clip_grad=1.0,
            grad_accum_steps=4,   # effectively 4× the batch size
            verbose=1,
        )
        history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(self, model, optimizer, loss_fn,
                 callbacks=None, clip_grad=0.0,
                 grad_accum_steps=1, verbose=1,
                 auto_tune_batch_size=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.clip_grad = clip_grad
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.verbose = verbose
        self.auto_tune_batch_size = auto_tune_batch_size
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader=None, epochs=10, scheduler=None):
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation (optional).
            epochs:       Number of epochs.
            scheduler:    LR scheduler instance (optional).
                          Called at epoch end. For ReduceLROnPlateau,
                          wrap it in LRSchedulerCallback instead.

        Returns:
            Training history dict.
        """
        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            self.model.train()
            train_loss = self._run_epoch(train_loader, training=True)
            self.history['train_loss'].append(train_loss)

            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self._run_validation(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

            if scheduler is not None and hasattr(scheduler, 'step'):
                if hasattr(scheduler, '_num_bad'):       # ReduceLROnPlateau
                    scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    scheduler.step()

            if self.verbose and epoch % self.verbose == 0:
                elapsed = time.time() - t0
                lr_str = f"lr={self.optimizer.get_lr():.2e}"
                msg = (f"Epoch {epoch:>4}/{epochs} | "
                       f"loss={train_loss:.4f} | {lr_str}")
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
                msg += f" | {elapsed:.1f}s"
                print(msg)

            stop = False
            for cb in self.callbacks:
                if cb.on_epoch_end(epoch, self, train_loss, val_loss):
                    stop = True
            if stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.history

    def evaluate(self, loader):
        """
        Evaluate model on a DataLoader.

        Returns:
            (avg_loss, accuracy) tuple.
        """
        self.model.eval()
        return self._run_validation(loader)

    def predict(self, X):
        """
        Inference on a Tensor or numpy array.

        Returns:
            numpy array of predicted class indices.
        """
        self.model.eval()
        if not isinstance(X, Tensor):
            X = Tensor(X)
        with _no_grad():
            out = self.model(X)
        return out.data.argmax(axis=1)

    def predict_proba(self, X):
        """
        Inference returning raw logit probabilities (after softmax).

        Returns:
            numpy array of shape (N, num_classes).
        """
        self.model.eval()
        if not isinstance(X, Tensor):
            X = Tensor(X)
        with _no_grad():
            out = self.model(X)
        e = np.exp(out.data - out.data.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_epoch(self, loader, training=True):
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        iterator = iter(loader)
        batch_idx = 0

        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            except MemoryError as e:
                if getattr(self, "auto_tune_batch_size", False) and loader.batch_size > 1:
                    print(f"\n⚠️  MemoryError caught: {e}")
                    print(f"   Dynamic Auto-tuning: halving batch_size from {loader.batch_size} to {loader.batch_size // 2} "
                          f"and doubling grad_accum_steps from {self.grad_accum_steps} to {self.grad_accum_steps * 2}")

                    loader.batch_size = max(1, loader.batch_size // 2)
                    self.grad_accum_steps *= 2

                    from ..core.memory import memory_manager
                    import gc
                    memory_manager.clear_cache()
                    gc.collect()

                    iterator = iter(loader)
                    total_loss = 0.0
                    n_batches = 0
                    self.optimizer.zero_grad()
                    batch_idx = 0
                    continue
                else:
                    raise e

            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                X, y = batch[0], batch[1]
            else:
                raise ValueError("DataLoader must yield (X, y) tuples.")

            try:
                out = self.model(X)
                loss = self.loss_fn(out, y)

                if self.grad_accum_steps > 1:
                    scale_factor = 1.0 / self.grad_accum_steps
                    loss_scaled = loss * scale_factor
                    loss_scaled.backward()
                else:
                    loss.backward()
            except MemoryError as e:
                if getattr(self, "auto_tune_batch_size", False) and loader.batch_size > 1:
                    print(f"\n⚠️  MemoryError caught during forward/backward pass: {e}")
                    print(f"   Dynamic Auto-tuning: halving batch_size from {loader.batch_size} to {loader.batch_size // 2} "
                          f"and doubling grad_accum_steps from {self.grad_accum_steps} to {self.grad_accum_steps * 2}")

                    loader.batch_size = max(1, loader.batch_size // 2)
                    self.grad_accum_steps *= 2

                    from ..core.memory import memory_manager
                    import gc
                    memory_manager.clear_cache()
                    gc.collect()

                    iterator = iter(loader)
                    total_loss = 0.0
                    n_batches = 0
                    self.optimizer.zero_grad()
                    batch_idx = 0
                    continue
                else:
                    raise e

            total_loss += float(loss.item())
            n_batches += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.clip_grad > 0:
                    clip_grad_norm(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

            batch_idx += 1

        # Flush any remaining accumulated gradients
        remainder = n_batches % self.grad_accum_steps
        if remainder != 0:
            if self.clip_grad > 0:
                clip_grad_norm(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / max(n_batches, 1)

    def _run_validation(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with _no_grad():
            for batch in loader:
                X, y = batch[0], batch[1]
                out = self.model(X)
                loss = self.loss_fn(out, y)
                total_loss += float(loss.item())
                pred = out.data.argmax(axis=1)
                target = y.data.flatten().astype(int)
                correct += int((pred == target).sum())
                total += len(target)

        n = len(loader) or 1
        return total_loss / n, correct / max(total, 1)
