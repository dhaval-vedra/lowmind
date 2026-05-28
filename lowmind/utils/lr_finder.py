"""
LowMind LR Finder — automatically find the best learning rate
Based on the paper: "Cyclical Learning Rates for Training Neural Networks" (Smith 2015)
"""
import numpy as np
import copy


class LRFinder:
    """
    Find the optimal learning rate by exponentially increasing it over one pass
    and recording the loss at each step.

    Args:
        model:     A lm.Module.
        optimizer: An optimizer instance.
        loss_fn:   Loss function (output, target) -> Tensor.

    Example::

        finder = lm.LRFinder(model, optimizer, lm.cross_entropy_loss)
        finder.find(train_loader, start_lr=1e-7, end_lr=1.0, n_iter=100)
        finder.print_results()
        best_lr = finder.suggested_lr()
    """

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {'lr': [], 'loss': [], 'smooth_loss': []}
        self._best_state = None
        self._orig_lr = optimizer.get_lr()

    def find(self, loader, start_lr=1e-7, end_lr=1.0, n_iter=100,
             beta=0.98, diverge_factor=5.0):
        """
        Run the learning rate range test.

        Args:
            loader:         DataLoader to iterate over.
            start_lr:       Starting learning rate.
            end_lr:         Ending learning rate.
            n_iter:         Number of iterations (steps).
            beta:           Smoothing factor for loss (default 0.98).
            diverge_factor: Stop if loss exceeds best * factor (default 5).
        """
        # Save model & optimizer state
        self._best_state = {
            'state_dict': self.model.state_dict(),
            'orig_lr': self._orig_lr,
        }

        mul = (end_lr / start_lr) ** (1.0 / (n_iter - 1))
        lr = start_lr
        self.optimizer.set_lr(lr)

        avg_loss = 0.0
        best_loss = float('inf')
        step = 0

        self.history = {'lr': [], 'loss': [], 'smooth_loss': []}
        self.model.train()

        loader_iter = iter(loader)
        while step < n_iter:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            X, y = batch[0], batch[1]

            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.loss_fn(out, y)
            loss_val = float(loss.item())

            # Smooth the loss
            avg_loss = beta * avg_loss + (1 - beta) * loss_val
            smooth = avg_loss / (1 - beta ** (step + 1))

            self.history['lr'].append(lr)
            self.history['loss'].append(loss_val)
            self.history['smooth_loss'].append(smooth)

            if smooth < best_loss:
                best_loss = smooth

            if step > 0 and smooth > diverge_factor * best_loss:
                print(f"  LR Finder: loss diverged at lr={lr:.2e}. Stopping.")
                break

            loss.backward()
            self.optimizer.step()

            lr *= mul
            self.optimizer.set_lr(lr)
            step += 1

        # Restore model weights and original lr
        self.model.load_state_dict(self._best_state['state_dict'])
        self.optimizer.set_lr(self._best_state['orig_lr'])

    def suggested_lr(self):
        """
        Return the learning rate with the steepest loss decrease
        (in the middle of the good range, not the minimum loss point).

        Returns:
            Float — suggested learning rate.
        """
        losses = np.array(self.history['smooth_loss'])
        lrs = np.array(self.history['lr'])
        if len(losses) < 3:
            return lrs[0] if len(lrs) > 0 else 1e-3

        # Find steepest negative gradient
        grads = np.gradient(losses)
        best_idx = int(np.argmin(grads[1:-1])) + 1   # skip edges
        return float(lrs[best_idx])

    def print_results(self, n_cols=60):
        """Print an ASCII chart of loss vs learning rate."""
        lrs = self.history['lr']
        losses = self.history['smooth_loss']
        if not lrs:
            print("  No results yet — run .find() first.")
            return

        print("\n  LR Finder Results")
        print("  " + "-" * n_cols)
        max_loss = max(losses) + 1e-8
        min_loss = min(losses) - 1e-8

        step = max(1, len(lrs) // 20)
        for i in range(0, len(lrs), step):
            lr = lrs[i]
            loss = losses[i]
            bar_len = int((1 - (loss - min_loss) / (max_loss - min_loss)) * (n_cols - 18))
            bar = "#" * max(0, bar_len)
            print(f"  {lr:.2e}  {loss:6.4f}  |{bar}")

        suggested = self.suggested_lr()
        print(f"\n  Suggested LR: {suggested:.2e}")
        print(f"  (Use 1/10 of this for safe convergence: {suggested/10:.2e})")
