"""
LowMind no_grad — disable gradient tracking for inference
"""
import threading

_local = threading.local()


def _grad_enabled():
    return getattr(_local, 'grad_enabled', True)


def _set_grad_enabled(flag: bool):
    _local.grad_enabled = flag


class no_grad:
    """
    Context manager that disables gradient computation.

    Use during inference to skip building the computational graph,
    which saves memory and speeds up forward passes.

    Example::

        model.eval()
        with lm.no_grad():
            out = model(X)          # no graph built, no memory wasted
            preds = out.data.argmax(axis=1)

    Also works as a decorator::

        @lm.no_grad()
        def predict(model, X):
            return model(X)
    """

    def __enter__(self):
        self._prev = _grad_enabled()
        _set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        _set_grad_enabled(self._prev)

    # Decorator usage
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with no_grad():
                return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper


class enable_grad:
    """
    Context manager that re-enables gradient computation inside a no_grad block.

    Example::

        with lm.no_grad():
            x = model.encoder(X)       # no grad
            with lm.enable_grad():
                loss = criterion(x, y) # grad re-enabled
    """

    def __enter__(self):
        self._prev = _grad_enabled()
        _set_grad_enabled(True)
        return self

    def __exit__(self, *args):
        _set_grad_enabled(self._prev)
