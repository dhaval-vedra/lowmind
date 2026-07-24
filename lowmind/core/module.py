"""
LowMind Module — base class for all neural network modules
"""
import pickle
import gzip
from collections import OrderedDict
from .tensor import Tensor


class Module:
    """
    Base class for all LowMind neural network modules.

    Subclass this and implement `forward(self, x)`.

    Example::

        class MLP(lm.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = lm.Linear(128, 64)
                self.fc2 = lm.Linear(64, 10)

            def forward(self, x):
                return self.fc2(self.fc1(x).relu())
    """

    def __init__(self):
        self._parameters: OrderedDict = OrderedDict()
        self._modules: OrderedDict = OrderedDict()
        self._buffers: OrderedDict = OrderedDict()
        self.training: bool = True

    # ------------------------------------------------------------------
    # Attribute routing: auto-register Tensors (params) and Modules
    # ------------------------------------------------------------------

    def __setattr__(self, name, value):
        # Clean up old registrations if overwriting
        for collection in ('_parameters', '_modules', '_buffers'):
            if hasattr(self, collection) and name in getattr(self, collection, {}):
                getattr(self, collection).pop(name)

        if isinstance(value, Tensor) and value.requires_grad:
            if not hasattr(self, '_parameters'):
                super().__setattr__('_parameters', OrderedDict())
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, '_modules'):
                super().__setattr__('_modules', OrderedDict())
            self._modules[name] = value
        super().__setattr__(name, value)

    def __getattr__(self, name):
        # Fallback: look up in parameters / modules / buffers
        for collection in ('_parameters', '_modules', '_buffers'):
            try:
                d = super().__getattribute__(collection)
                if name in d:
                    return d[name]
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ------------------------------------------------------------------
    # Parameter iteration
    # ------------------------------------------------------------------

    def parameters(self):
        """Yield all learnable parameters (recursively)."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=''):
        for name, param in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name), param
        for mod_name, module in self._modules.items():
            sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(prefix=sub_prefix)

    def modules(self):
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix=''):
        yield prefix or 'root', self
        for name, module in self._modules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(prefix=sub_prefix)

    # ------------------------------------------------------------------
    # Training / eval mode
    # ------------------------------------------------------------------

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Parameter count / summary
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only=True):
        """Return total parameter count."""
        total = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            total += p.data.size
        return total

    def quantize(self):
        """
        Quantize the model weights in-place to simulate INT8 precision.
        """
        import numpy as np
        for name, param in self.named_parameters():
            if "weight" in name:
                max_val = np.max(np.abs(param.data))
                scale = max_val / 127.0 if max_val > 0 else 1.0
                q_data = np.round(param.data / scale).astype(np.int8)
                param.data = q_data.astype(np.float32) * scale
        print("Model quantized to INT8 simulation successfully!")
        return self

    def summary(self):
        """Print a short architecture summary."""
        print(f"{'Module':<30} {'Output Shape':<20} {'Params':>10}")
        print("-" * 62)
        total = 0
        for name, module in self.named_modules():
            params = sum(p.data.size for p in module._parameters.values())
            total += params
            if params > 0 or name == 'root':
                print(f"{name:<30} {'?':<20} {params:>10,}")
        print("-" * 62)
        print(f"{'Total trainable parameters':<50} {total:>10,}")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _named_buffers(self, prefix=''):
        """
        Yield (name, array) for numpy-array attributes that are NOT parameters
        but need to be saved/restored (e.g. BatchNorm running_mean/running_var).
        """
        import numpy as np
        _BUFFER_NAMES = ('running_mean', 'running_var', 'num_batches_tracked')
        for attr in _BUFFER_NAMES:
            val = getattr(self, attr, None)
            if val is not None and isinstance(val, np.ndarray):
                full_name = f"{prefix}.{attr}" if prefix else attr
                yield full_name, val
        for mod_name, mod in self._modules.items():
            child_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from mod._named_buffers(prefix=child_prefix)

    def state_dict(self):
        """
        Return a dict mapping parameter and buffer names to numpy arrays.

        Includes both trainable parameters and non-trainable buffers like
        BatchNorm running statistics.
        """
        sd = {name: param.data.copy() for name, param in self.named_parameters()}
        for name, buf in self._named_buffers():
            sd[f"__buf__{name}"] = buf.copy()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Load parameters and buffers from a state dict."""
        import numpy as np
        # Split into params and buffers
        buf_prefix = '__buf__'
        param_sd = {k: v for k, v in state_dict.items() if not k.startswith(buf_prefix)}
        buf_sd   = {k[len(buf_prefix):]: v for k, v in state_dict.items() if k.startswith(buf_prefix)}

        current = dict(self.named_parameters())
        for name, data in param_sd.items():
            if name in current:
                current[name].data[:] = data
            elif strict:
                raise KeyError(f"Unexpected key in state_dict: '{name}'")
        if strict:
            missing = set(current.keys()) - set(param_sd.keys())
            if missing:
                raise KeyError(f"Missing keys in state_dict: {missing}")

        # Restore buffers (best-effort, not strict)
        current_bufs = dict(self._named_buffers())
        for name, data in buf_sd.items():
            if name in current_bufs:
                current_bufs[name][:] = data

    def save(self, path, compress=True):
        """
        Save model weights to a file.

        Args:
            path: File path (suggested extension: .lm or .lmz for compressed).
            compress: Use gzip compression (default True).
        """
        sd = self.state_dict()
        if compress:
            with gzip.open(path, 'wb') as f:
                pickle.dump(sd, f, protocol=4)
        else:
            with open(path, 'wb') as f:
                pickle.dump(sd, f, protocol=4)
        print(f"Model saved → {path}")

    def load(self, path):
        """
        Load model weights from a file saved with `save()`.

        Args:
            path: Path to the saved file.
        """
        try:
            with gzip.open(path, 'rb') as f:
                sd = pickle.load(f)
        except (OSError, gzip.BadGzipFile):
            with open(path, 'rb') as f:
                sd = pickle.load(f)
        self.load_state_dict(sd)
        print(f"Model loaded ← {path}")

    def __repr__(self):
        lines = [f"{type(self).__name__}("]
        for name, module in self._modules.items():
            mod_str = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines) if len(self._modules) > 0 else f"{type(self).__name__}()"
