"""
LowMind Weight Initialization Utilities

Functions:
    xavier_uniform_   — Xavier / Glorot uniform (default for Linear)
    xavier_normal_    — Xavier / Glorot normal
    kaiming_uniform_  — He uniform (default for Conv, ReLU nets)
    kaiming_normal_   — He normal
    orthogonal_       — Orthogonal matrix (good for RNNs)
    normal_           — Normal distribution
    uniform_          — Uniform distribution
    constant_         — Fill with a constant
    zeros_            — Fill with zeros
    ones_             — Fill with ones
    eye_              — Identity matrix
    init_module       — Initialize all params of a Module at once
"""
import numpy as np
from ..core.tensor import Tensor
from ..core.module import Module


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Xavier / Glorot uniform initialization.

    Fills *tensor* with values sampled from U(-a, a) where::

        a = gain * sqrt(6 / (fan_in + fan_out))

    Best for: tanh / sigmoid activations.

    Args:
        tensor: Tensor to initialize (in-place).
        gain:   Scaling factor (default 1.0).

    Returns:
        The modified tensor.
    """
    fan_in, fan_out = _compute_fans(tensor)
    a = gain * np.sqrt(6.0 / (fan_in + fan_out))
    tensor.data[:] = np.random.uniform(-a, a, tensor.data.shape).astype(np.float32)
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Xavier / Glorot normal initialization.

    Fills *tensor* with values sampled from N(0, std²) where::

        std = gain * sqrt(2 / (fan_in + fan_out))

    Args:
        tensor: Tensor to initialize (in-place).
        gain:   Scaling factor (default 1.0).
    """
    fan_in, fan_out = _compute_fans(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data[:] = (np.random.randn(*tensor.data.shape) * std).astype(np.float32)
    return tensor


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = 'fan_in',
                     nonlinearity: str = 'relu') -> Tensor:
    """
    He / Kaiming uniform initialization.

    Fills *tensor* with values from U(-bound, bound) where::

        bound = sqrt(3) * sqrt(2 / (1 + a²) / fan)

    Best for: ReLU / LeakyReLU activations.

    Args:
        tensor:       Tensor to initialize (in-place).
        a:            Negative slope for LeakyReLU (default 0 = ReLU).
        mode:         'fan_in' (default) or 'fan_out'.
        nonlinearity: 'relu' or 'leaky_relu'.
    """
    fan = _compute_fans(tensor)[0] if mode == 'fan_in' else _compute_fans(tensor)[1]
    gain = _calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    tensor.data[:] = np.random.uniform(-bound, bound, tensor.data.shape).astype(np.float32)
    return tensor


def kaiming_normal_(tensor: Tensor, a: float = 0, mode: str = 'fan_in',
                    nonlinearity: str = 'relu') -> Tensor:
    """
    He / Kaiming normal initialization.

    Fills *tensor* with values from N(0, std²) where::

        std = gain / sqrt(fan)

    Best for: ReLU / LeakyReLU activations.

    Args:
        tensor:       Tensor to initialize (in-place).
        a:            Negative slope for LeakyReLU (default 0 = ReLU).
        mode:         'fan_in' (default) or 'fan_out'.
        nonlinearity: 'relu' or 'leaky_relu'.
    """
    fan = _compute_fans(tensor)[0] if mode == 'fan_in' else _compute_fans(tensor)[1]
    gain = _calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    tensor.data[:] = (np.random.randn(*tensor.data.shape) * std).astype(np.float32)
    return tensor


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Orthogonal matrix initialization (Saxe et al. 2013).

    Fills *tensor* with a (semi-)orthogonal matrix. Excellent for RNNs
    as it preserves gradient norms through time.

    Args:
        tensor: 2-D or higher tensor. The last two dims form the matrix.
        gain:   Scaling factor (default 1.0).
    """
    shape = tensor.data.shape
    if tensor.data.ndim < 2:
        raise ValueError("orthogonal_ requires tensor with at least 2 dims")
    rows = shape[0]
    cols = int(np.prod(shape[1:]))
    # Use SVD: always gives correctly shaped orthogonal factor
    flat = np.random.randn(rows, cols)
    U, _, Vh = np.linalg.svd(flat, full_matrices=False)
    # U: (rows, k), Vh: (k, cols), k = min(rows, cols)
    Q = U if rows >= cols else Vh   # shape is always (rows, cols)
    tensor.data[:] = (gain * Q.reshape(shape)).astype(np.float32)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor with values from N(mean, std²)."""
    tensor.data[:] = (np.random.randn(*tensor.data.shape) * std + mean).astype(np.float32)
    return tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fill tensor with values from U(a, b)."""
    tensor.data[:] = np.random.uniform(a, b, tensor.data.shape).astype(np.float32)
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fill tensor with a constant value."""
    tensor.data.fill(val)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """Fill tensor with zeros."""
    return constant_(tensor, 0.0)


def ones_(tensor: Tensor) -> Tensor:
    """Fill tensor with ones."""
    return constant_(tensor, 1.0)


def eye_(tensor: Tensor) -> Tensor:
    """Fill 2-D tensor with the identity matrix."""
    if tensor.data.ndim != 2:
        raise ValueError("eye_ requires a 2-D tensor")
    tensor.data[:] = np.eye(*tensor.data.shape, dtype=np.float32)
    return tensor


def init_module(module: Module, weight_init='kaiming_uniform',
                bias_init='zeros') -> Module:
    """
    Initialize all Linear and Conv2d parameters in a module.

    Args:
        module:      Any lm.Module.
        weight_init: One of 'xavier_uniform', 'xavier_normal',
                     'kaiming_uniform', 'kaiming_normal', 'orthogonal',
                     or a callable (tensor) -> None.
        bias_init:   One of 'zeros', 'ones', or a callable.

    Returns:
        The module (in-place initialization).

    Example::

        model = lm.Sequential(lm.Linear(128, 64), lm.ReLU(), lm.Linear(64, 10))
        lm.init_module(model, weight_init='xavier_uniform', bias_init='zeros')
    """
    _WEIGHT_INITS = {
        'xavier_uniform': xavier_uniform_,
        'xavier_normal': xavier_normal_,
        'kaiming_uniform': kaiming_uniform_,
        'kaiming_normal': kaiming_normal_,
        'orthogonal': orthogonal_,
    }
    _BIAS_INITS = {
        'zeros': zeros_,
        'ones': ones_,
    }

    w_fn = _WEIGHT_INITS.get(weight_init, weight_init)
    b_fn = _BIAS_INITS.get(bias_init, bias_init)

    for mod in module.modules():
        if hasattr(mod, 'weight') and isinstance(mod.weight, Tensor):
            w_fn(mod.weight)
        if hasattr(mod, 'bias') and isinstance(mod.bias, Tensor):
            b_fn(mod.bias)

    return module


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_fans(tensor: Tensor):
    """Return (fan_in, fan_out) for a weight tensor."""
    shape = tensor.data.shape
    if len(shape) < 2:
        return shape[0], shape[0]
    if len(shape) == 2:   # Linear
        return shape[1], shape[0]
    # Conv weights: (out_ch, in_ch, kH, kW)
    receptive = int(np.prod(shape[2:]))
    fan_in = shape[1] * receptive
    fan_out = shape[0] * receptive
    return fan_in, fan_out


def _calculate_gain(nonlinearity: str, param=None) -> float:
    GAINS = {
        'sigmoid': 1.0,
        'tanh': 5.0 / 3,
        'relu': np.sqrt(2.0),
        'leaky_relu': np.sqrt(2.0 / (1 + (param or 0.01) ** 2)),
        'linear': 1.0,
        'selu': 3.0 / 4,
    }
    return GAINS.get(nonlinearity, 1.0)
