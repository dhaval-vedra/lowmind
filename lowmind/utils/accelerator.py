"""
LowMind Core Bottleneck Accelerator — JIT and stride-based C-level performance booster
"""
import numpy as np

try:
    from numba import jit as _jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def _jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@_jit(nopython=True, cache=True, fastmath=True)
def fast_col2im_numba(col_rs, x_padded, kH, kW, sH, sW, out_H, out_W):
    """JIT compiled fast col2im mapping."""
    for i in range(kH):
        for j in range(kW):
            x_padded[:, :, i:i + sH * out_H:sH, j:j + sW * out_W:sW] += col_rs[:, :, i, j, :, :]
    return x_padded


def col2im_optimized(col, x_shape, kH, kW, sH, sW, padding):
    """Highly optimized col2im implementation with optional JIT acceleration."""
    N, C, H, W = x_shape
    pH, pW = padding
    H_padded, W_padded = H + 2 * pH, W + 2 * pW
    out_H = (H_padded - kH) // sH + 1
    out_W = (W_padded - kW) // sW + 1
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=np.float32)
    col_rs = col.reshape(N, C, kH, kW, out_H, out_W)

    if _NUMBA_AVAILABLE:
        try:
            x_padded = fast_col2im_numba(col_rs, x_padded, kH, kW, sH, sW, out_H, out_W)
        except Exception:
            # Fallback
            for i in range(kH):
                for j in range(kW):
                    x_padded[:, :, i:i + sH * out_H:sH, j:j + sW * out_W:sW] += col_rs[:, :, i, j, :, :]
    else:
        for i in range(kH):
            for j in range(kW):
                x_padded[:, :, i:i + sH * out_H:sH, j:j + sW * out_W:sW] += col_rs[:, :, i, j, :, :]

    return x_padded[:, :, pH:pH + H, pW:pW + W] if (pH > 0 or pW > 0) else x_padded


def is_jit_accelerated():
    """Returns True if Numba compilation is active."""
    return _NUMBA_AVAILABLE
