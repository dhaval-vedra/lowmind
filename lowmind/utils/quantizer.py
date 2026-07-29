"""
LowMind Quantizer — Post-Training Integer (INT8) Quantization utilities
"""
import numpy as np
from ..core.tensor import Tensor


class QuantizedTensor:
    """
    Represents a quantized tensor storing data as int8 with a scale factor.
    """
    def __init__(self, data, scale):
        if isinstance(data, Tensor):
            data = data.numpy()
        self.q_data = data.astype(np.int8)
        self.scale = float(scale)

    def dequantize(self):
        """Convert back to float32."""
        return self.q_data.astype(np.float32) * self.scale


def quantize_weight(tensor):
    """
    Quantize a weight Tensor's data to INT8.
    Returns (quantized_data_int8, scale_factor).
    """
    weight_data = tensor.numpy() if isinstance(tensor, Tensor) else np.array(tensor, dtype=np.float32)
    max_val = np.max(np.abs(weight_data))
    if max_val == 0:
        scale = 1.0
    else:
        scale = float(max_val / 127.0)

    q_data = np.round(weight_data / scale).astype(np.int8)
    return q_data, scale


def quantize_model(model):
    """
    In-place Post-Training Quantization of model weights to simulate INT8 weights.
    Also returns a dictionary of scale factors.
    """
    scales = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            q_data, scale = quantize_weight(param)
            # Simulate INT8 weight in-place by dequantizing back to float32
            # This is standard Post-Training Quantization simulation (Fake Quantization)
            param.data = q_data.astype(np.float32) * scale
            scales[name] = scale
    return scales


def fake_quantize(tensor):
    """
    Fake quantize a Tensor in the forward pass using the Straight-Through Estimator (STE).
    Supports Quantization Aware Training (QAT).
    """
    max_val = np.max(np.abs(tensor.data))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q_data = np.round(tensor.data / scale).astype(np.float32) * scale

    out = Tensor(q_data, requires_grad=tensor.requires_grad, _children=(tensor,), _op='fake_quantize')

    def _backward():
        if tensor.requires_grad:
            tensor._ensure_grad()
            # STE (Straight-Through Estimator): pass gradient directly
            tensor.grad += out.grad

    out._backward = _backward
    return out


def prepare_qat(model, enabled=True):
    """
    Enable or disable Quantization Aware Training (QAT) on all layers of a model.
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            module.qat_mode = enabled
