"""
LowMind Neural Network Layers
Includes: Linear, Conv2d (with proper backward), BatchNorm1d, BatchNorm2d,
          MaxPool2d, AvgPool2d, Flatten, Embedding, Dropout
"""
import numpy as np
from ..core.tensor import Tensor
from ..core.module import Module


class Linear(Module):
    """
    Fully-connected (dense) layer: y = x @ W.T + b

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If False, no bias term (default True).

    Shape:
        Input:  (*, in_features)
        Output: (*, out_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier / Glorot uniform initialisation
        scale = np.sqrt(6.0 / (in_features + out_features))
        w = np.random.uniform(-scale, scale, (out_features, in_features)).astype(np.float32)
        self.weight = Tensor(w, requires_grad=True)

        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self._bias_none = True  # keep attr out of __setattr__ routing

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if hasattr(self, 'bias') and self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        has_bias = hasattr(self, 'bias')
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={has_bias})"


class Conv2d(Module):
    """
    2-D Convolutional layer with proper gradient computation.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output (filter) channels.
        kernel_size:  int or (kH, kW).
        stride:       int or (sH, sW).
        padding:      int or (pH, pW).
        bias:         Add bias (default True).

    Shape:
        Input:  (N, C_in, H, W)
        Output: (N, C_out, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

        # He (Kaiming) uniform initialisation
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = np.sqrt(2.0 / fan_in)
        w = (np.random.randn(out_channels, in_channels, *self.kernel_size) * scale).astype(np.float32)
        self.weight = Tensor(w, requires_grad=True)

        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
        else:
            self._bias_none = True

    @staticmethod
    def _pad(x_data, padding):
        ph, pw = padding
        if ph == 0 and pw == 0:
            return x_data
        return np.pad(x_data, ((0, 0), (0, 0), (ph, ph), (pw, pw)))

    @staticmethod
    def _im2col(x_padded, kH, kW, sH, sW, out_H, out_W):
        """Convert input patches to columns for efficient convolution (highly optimized stride tricks)."""
        N, C, H, W = x_padded.shape
        shape = (N, C, out_H, out_W, kH, kW)
        strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            x_padded.strides[2] * sH,
            x_padded.strides[3] * sW,
            x_padded.strides[2],
            x_padded.strides[3]
        )
        cols = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        return cols.transpose(0, 1, 4, 5, 2, 3).reshape(N, C * kH * kW, out_H * out_W).copy()

    @staticmethod
    def _col2im(col, x_shape, kH, kW, sH, sW, padding):
        from ..utils.accelerator import col2im_optimized
        return col2im_optimized(col, x_shape, kH, kW, sH, sW, padding)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        out_H = (H + 2 * pH - kH) // sH + 1
        out_W = (W + 2 * pW - kW) // sW + 1

        x_padded = self._pad(x.data, self.padding)
        col = self._im2col(x_padded, kH, kW, sH, sW, out_H, out_W)

        W_flat = self.weight.data.reshape(self.out_channels, -1)
        out_data = (W_flat @ col).reshape(N, self.out_channels, out_H, out_W)

        if hasattr(self, 'bias'):
            out_data = out_data + self.bias.data.reshape(1, -1, 1, 1)

        requires_grad = x.requires_grad or self.weight.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                     _children=(x, self.weight), _op='conv2d')

        def _backward():
            if out.grad is None:
                return
            dout = out.grad  # (N, C_out, out_H, out_W)

            if hasattr(self, 'bias') and self.bias.requires_grad:
                self.bias._ensure_grad()
                self.bias.grad += dout.sum(axis=(0, 2, 3))

            if self.weight.requires_grad:
                self.weight._ensure_grad()
                dout_flat = dout.reshape(self.out_channels, -1)  # (C_out, N*out_H*out_W)
                # col is (N, C_in*kH*kW, out_H*out_W) -> (C_in*kH*kW, N*out_H*out_W)
                col_T = col.transpose(1, 0, 2).reshape(C * kH * kW, -1)
                self.weight.grad += (dout_flat @ col_T.T).reshape(self.weight.data.shape)

            if x.requires_grad:
                x._ensure_grad()
                dout_col = W_flat.T @ dout.reshape(self.out_channels, -1)  # (C*kH*kW, N*out_H*out_W)
                dout_col = dout_col.reshape(N, C * kH * kW, out_H * out_W)
                dx = self._col2im(dout_col, x.data.shape, kH, kW, sH, sW, self.padding)
                x.grad += dx

        out._backward = _backward
        return out

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class BatchNorm1d(Module):
    """
    Batch Normalisation for 2-D inputs (N, features).

    Args:
        num_features: Number of features.
        eps:          Numerical stability (default 1e-5).
        momentum:     Running stat update factor (default 0.1).
        affine:       Learnable scale + shift (default True).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)

        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = x_norm.copy()

        if self.affine:
            out_data = out_data * self.gamma.data + self.beta.data

        requires_grad = x.requires_grad or (self.affine and self.gamma.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad,
                     _children=(x,), _op='bn1d')

        def _backward():
            if out.grad is None:
                return
            N = x.data.shape[0]
            dout = out.grad

            if self.affine:
                if self.gamma.requires_grad:
                    self.gamma._ensure_grad()
                    self.gamma.grad += (dout * x_norm).sum(axis=0)
                if self.beta.requires_grad:
                    self.beta._ensure_grad()
                    self.beta.grad += dout.sum(axis=0)
                dxn = dout * self.gamma.data
            else:
                dxn = dout

            if x.requires_grad:
                x._ensure_grad()
                std_inv = 1.0 / np.sqrt(var + self.eps)
                dx = (dxn - dxn.mean(axis=0) - x_norm * (dxn * x_norm).mean(axis=0)) * std_inv
                x.grad += dx

        out._backward = _backward
        return out


class BatchNorm2d(Module):
    """
    Batch Normalisation for 4-D inputs (N, C, H, W).

    Args:
        num_features: Number of channels C.
        eps:          Numerical stability (default 1e-5).
        momentum:     Running stat update factor (default 0.1).
        affine:       Learnable gamma / beta (default True).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)

        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        x_flat = x.data.reshape(N, C, -1)  # (N, C, H*W)

        if self.training:
            mean = x_flat.mean(axis=(0, 2))
            var = x_flat.var(axis=(0, 2))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var

        mean_b = mean.reshape(1, C, 1, 1)
        var_b = var.reshape(1, C, 1, 1)
        x_norm = (x.data - mean_b) / np.sqrt(var_b + self.eps)

        out_data = x_norm.copy()
        if self.affine:
            out_data = out_data * self.gamma.data.reshape(1, C, 1, 1) + self.beta.data.reshape(1, C, 1, 1)

        requires_grad = x.requires_grad or (self.affine and self.gamma.requires_grad)
        out = Tensor(out_data, requires_grad=requires_grad, _children=(x,), _op='bn2d')

        def _backward():
            if out.grad is None:
                return
            dout = out.grad
            if self.affine:
                if self.gamma.requires_grad:
                    self.gamma._ensure_grad()
                    self.gamma.grad += (dout * x_norm).sum(axis=(0, 2, 3))
                if self.beta.requires_grad:
                    self.beta._ensure_grad()
                    self.beta.grad += dout.sum(axis=(0, 2, 3))
                dxn = dout * self.gamma.data.reshape(1, C, 1, 1)
            else:
                dxn = dout

            if x.requires_grad:
                x._ensure_grad()
                M = N * H * W
                std_inv = 1.0 / np.sqrt(var_b + self.eps)
                dx = (dxn - dxn.mean(axis=(0, 2, 3), keepdims=True)
                      - x_norm * (dxn * x_norm).mean(axis=(0, 2, 3), keepdims=True)) * std_inv
                x.grad += dx

        out._backward = _backward
        return out


class MaxPool2d(Module):
    """
    2-D Max Pooling.

    Args:
        kernel_size: int or (kH, kW).
        stride:      int or (sH, sW). Defaults to kernel_size.
        padding:     int or (pH, pW).
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        x_pad = np.pad(x.data, ((0, 0), (0, 0), (pH, pH), (pW, pW)), constant_values=-np.inf) if (pH or pW) else x.data
        out_H = (x_pad.shape[2] - kH) // sH + 1
        out_W = (x_pad.shape[3] - kW) // sW + 1

        out_data = np.zeros((N, C, out_H, out_W), dtype=np.float32)
        mask = np.zeros_like(x_pad, dtype=bool)

        for i in range(out_H):
            for j in range(out_W):
                h0, w0 = i * sH, j * sW
                patch = x_pad[:, :, h0:h0 + kH, w0:w0 + kW]
                out_data[:, :, i, j] = patch.max(axis=(2, 3))
                local_mask = (patch == out_data[:, :, i:i + 1, j:j + 1])
                mask[:, :, h0:h0 + kH, w0:w0 + kW] |= local_mask

        out = Tensor(out_data, requires_grad=x.requires_grad,
                     _children=(x,), _op='maxpool2d')

        def _backward():
            if x.requires_grad and out.grad is not None:
                x._ensure_grad()
                dout_pad = np.zeros_like(x_pad)
                for i in range(out_H):
                    for j in range(out_W):
                        h0, w0 = i * sH, j * sW
                        local_mask = mask[:, :, h0:h0 + kH, w0:w0 + kW]
                        dout_pad[:, :, h0:h0 + kH, w0:w0 + kW] += (
                            local_mask * out.grad[:, :, i:i + 1, j:j + 1]
                        )
                if pH or pW:
                    x.grad += dout_pad[:, :, pH:pH + H, pW:pW + W]
                else:
                    x.grad += dout_pad

        out._backward = _backward
        return out


class AvgPool2d(Module):
    """2-D Average Pooling."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = self.kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        x_pad = np.pad(x.data, ((0, 0), (0, 0), (pH, pH), (pW, pW))) if (pH or pW) else x.data
        out_H = (x_pad.shape[2] - kH) // sH + 1
        out_W = (x_pad.shape[3] - kW) // sW + 1
        out_data = np.zeros((N, C, out_H, out_W), dtype=np.float32)
        for i in range(out_H):
            for j in range(out_W):
                out_data[:, :, i, j] = x_pad[:, :, i * sH:i * sH + kH, j * sW:j * sW + kW].mean(axis=(2, 3))
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op='avgpool2d')

        def _backward():
            if x.requires_grad and out.grad is not None:
                x._ensure_grad()
                dout_pad = np.zeros_like(x_pad)
                scale = 1.0 / (kH * kW)
                for i in range(out_H):
                    for j in range(out_W):
                        dout_pad[:, :, i * sH:i * sH + kH, j * sW:j * sW + kW] += (
                            out.grad[:, :, i:i + 1, j:j + 1] * scale
                        )
                if pH or pW:
                    x.grad += dout_pad[:, :, pH:pH + H, pW:pW + W]
                else:
                    x.grad += dout_pad

        out._backward = _backward
        return out


class Flatten(Module):
    """
    Flatten a tensor from `start_dim` onwards.

    Args:
        start_dim: First dim to flatten (default 1, i.e. keep batch dim).
    """

    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


class Dropout(Module):
    """
    Randomly zero elements during training (inverted dropout).

    Args:
        p: Probability of dropping a unit (default 0.5).
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask_data = (np.random.rand(*x.data.shape) >= self.p).astype(np.float32) / (1 - self.p)
        mask = Tensor(mask_data)
        return x * mask

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Embedding(Module):
    """
    Simple lookup table for embedding integer indices.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim:  Embedding vector size.
        padding_idx:    If given, zeros out that index gradient.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        w = (np.random.randn(num_embeddings, embedding_dim) * 0.01).astype(np.float32)
        self.weight = Tensor(w, requires_grad=True)

    def forward(self, indices) -> Tensor:
        idx = indices.data.astype(int) if isinstance(indices, Tensor) else np.array(indices, dtype=int)
        out_data = self.weight.data[idx]
        out = Tensor(out_data, requires_grad=self.weight.requires_grad,
                     _children=(self.weight,), _op='embedding')

        def _backward():
            if self.weight.requires_grad and out.grad is not None:
                self.weight._ensure_grad()
                np.add.at(self.weight.grad, idx.flatten(), out.grad.reshape(-1, self.embedding_dim))
                if self.padding_idx is not None:
                    self.weight.grad[self.padding_idx] = 0

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
