"""
LowMind Tensor — core autograd engine
Bug-fixed and extended version with full broadcasting, tuple-axis support,
and proper gradient tracking for all operations.
"""
import numpy as np
from .memory import memory_manager


def _grad_enabled():
    """Check if gradient tracking is enabled (respects no_grad context)."""
    try:
        from .no_grad import _grad_enabled as _check
        return _check()
    except ImportError:
        return True


class Tensor:
    """
    N-dimensional array with automatic differentiation.

    Supports:
    - Arithmetic: +, -, *, /, **, @
    - Reductions: sum, mean, max, min
    - Shape ops: reshape, transpose, squeeze, unsqueeze, flatten
    - Activations: relu, sigmoid, tanh, exp, log, leaky_relu, elu, gelu, softmax
    - Indexing with basic gradient support
    """

    def __init__(self, data, requires_grad=False, _children=(), _op='',
                 device='cpu', name=None, dtype=np.float32):
        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype, copy=False)
        elif isinstance(data, Tensor):
            self.data = data.data.astype(dtype, copy=False)
        else:
            self.data = np.array(data, dtype=dtype)

        # Honour no_grad context: suppress gradient tracking for computed tensors
        # (leaf tensors created by the user with explicit requires_grad=True keep their flag)
        if requires_grad and not _grad_enabled() and _op != '':
            requires_grad = False

        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.device = device
        self.name = name
        self._version = 0

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def _ensure_grad(self):
        if self.requires_grad and self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    @staticmethod
    def _sum_to(grad, target_shape):
        """Reduce *grad* to match *target_shape* (reverse broadcasting)."""
        if grad.shape == target_shape:
            return grad
        # Sum over leading extra dims
        ndim_diff = grad.ndim - len(target_shape)
        if ndim_diff > 0:
            grad = grad.sum(axis=tuple(range(ndim_diff)))
        # Sum over broadcast (size-1) dims
        for i, (tgt, src) in enumerate(zip(target_shape, grad.shape)):
            if tgt == 1 and src > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(target_shape)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Called .backward() on a Tensor that does not require gradients.")

        if grad is not None:
            self.grad = np.array(grad, dtype=np.float32) if not isinstance(grad, np.ndarray) else grad
        elif self.grad is None:
            self.grad = np.ones_like(self.data, dtype=np.float32)

        # Topological sort
        topo, visited = [], set()

        def _build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    _build(child)
                topo.append(v)

        _build(self)

        for node in reversed(topo):
            if node.requires_grad and node.grad is None:
                node.grad = np.zeros_like(node.data, dtype=np.float32)
            node._backward()

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.0)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += Tensor._sum_to(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += Tensor._sum_to(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += Tensor._sum_to(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += Tensor._sum_to(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                # Handle 1-D "vector" case: W @ x where x is 1-D
                if other.data.ndim == 1:
                    # outer product: dL/dW = out.grad (col) ⊗ x (row)
                    self.grad += np.outer(out.grad, other.data)
                else:
                    self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad()
                # self.T @ out.grad works correctly for both matrix and vector
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        if isinstance(power, Tensor):
            power = float(power.data)
        out = Tensor(self.data ** power,
                     requires_grad=self.requires_grad,
                     _children=(self,), _op=f'**{power}')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(-other))

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    # Reflected operators
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return Tensor(other) - self

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __rmatmul__(self, other):
        return Tensor(other) @ self

    # ------------------------------------------------------------------
    # Reductions  (FIX: proper tuple-axis support)
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        # FIX: tuple axis support (was broken in original)
        if axis is not None:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.data.shape[a] for a in axes]))
        else:
            count = self.data.size

        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad / count
                if axis is not None and not keepdims:
                    axes_norm = (axis,) if isinstance(axis, int) else tuple(axis)
                    for ax in sorted(axes_norm):
                        grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op='max')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                max_expanded = np.expand_dims(out_data, axis=axis) if (axis is not None and not keepdims) else out_data
                mask = (self.data == np.broadcast_to(max_expanded if axis is not None else out_data, self.data.shape))
                self.grad += np.broadcast_to(grad, self.data.shape) * mask

        out._backward = _backward
        return out

    def min(self, axis=None, keepdims=False):
        out_data = np.min(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op='min')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                min_expanded = np.expand_dims(out_data, axis=axis) if (axis is not None and not keepdims) else out_data
                mask = (self.data == np.broadcast_to(min_expanded if axis is not None else out_data, self.data.shape))
                self.grad += np.broadcast_to(grad, self.data.shape) * mask

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes=axes),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='T')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                inv_axes = None if axes is None else tuple(np.argsort(axes))
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def flatten(self, start_dim=1):
        shape = self.data.shape[:start_dim] + (-1,)
        return self.reshape(shape)

    def squeeze(self, axis=None):
        out = Tensor(np.squeeze(self.data, axis=axis),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='squeeze')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def unsqueeze(self, axis):
        out = Tensor(np.expand_dims(self.data, axis=axis),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='unsqueeze')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activation functions (on Tensor, not as standalone layers)
    # ------------------------------------------------------------------

    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def leaky_relu(self, negative_slope=0.01):
        out_data = np.where(self.data > 0, self.data, negative_slope * self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op='leaky_relu')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.where(self.data > 0, 1.0, negative_slope) * out.grad

        out._backward = _backward
        return out

    def elu(self, alpha=1.0):
        out_data = np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1))
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op='elu')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.where(self.data > 0, 1.0, out_data + alpha) * out.grad

        out._backward = _backward
        return out

    def gelu(self):
        """Gaussian Error Linear Unit."""
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (self.data + 0.044715 * self.data ** 3)))
        out_data = self.data * cdf
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op='gelu')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                pdf = np.exp(-0.5 * self.data ** 2) / np.sqrt(2 * np.pi)
                d_gelu = cdf + self.data * pdf
                self.grad += d_gelu * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -88, 88)))
        out = Tensor(s, requires_grad=self.requires_grad,
                     _children=(self,), _op='sigmoid')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad,
                     _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad,
                     _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                dot = (out.grad * s).sum(axis=axis, keepdims=True)
                self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad,
                     _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        eps = 1e-8
        out = Tensor(np.log(self.data + eps),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad / (self.data + eps)

        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad,
                     _children=(self,), _op='abs')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.sign(self.data) * out.grad

        out._backward = _backward
        return out

    def clip(self, min_val=None, max_val=None):
        out = Tensor(np.clip(self.data, min_val, max_val),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='clip')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                mask = np.ones_like(self.data)
                if min_val is not None:
                    mask[self.data < min_val] = 0
                if max_val is not None:
                    mask[self.data > max_val] = 0
                self.grad += mask * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad,
                     _children=(self,), _op='index')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    # ------------------------------------------------------------------
    # Properties and utilities
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def item(self):
        return self.data.item() if self.data.size == 1 else self.data

    def numpy(self):
        return self.data

    def detach(self):
        """Return a new Tensor with the same data but no gradient tracking."""
        return Tensor(self.data.copy(), requires_grad=False)

    def copy(self):
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        if self.grad is not None:
            t.grad = self.grad.copy()
        return t

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, requires_grad={self.requires_grad})"

    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    # Chunked matmul for very large matrices
    # ------------------------------------------------------------------

    def matmul_memory_efficient(self, other, chunk_size=512):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.size * other.data.shape[-1] > 2_000_000:
            return self._chunked_matmul(other, chunk_size)
        return self @ other

    def _chunked_matmul(self, other, chunk_size=512):
        A, B = self.data, other.data
        result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
        for i in range(0, A.shape[0], chunk_size):
            ie = min(i + chunk_size, A.shape[0])
            for j in range(0, B.shape[1], chunk_size):
                je = min(j + chunk_size, B.shape[1])
                result[i:ie, j:je] = A[i:ie] @ B[:, j:je]
        out = Tensor(result,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='chunked_mm')

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad @ B.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += A.T @ out.grad

        out._backward = _backward
        return out

    # Comparison (no grad)
    # Make Tensor hashable by identity so it can be put in sets
    __hash__ = object.__hash__

    def __eq__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data == other_data)

    def __lt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data < other_data)

    def __gt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data > other_data)


# ------------------------------------------------------------------
# Tensor factory functions
# ------------------------------------------------------------------

def zeros(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)


def rand(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)


def arange(start, stop=None, step=1, requires_grad=False):
    if stop is None:
        start, stop = 0, start
    return Tensor(np.arange(start, stop, step, dtype=np.float32), requires_grad=requires_grad)


def from_numpy(arr, requires_grad=False):
    return Tensor(arr.astype(np.float32, copy=False), requires_grad=requires_grad)


def cat(tensors, axis=0):
    """Concatenate a list of Tensors along an axis."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    rg = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=rg, _children=tuple(tensors), _op='cat')

    sizes = [t.data.shape[axis] for t in tensors]

    def _backward():
        offset = 0
        for t, s in zip(tensors, sizes):
            if t.requires_grad:
                t._ensure_grad()
                slices = [slice(None)] * out.data.ndim
                slices[axis] = slice(offset, offset + s)
                t.grad += out.grad[tuple(slices)]
            offset += s

    out._backward = _backward
    return out


def stack(tensors, axis=0):
    """Stack tensors along a new axis."""
    expanded = [t.unsqueeze(axis) for t in tensors]
    return cat(expanded, axis=axis)


def clip_grad_norm(parameters, max_norm):
    """Clip gradient norm of parameters in-place. Returns the total norm."""
    total_norm = 0.0
    params = [p for p in parameters if p.grad is not None]
    for p in params:
        total_norm += np.sum(p.grad ** 2)
    total_norm = float(np.sqrt(total_norm))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= scale
    return total_norm
