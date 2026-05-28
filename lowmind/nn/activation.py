"""
LowMind Activation Functions as Module layers.
Use these inside Sequential or as named sub-modules.
"""
from ..core.module import Module
from ..core.tensor import Tensor


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.negative_slope)

    def __repr__(self):
        return f"LeakyReLU(slope={self.negative_slope})"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return x.elu(self.alpha)

    def __repr__(self):
        return f"ELU(alpha={self.alpha})"


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()

    def __repr__(self):
        return "GELU()"


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"


class LogSoftmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(self.axis).log()

    def __repr__(self):
        return f"LogSoftmax(axis={self.axis})"
