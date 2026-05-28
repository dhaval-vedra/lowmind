"""
LowMind Built-in Model Architectures — optimized for low-end devices
"""
import numpy as np
from ..core.module import Module
from ..core.tensor import Tensor
from ..nn.layers import Linear, Conv2d, BatchNorm2d, MaxPool2d, Flatten, Dropout
from ..nn.sequential import Sequential
from ..nn.activation import ReLU


class MicroMLP(Module):
    """
    Ultra-lightweight Multi-Layer Perceptron.

    Good for tabular data or small image classification (after flatten).

    Args:
        input_size:   Number of input features.
        hidden_sizes: List of hidden layer sizes.
        output_size:  Number of output classes.
        dropout:      Dropout probability (default 0.0 = off).
        activation:   Activation class (default ReLU).

    Example::

        model = lm.MicroMLP(784, [128, 64], 10, dropout=0.3)
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.0, activation=None):
        super().__init__()
        if activation is None:
            activation = ReLU

        sizes = [input_size] + list(hidden_sizes)
        layers = []
        for i, (in_f, out_f) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(Linear(in_f, out_f))
            layers.append(activation())
            if dropout > 0:
                layers.append(Dropout(dropout))

        layers.append(Linear(sizes[-1], output_size))
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MicroCNN(Module):
    """
    Ultra-lightweight CNN for small image classification.
    Fits comfortably within Raspberry Pi memory limits.

    Architecture:
        Conv2d(3→16) → BN → ReLU → MaxPool
        Conv2d(16→32) → BN → ReLU → MaxPool
        Flatten → Linear → ReLU → Linear

    Args:
        in_channels:  Input image channels (default 3 for RGB, 1 for grayscale).
        num_classes:  Number of output classes (default 10).
        input_size:   Spatial input size HxW (default 32, for 32x32 images).
        dropout:      Dropout before the final linear (default 0.2).

    Example::

        model = lm.MicroCNN(in_channels=1, num_classes=10, input_size=28)
        x = lm.Tensor(np.random.randn(8, 1, 28, 28))
        out = model(x)   # shape: (8, 10)
    """

    def __init__(self, in_channels=3, num_classes=10, input_size=32, dropout=0.2):
        super().__init__()
        self.features = Sequential(
            Conv2d(in_channels, 16, 3, padding=1),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2),
            Conv2d(16, 32, 3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),
        )
        # After two 2x pooling layers spatial dims are input_size // 4
        reduced = input_size // 4
        flat_size = 32 * reduced * reduced
        self.classifier = Sequential(
            Flatten(),
            Linear(flat_size, 128),
            ReLU(),
            Dropout(dropout),
            Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


class TinyResBlock(Module):
    """
    Residual block for TinyResNet — 2 conv layers with a skip connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, 3, padding=1)
        self.bn1 = BatchNorm2d(channels)
        self.conv2 = Conv2d(channels, channels, 3, padding=1)
        self.bn2 = BatchNorm2d(channels)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Simple element-wise add with residual (same channels, same spatial)
        out_data = out.data + residual.data
        result = Tensor(out_data, requires_grad=out.requires_grad or residual.requires_grad,
                        _children=(out, residual), _op='resblock')

        def _backward():
            if out.requires_grad:
                out._ensure_grad()
                out.grad += result.grad
            if residual.requires_grad:
                residual._ensure_grad()
                residual.grad += result.grad

        result._backward = _backward
        return self.relu(result)


class TinyResNet(Module):
    """
    Tiny ResNet — adds residual connections for better gradient flow.
    More accurate than MicroCNN on harder datasets but slightly heavier.

    Args:
        in_channels:  Input channels (default 3).
        num_classes:  Output classes (default 10).
        input_size:   Spatial size (default 32).
        base_filters: Starting filter count (default 16).

    Example::

        model = lm.TinyResNet(in_channels=3, num_classes=10, input_size=32)
    """

    def __init__(self, in_channels=3, num_classes=10, input_size=32, base_filters=16):
        super().__init__()
        f = base_filters
        self.stem = Sequential(
            Conv2d(in_channels, f, 3, padding=1),
            BatchNorm2d(f),
            ReLU(),
            MaxPool2d(2),
        )
        self.block1 = TinyResBlock(f)
        self.pool1 = MaxPool2d(2)
        self.classifier = Sequential(
            Flatten(),
            Linear(f * (input_size // 4) ** 2, 128),
            ReLU(),
            Dropout(0.25),
            Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        return self.classifier(x)
