# LowMind - Raspberry Pi Optimized Deep Learning Framework

![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Optimized-red)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A lightweight, memory-efficient deep learning framework specifically optimized for Raspberry Pi and resource-constrained devices.

## Features

- 🚀 **Ultra Memory-Efficient** - Advanced memory management for limited RAM
- 📱 **Raspberry Pi Optimized** - Specialized for ARM architecture
- 🔥 **No Dependencies** - Pure NumPy implementation
- 📊 **Real-time Monitoring** - System health and memory tracking
- 🧠 **Auto-Gradient** - Automatic differentiation system
- ⚡ **Micro Models** - Pre-optimized tiny model architectures

## Installation

```bash
# Clone the repository
git clone https://github.com/dhaval-gamet/lowmind.git
cd lowmind

# Install dependencies
pip install numpy psutil
```

## Quick Start

### Basic Tensor Operations

```python
from lowmind import Tensor

# Create tensors
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
c = a + b
d = c * 2
e = d.relu()

print(f"Result: {e.data}")

# Backward pass
e.backward()
print(f"Gradient of a: {a.grad}")
```

### Simple Neural Network

```python
from lowmind import Tensor, Linear, cross_entropy_loss, SGD
import numpy as np

# Create a simple linear model
model = Linear(10, 2)  # 10 input features, 2 output classes
optimizer = SGD(model.parameters(), lr=0.01)

# Sample data
x = Tensor(np.random.randn(32, 10))  # 32 samples, 10 features
y = Tensor(np.random.randint(0, 2, (32,)))  # 32 labels

# Forward pass
output = model(x)
loss = cross_entropy_loss(output, y)

# Backward pass and optimization
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Loss: {loss.item()}")
```

### Raspberry Pi Monitoring

```python
from lowmind import RaspberryPiAdvancedMonitor

# Initialize monitor
monitor = RaspberryPiAdvancedMonitor()

# Get system status
monitor.print_detailed_status()

# Check health score
health = monitor.get_health_score()
print(f"System Health: {health:.1f}/100")
```

## Memory Management

```python
from lowmind import memory_manager, memory_trace

# Monitor memory usage
with memory_trace("My Operation"):
    # Your memory-intensive operations here
    large_tensor = Tensor(np.random.randn(1000, 1000))
    result = large_tensor.matmul_memory_efficient(large_tensor.T)

# Check memory info
mem_info = memory_manager.get_memory_info()
print(f"Memory used: {mem_info['allocated_mb']:.2f}MB")
```

## Advanced Examples

For more comprehensive examples, please check the `test/` folder:

- `test_basic_operations.py` - Basic tensor operations and autograd
- `test_neural_networks.py` - Neural network training examples
- `test_memory_management.py` - Memory optimization demonstrations
- `test_raspberry_pi.py` - Raspberry Pi specific optimizations
- `test_models.py` - Pre-built model architectures

## Model Architectures

### MicroCNN (Ultra-lightweight CNN)

```python
from lowmind import MicroCNN

# Create a tiny CNN for image classification
model = MicroCNN(num_classes=10)  # 10 classes

# Input: (batch_size, 3, 32, 32)
input_tensor = Tensor(np.random.randn(1, 3, 32, 32))
output = model(input_tensor)

print(f"Output shape: {output.shape}")
```

## Performance Tips

1. **Use `matmul_memory_efficient`** for large matrix multiplications
2. **Enable `low_memory_mode`** for memory-constrained devices
3. **Use `MicroCNN`** for computer vision tasks on Raspberry Pi
4. **Monitor system health** with `RaspberryPiAdvancedMonitor`
5. **Free unused tensors** with `memory_manager.free_unused()`

## API Reference

### Core Classes

- `Tensor` - Main tensor class with autograd
- `Module` - Base class for all neural network modules
- `Linear` - Fully connected layer
- `Conv2d` - 2D convolutional layer
- `Dropout` - Dropout layer for regularization

### Optimizers

- `SGD` - Stochastic Gradient Descent with momentum

### Loss Functions

- `cross_entropy_loss` - Cross entropy for classification
- `mse_loss` - Mean squared error for regression

### Utilities

- `MemoryManager` - Advanced memory management
- `RaspberryPiAdvancedMonitor` - System monitoring
- `memory_trace` - Context manager for memory profiling

## System Requirements

- **Python**: 3.6+
- **RAM**: 512MB+ (1GB recommended)
- **Storage**: 100MB free space
- **OS**: Raspberry Pi OS, Ubuntu, Debian

## Raspberry Pi Optimization

The framework includes several Raspberry Pi specific optimizations:

- **ARM-optimized NumPy operations**
- **Aggressive memory cleanup**
- **Temperature monitoring**
- **Dynamic batch size adjustment**
- **Chunked processing for large tensors**

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- 📧 Email: gametidhaval980@gmail.com
- 💬 Issues: [GitHub Issues](https://github.com/dhaval-gamet/lowmind/issues)


---

**Happy Coding!** 🚀 Let's make AI accessible on every device.