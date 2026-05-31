
<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
</div>

# LowMind — Ultra-Lightweight Deep Learning Framework

<div align="center">

**Deep Learning on Raspberry Pi and Low-End Devices Made Easy**

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-orange)](https://github.com/dhaval-vedra/lowmind)
[![Platform](https://img.shields.io/badge/platform-raspberry%20pi%20%7C%20linux%20%7C%20macos%20%7C%20windows-lightgrey)](https://www.raspberrypi.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy%20%7C%20psutil-yellow)](requirements.txt)

*"Democratizing Deep Learning for Resource-Constrained Environments"*

</div>

---

## What is LowMind?

**LowMind** is a pure-NumPy deep learning framework built from scratch for Raspberry Pi, embedded systems, and any resource-constrained environment. It gives you a PyTorch-like API without the multi-GB installation — just NumPy and psutil.

```bash
pip install lowmind
```

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full API Reference](#full-api-reference)
  - [Tensors](#tensors)
  - [Layers](#layers)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
  - [LR Schedulers](#lr-schedulers)
  - [Data Utilities](#data-utilities)
  - [Metrics](#metrics)
  - [Trainer](#trainer)
  - [Callbacks](#callbacks)
  - [Pre-built Models](#pre-built-models)
  - [System Monitor](#system-monitor)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Raspberry Pi Tips](#raspberry-pi-tips)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Category | What's included |
|---|---|
| **Autograd** | Reverse-mode automatic differentiation, full broadcasting, tuple-axis support |
| **Layers** | Linear, Conv2d, BatchNorm1d/2d, MaxPool2d, AvgPool2d, Flatten, Dropout, Embedding |
| **Activations** | ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh, Softmax, LogSoftmax |
| **Loss Functions** | CrossEntropy, BCE, MSE, MAE, Huber, NLL |
| **Optimizers** | SGD (+ Nesterov), Adam, AdamW, RMSprop, AdaGrad |
| **LR Schedulers** | StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, LinearWarmup |
| **Data** | Dataset, TensorDataset, DataLoader, train_test_split |
| **Metrics** | accuracy, top-k accuracy, precision, recall, F1, confusion matrix, R², MSE, MAE |
| **Trainer** | High-level training loop with callbacks, gradient clipping, validation |
| **Callbacks** | EarlyStopping, ModelCheckpoint, LRSchedulerCallback, History |
| **Models** | MicroMLP, MicroCNN, TinyResNet |
| **Monitoring** | SystemMonitor, memory_trace, health_score |
| **Model I/O** | save/load (compressed gzip or plain pickle), state_dict, load_state_dict |

---

## Installation

### From PyPI (recommended)
```bash
pip install lowmind
```

### From Source
```bash
git clone https://github.com/dhaval-vedra/lowmind.git
cd lowmind
pip install -e .
```

### Raspberry Pi (system packages)
```bash
sudo apt update
sudo apt install python3-pip python3-numpy python3-psutil
pip3 install lowmind
```

### Requirements
```
numpy>=1.19.0
psutil>=5.8.0
```

---

## Quick Start

```python
import lowmind as lm
import numpy as np

# Build a model
model = lm.Sequential(
    lm.Linear(784, 128),
    lm.ReLU(),
    lm.Dropout(0.3),
    lm.Linear(128, 10),
)

# Create optimizer
optimizer = lm.Adam(model.parameters(), lr=1e-3)

# Prepare data
X = np.random.randn(1000, 784).astype(np.float32)
y = np.random.randint(0, 10, 1000)
loader = lm.DataLoader(lm.TensorDataset(X, y), batch_size=64, shuffle=True)

# Training loop
for epoch in range(20):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = lm.cross_entropy_loss(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")
```

---

## Full API Reference

### Tensors

`lm.Tensor` is the core data structure — an N-dimensional array with automatic gradient tracking.

#### Creating Tensors

```python
import lowmind as lm
import numpy as np

# From data
t = lm.Tensor([1.0, 2.0, 3.0])                      # from list
t = lm.Tensor(np.array([[1, 2], [3, 4]]))            # from numpy
t = lm.Tensor(5.0, requires_grad=True)               # scalar with grad

# Factory functions
lm.zeros(3, 4)          # shape (3,4) filled with 0
lm.ones(2, 2)           # shape (2,2) filled with 1
lm.randn(10, 10)        # shape (10,10) random normal
lm.rand(5, 5)           # shape (5,5) random uniform [0,1]
lm.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
lm.from_numpy(arr)      # wrap a numpy array
```

#### Arithmetic

```python
a = lm.Tensor([1., 2., 3.], requires_grad=True)
b = lm.Tensor([4., 5., 6.], requires_grad=True)

c = a + b           # addition
c = a - b           # subtraction
c = a * b           # element-wise multiply
c = a / b           # element-wise divide
c = a ** 2          # power
c = a @ b           # matrix multiply (for 2-D)
c = -a              # negation
```

#### Reductions

```python
x = lm.Tensor([[1., 2.], [3., 4.]])

x.sum()                         # scalar: 10.0
x.sum(axis=0)                   # [4., 6.]
x.sum(axis=1, keepdims=True)    # [[3.], [7.]]
x.mean()                        # 2.5
x.mean(axis=(2, 3))             # works with tuple axis (CNN global pooling)
x.max(axis=1)                   # row-wise max
x.min()                         # global min
```

#### Activations (on Tensor)

```python
x = lm.Tensor([-2., -1., 0., 1., 2.])

x.relu()                    # [0, 0, 0, 1, 2]
x.sigmoid()                 # [0.12, 0.27, 0.5, 0.73, 0.88]
x.tanh()                    # [-0.96, -0.76, 0, 0.76, 0.96]
x.leaky_relu(0.01)          # [-0.02, -0.01, 0, 1, 2]
x.elu(1.0)                  # smooth version of relu
x.gelu()                    # gaussian error linear
x.softmax(axis=-1)          # probability distribution
x.exp()                     # element-wise e^x
x.log()                     # element-wise ln(x)
x.abs()                     # absolute value
x.clip(-1, 1)               # clamp values
```

#### Shape Operations

```python
x = lm.Tensor(np.arange(24).reshape(2, 3, 4))

x.reshape(6, 4)             # (6, 4)
x.flatten(start_dim=1)      # (2, 12)
x.transpose((0, 2, 1))     # (2, 4, 3)
x.T                         # transpose (last two dims)
x.squeeze(axis=1)           # remove size-1 dims
x.unsqueeze(axis=0)         # add dim
x[0]                        # index — gradient flows through
```

#### Autograd

```python
# Compute gradient of y = x^2 + 2x + 1 at x=3
x = lm.Tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad)   # 8.0  (dy/dx = 2x + 2 = 8)

# Multi-variable
a = lm.Tensor([1., 2.], requires_grad=True)
b = lm.Tensor([3., 4.], requires_grad=True)
loss = (a * b).sum()
loss.backward()
print(a.grad)   # [3., 4.]
print(b.grad)   # [1., 2.]

# Gradient clipping
lm.clip_grad_norm(model.parameters(), max_norm=1.0)
```

#### Utility Methods

```python
t.item()        # extract Python float (for 0-d or 1-element tensors)
t.numpy()       # get the underlying numpy array
t.detach()      # new tensor without grad tracking
t.copy()        # full copy including grad
t.shape         # shape tuple
t.ndim          # number of dimensions
t.size          # total number of elements
t.dtype         # numpy dtype (always float32)
t.zero_grad()   # fill grad with zeros
repr(t)         # Tensor(shape=(3,), dtype=float32, requires_grad=True)
```

---

### Layers

All layers are subclasses of `lm.Module`. They can be used standalone or combined in `lm.Sequential`.

#### Linear

```python
layer = lm.Linear(in_features=784, out_features=256, bias=True)
# Input:  (N, 784)
# Output: (N, 256)
```

#### Conv2d

```python
layer = lm.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,       # or (3, 3)
    stride=1,            # or (1, 1)
    padding=1,           # or (1, 1)
    bias=True,
)
# Input:  (N, 3, H, W)
# Output: (N, 32, H, W)  when padding=1, stride=1
```

#### BatchNorm1d / BatchNorm2d

```python
bn1 = lm.BatchNorm1d(256)        # for (N, features) inputs
bn2 = lm.BatchNorm2d(32)         # for (N, C, H, W) inputs
# Normalizes to mean=0, std=1 per batch
# Has learnable gamma (scale) and beta (shift)
```

#### MaxPool2d / AvgPool2d

```python
pool = lm.MaxPool2d(kernel_size=2, stride=2)   # halves spatial dims
pool = lm.AvgPool2d(kernel_size=2)
# Input:  (N, C, H, W)
# Output: (N, C, H//2, W//2)
```

#### Flatten

```python
flatten = lm.Flatten(start_dim=1)
# (N, C, H, W) → (N, C*H*W)
```

#### Dropout

```python
drop = lm.Dropout(p=0.5)   # 50% dropout during training
# Automatically disabled during model.eval()
```

#### Embedding

```python
embed = lm.Embedding(num_embeddings=10000, embedding_dim=128)
indices = lm.Tensor([0, 3, 7])
out = embed(indices)   # (3, 128)
```

#### Building Custom Modules

```python
class MyBlock(lm.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = lm.Linear(in_features, out_features)
        self.bn = lm.BatchNorm1d(out_features)

    def forward(self, x: lm.Tensor) -> lm.Tensor:
        return self.bn(self.fc(x)).relu()

block = MyBlock(64, 32)
out = block(lm.Tensor(np.random.randn(8, 64).astype(np.float32)))
# out.shape → (8, 32)
```

---

### Sequential

Stack layers in order:

```python
from collections import OrderedDict

# Positional
model = lm.Sequential(
    lm.Linear(784, 256),
    lm.ReLU(),
    lm.BatchNorm1d(256),
    lm.Dropout(0.3),
    lm.Linear(256, 10),
)

# Named (OrderedDict)
model = lm.Sequential(OrderedDict([
    ('fc1',  lm.Linear(784, 256)),
    ('relu', lm.ReLU()),
    ('fc2',  lm.Linear(256, 10)),
]))

print(model)           # shows architecture
model.num_parameters() # total trainable parameter count
```

---

### Loss Functions

All loss functions return a scalar `Tensor` with `requires_grad=True`.

#### Cross-Entropy (classification)

```python
# logits: (N, C)  targets: (N,) integer class indices
loss = lm.cross_entropy_loss(logits, targets)
loss = lm.cross_entropy_loss(logits, targets, reduction='sum')
```

#### Binary Cross-Entropy (binary classification)

```python
# output: probabilities [0,1] or raw logits
loss = lm.binary_cross_entropy_loss(output, targets)
loss = lm.binary_cross_entropy_loss(logits, targets, from_logits=True)
```

#### MSE (regression)

```python
loss = lm.mse_loss(predictions, targets)
loss = lm.mse_loss(predictions, targets, reduction='sum')
```

#### MAE (regression, outlier-robust)

```python
loss = lm.mae_loss(predictions, targets)
```

#### Huber Loss (smooth L1)

```python
# Quadratic for |error| < delta, linear otherwise
loss = lm.huber_loss(predictions, targets, delta=1.0)
```

#### NLL Loss (after log-softmax)

```python
log_probs = lm.LogSoftmax()(logits)     # (N, C)
loss = lm.nll_loss(log_probs, targets)  # (N,)
```

---

### Optimizers

All optimizers share the same interface:

```python
optimizer = lm.Adam(model.parameters(), lr=1e-3)

# Each training step:
optimizer.zero_grad()   # reset gradients
loss.backward()         # compute gradients
optimizer.step()        # update weights
```

#### SGD

```python
optimizer = lm.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,          # Nesterov-style momentum
    weight_decay=1e-4,     # L2 regularization
    nesterov=True,         # Nesterov momentum
)
```

#### Adam

```python
optimizer = lm.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),    # (beta1, beta2)
    eps=1e-8,
    weight_decay=0.0,
    amsgrad=False,         # AMSGrad variant
)
```

#### AdamW

```python
# Adam with decoupled weight decay (preferred for regularization)
optimizer = lm.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

#### RMSprop

```python
optimizer = lm.RMSprop(
    model.parameters(),
    lr=1e-3,
    alpha=0.99,            # smoothing factor
    momentum=0.0,
    weight_decay=0.0,
)
```

#### AdaGrad

```python
optimizer = lm.AdaGrad(model.parameters(), lr=0.01)
```

---

### LR Schedulers

#### StepLR — decay every N epochs

```python
scheduler = lm.StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

#### CosineAnnealingLR — smooth cosine decay

```python
scheduler = lm.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
```

#### ReduceLROnPlateau — reduce when stuck

```python
scheduler = lm.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5, verbose=True)
for epoch in range(epochs):
    val_loss = validate(...)
    scheduler.step(val_loss)      # pass the metric
```

#### MultiStepLR

```python
scheduler = lm.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
```

#### ExponentialLR

```python
scheduler = lm.ExponentialLR(optimizer, gamma=0.95)
```

#### LinearWarmupLR

```python
scheduler = lm.LinearWarmupLR(optimizer, warmup_steps=1000, target_lr=1e-3)
```

#### CyclicLR

```python
scheduler = lm.CyclicLR(
    optimizer, base_lr=1e-4, max_lr=1e-1,
    step_size=2000, mode='triangular')
for batch in loader:
    train(...)
    scheduler.step()    # step per batch, not per epoch
```

---

### Data Utilities

#### Dataset + TensorDataset

```python
# Wrap numpy arrays or Tensors
ds = lm.TensorDataset(X_train, y_train)
print(len(ds))           # number of samples
X, y = ds[0]             # get first sample

# Custom Dataset
class MyDataset(lm.Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

#### DataLoader

```python
loader = lm.DataLoader(
    dataset=ds,
    batch_size=64,
    shuffle=True,       # shuffle before each epoch
    drop_last=False,    # drop incomplete last batch
)

for X_batch, y_batch in loader:
    # X_batch and y_batch are Tensors
    pass

print(len(loader))   # number of batches
```

#### train_test_split

```python
X_train, X_val, y_train, y_val = lm.train_test_split(
    X, y,
    test_size=0.2,   # 20% validation
    shuffle=True,
    seed=42,
)
```

---

### Metrics

All metrics accept Tensors or numpy arrays.

```python
# Classification
lm.accuracy(predictions, targets)               # 0-1 float
lm.top_k_accuracy(logits, targets, k=5)         # 0-1 float
lm.precision(logits, targets, num_classes=10)   # macro by default
lm.recall(logits, targets, num_classes=10)
lm.f1_score(logits, targets, num_classes=10)
lm.confusion_matrix(logits, targets)            # (C, C) numpy array

# Regression
lm.r2_score(predictions, targets)               # R² coefficient
lm.mean_squared_error(predictions, targets)     # MSE
lm.mean_absolute_error(predictions, targets)    # MAE

# All precision/recall/f1 support average='macro', 'micro', or 'none'
per_class_f1 = lm.f1_score(logits, targets, num_classes=10, average='none')
```

---

### Trainer

High-level training loop — handles training, validation, logging, and callbacks automatically.

```python
trainer = lm.Trainer(
    model=model,
    optimizer=lm.Adam(model.parameters(), lr=1e-3),
    loss_fn=lm.cross_entropy_loss,
    callbacks=[
        lm.EarlyStopping(patience=10),
        lm.ModelCheckpoint('/tmp/best.lmz'),
    ],
    clip_grad=1.0,       # gradient norm clipping (0 = off)
    verbose=1,           # print every N epochs
)

history = trainer.fit(train_loader, val_loader, epochs=100)
# history = {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}

# Evaluate
val_loss, val_acc = trainer.evaluate(val_loader)

# Inference
predictions = trainer.predict(X_test)   # numpy array of class indices
```

---

### Callbacks

#### EarlyStopping

```python
cb = lm.EarlyStopping(
    patience=10,       # epochs to wait
    min_delta=1e-4,    # minimum improvement
    mode='min',        # 'min' for loss, 'max' for accuracy
    verbose=True,
)
```

#### ModelCheckpoint

```python
cb = lm.ModelCheckpoint(
    filepath='/tmp/best_model.lmz',
    monitor='val_loss',
    mode='min',
    verbose=True,
    save_best_only=True,
)
```

#### LRSchedulerCallback

```python
scheduler = lm.ReduceLROnPlateau(optimizer, patience=5)
cb = lm.LRSchedulerCallback(scheduler, monitor='val_loss')
```

#### History

```python
history_cb = lm.History()
trainer.fit(train_loader, val_loader, epochs=50)
print(history_cb.history['train_loss'])
```

---

### Pre-built Models

#### MicroMLP — for tabular / flat data

```python
model = lm.MicroMLP(
    input_size=784,
    hidden_sizes=[256, 128],   # list of hidden layer sizes
    output_size=10,
    dropout=0.3,
)
```

#### MicroCNN — for small images

```python
model = lm.MicroCNN(
    in_channels=3,      # 3 = RGB, 1 = grayscale
    num_classes=10,
    input_size=32,      # spatial size (HxW must be square)
    dropout=0.2,
)
# Input:  (N, 3, 32, 32)
# Output: (N, 10)
```

#### TinyResNet — with residual connections

```python
model = lm.TinyResNet(
    in_channels=3,
    num_classes=10,
    input_size=32,
    base_filters=16,    # reduce to 8 for very constrained devices
)
```

#### Model I/O

```python
# Save weights (compressed gzip — recommended)
model.save('/path/to/model.lmz')

# Save uncompressed
model.save('/path/to/model.lm', compress=False)

# Load into a same-architecture model
model.load('/path/to/model.lmz')

# Access raw state dict
sd = model.state_dict()          # {'0.weight': ndarray, '0.bias': ndarray, ...}
model.load_state_dict(sd)        # restore from dict
model.load_state_dict(sd, strict=False)  # ignore missing keys

# Count parameters
model.num_parameters()           # total trainable params
model.summary()                  # print architecture table
```

---

### System Monitor

```python
# Configure memory limit (especially important on Raspberry Pi)
lm.configure_memory(max_mb=128)   # default 256MB

# Monitor system health
monitor = lm.SystemMonitor()
monitor.print_status()            # print CPU, RAM, temp stats

stats = monitor.get_stats()       # dict of all stats
score = monitor.health_score()    # 0-100 score

# Trace memory usage of a block
with lm.memory_trace("Forward Pass"):
    out = model(X)

# Optimize for inference (drop gradient buffers)
lm.memory_manager.optimize_for_inference()

# Get current memory info
info = lm.memory_manager.get_memory_info()
# {'allocated_mb': 12.3, 'max_mb': 256.0, 'usage_percent': 4.8, ...}
```

---

## Examples

Ten complete examples are in the `examples/` folder:

| File | What it demonstrates |
|---|---|
| `01_basic_tensors.py` | Tensor creation, arithmetic, autograd from scratch |
| `02_linear_regression.py` | Linear regression with SGD, custom training loop |
| `03_mlp_classification.py` | XOR classification with Sequential, Adam, DataLoader |
| `04_mnist_like.py` | Full pipeline: MicroMLP + Trainer + EarlyStopping + ModelCheckpoint |
| `05_cnn_image.py` | MicroCNN for image classification, BatchNorm, MaxPool |
| `06_optimizers_comparison.py` | Benchmark SGD vs Adam vs RMSprop vs AdaGrad |
| `07_custom_layer.py` | Build custom attention layer, LayerNorm, transformer block |
| `08_save_load_model.py` | Save/load weights, state dict, transfer learning |
| `09_lr_schedulers.py` | Compare 6 LR scheduler strategies |
| `10_raspberry_pi_monitor.py` | System monitoring, memory tracing, health scoring |

Run any example:
```bash
cd lowmind_repo
python examples/01_basic_tensors.py
python examples/04_mnist_like.py
```

---

## Project Structure

```
lowmind/
├── lowmind/                 # Main package
│   ├── __init__.py          # Public API — all exports here
│   ├── core/
│   │   ├── tensor.py        # Tensor class + autograd engine
│   │   ├── memory.py        # MemoryManager (LRU, GC optimization)
│   │   └── module.py        # Module base class (save/load, parameter iteration)
│   ├── nn/
│   │   ├── layers.py        # Linear, Conv2d, BatchNorm, Pool, Flatten, Dropout, Embedding
│   │   ├── activation.py    # ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh, Softmax
│   │   ├── loss.py          # cross_entropy, bce, mse, mae, huber, nll
│   │   └── sequential.py    # Sequential container
│   ├── optim/
│   │   ├── sgd.py           # SGD + Nesterov momentum
│   │   ├── adam.py          # Adam, AdamW, RMSprop, AdaGrad
│   │   └── scheduler.py     # StepLR, CosineAnnealingLR, ReduceLROnPlateau, ...
│   ├── data/
│   │   └── dataloader.py    # Dataset, TensorDataset, DataLoader, train_test_split
│   ├── utils/
│   │   ├── metrics.py       # accuracy, precision, recall, f1, r2, ...
│   │   ├── trainer.py       # Trainer (high-level training loop)
│   │   ├── callbacks.py     # EarlyStopping, ModelCheckpoint, History, ...
│   │   └── monitor.py       # SystemMonitor, memory_trace
│   └── models/
│       └── micro_cnn.py     # MicroMLP, MicroCNN, TinyResNet
├── examples/                # 10 complete runnable examples
├── tests/                   # pytest test suite
├── docs/                    # Extended documentation
├── setup.py
├── requirements.txt
└── README.md
```

---

## Raspberry Pi Tips

```python
import lowmind as lm

# 1. Set memory limit appropriate for your Pi model
lm.configure_memory(max_mb=64)   # Pi Zero / 512MB Pi
lm.configure_memory(max_mb=128)  # Pi 3 (1GB)
lm.configure_memory(max_mb=256)  # Pi 4 (2GB+)

# 2. Use small batch sizes
loader = lm.DataLoader(ds, batch_size=8)   # Pi Zero
loader = lm.DataLoader(ds, batch_size=16)  # Pi 3/4

# 3. Use Pi-optimized architectures
model = lm.MicroMLP(784, [64], 10)             # smallest
model = lm.MicroCNN(in_channels=1, num_classes=10, input_size=28)

# 4. Monitor health during training
monitor = lm.SystemMonitor()
if monitor.health_score() < 40:
    print("Warning: system under stress — reduce batch size")

# 5. Free memory after training
lm.memory_manager.optimize_for_inference()
import gc; gc.collect()

# 6. Reduce model size for inference
model.save('/tmp/model.lmz', compress=True)   # ~70% smaller than plain
```

---

## Contributing

Contributions are welcome! Areas where help is needed:
- Performance benchmarks on more Pi models
- LSTM / GRU layers
- Quantization (INT8 inference)
- Distributed training across multiple Pis

**Submitting a PR:**
1. Fork and create a feature branch
2. Run tests: `pytest tests/ -v`
3. Add tests for new features
4. Submit a PR with a clear description

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**Built with care in India by Dhaval Vedra**

*Empowering AI at the edge — from data centers down to $35 computers*

</div>
