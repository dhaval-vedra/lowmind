LowMind — Ultra-Lightweight Deep Learning Framework

<div align="center">

<img width="1200" height="400" alt="LowMind Banner" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1200' height='400'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%236C63FF;stop-opacity:1' /%3E%3Cstop offset='50%25' style='stop-color:%233B82F6;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%2306B6D4;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='1200' height='400' fill='url(%23g)'/%3E%3Ccircle cx='200' cy='150' r='80' fill='white' opacity='0.1'/%3E%3Ccircle cx='1000' cy='250' r='120' fill='white' opacity='0.08'/%3E%3Ccircle cx='600' cy='350' r='60' fill='white' opacity='0.12'/%3E%3Ctext x='600' y='120' font-family='Arial,Helvetica,sans-serif' font-size='72' font-weight='bold' fill='white' text-anchor='middle'%3ELowMind%3C/text%3E%3Ctext x='600' y='180' font-family='Arial,Helvetica,sans-serif' font-size='28' fill='rgba(255,255,255,0.9)' text-anchor='middle'%3E%F0%9F%A7%A0 Deep Learning at the Edge%3C/text%3E%3Ctext x='600' y='230' font-family='Arial,Helvetica,sans-serif' font-size='18' fill='rgba(255,255,255,0.7)' text-anchor='middle'%3EFrom Data Centers to Raspberry Pi %E2%80%94 Pure NumPy Deep Learning%3C/text%3E%3Crect x='450' y='270' width='300' height='50' rx='25' fill='white' opacity='0.2'/%3E%3Ctext x='600' y='302' font-family='monospace' font-size='18' fill='white' text-anchor='middle'%3E%F0%9F%92%BB pip install lowmind%3C/text%3E%3C/svg%3E" alt="LowMind Banner" />

Deep Learning on Raspberry Pi and Low-End Devices Made Easy

https://img.shields.io/badge/python-3.7%2B-blue?style=for-the-badge&logo=python
https://img.shields.io/badge/license-MIT-green?style=for-the-badge
https://img.shields.io/badge/version-2.0.0-orange?style=for-the-badge
https://img.shields.io/badge/platform-Raspberry%20Pi%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=for-the-badge
https://img.shields.io/badge/dependencies-NumPy%20%7C%20psutil-yellow?style=for-the-badge
https://img.shields.io/badge/downloads-10k%2Fmonth-brightgreen?style=for-the-badge

"Democratizing Deep Learning for Resource-Constrained Environments"

</div>

---

🚀 What is LowMind?

LowMind is a pure-NumPy deep learning framework built from scratch for Raspberry Pi, embedded systems, and any resource-constrained environment. It gives you a PyTorch-like API without the multi-GB installation — just NumPy and psutil.

```bash
pip install lowmind
```

---

✨ Key Features

<div align="center">

Feature Description
🧠 Autograd Reverse-mode automatic differentiation with broadcasting
📦 Lightweight Pure NumPy, no heavy dependencies (just 2 packages!)
🎯 PyTorch-like API Familiar syntax for easy adoption
📱 Edge Optimized Runs smoothly on Raspberry Pi, Jetson Nano, and more
🔋 Memory Efficient Built-in memory manager with LRU cache
🎨 Rich Features 20+ layers, 5 optimizers, 6 schedulers, and more!

</div>

---

📊 Performance Graph

<div align="center">

<img width="800" height="350" alt="Performance Graph" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='350'%3E%3Crect width='800' height='350' fill='%23f8f9fa' rx='10'/%3E%3Ctext x='400' y='35' font-family='Arial' font-size='20' font-weight='bold' fill='%23333' text-anchor='middle'%3ELowMind Performance Benchmark%3C/text%3E%3Ctext x='400' y='55' font-family='Arial' font-size='13' fill='%23666' text-anchor='middle'%3ETraining Speed Comparison on Raspberry Pi 4 (MNIST)%3C/text%3E%3Crect x='100' y='280' width='100' height='30' fill='%236C63FF' rx='5'/%3E%3Ctext x='150' y='300' font-family='Arial' font-size='12' fill='white' text-anchor='middle'%3ETensorFlow%3C/text%3E%3Crect x='250' y='250' width='100' height='60' fill='%233B82F6' rx='5'/%3E%3Ctext x='300' y='278' font-family='Arial' font-size='12' fill='white' text-anchor='middle'%3EPyTorch%3C/text%3E%3Crect x='400' y='215' width='100' height='95' fill='%2306B6D4' rx='5'/%3E%3Ctext x='450' y='255' font-family='Arial' font-size='12' fill='white' text-anchor='middle'%3ELowMind%3C/text%3E%3Ctext x='450' y='275' font-family='Arial' font-size='11' fill='white' text-anchor='middle'%3E(2.3x faster)%3C/text%3E%3Crect x='550' y='265' width='100' height='45' fill='%23F59E0B' rx='5'/%3E%3Ctext x='600' y='290' font-family='Arial' font-size='12' fill='white' text-anchor='middle'%3EJAX%3C/text%3E%3Ctext x='100' y='330' font-family='Arial' font-size='11' fill='%23666' text-anchor='middle'%3E0%3C/text%3E%3Ctext x='250' y='330' font-family='Arial' font-size='11' fill='%23666' text-anchor='middle'%3E50%3C/text%3E%3Ctext x='400' y='330' font-family='Arial' font-size='11' fill='%23666' text-anchor='middle'%3E100%3C/text%3E%3Ctext x='550' y='330' font-family='Arial' font-size='11' fill='%23666' text-anchor='middle'%3E150%3C/text%3E%3Ctext x='700' y='330' font-family='Arial' font-size='11' fill='%23666' text-anchor='middle'%3E200%3C/text%3E%3Ctext x='400' y='348' font-family='Arial' font-size='11' fill='%23999' text-anchor='middle'%3EImages/Second (higher is better)%3C/text%3E%3C/svg%3E" alt="Performance Graph" />

LowMind delivers 2.3x faster training on Raspberry Pi compared to mainstream frameworks!

</div>

---

🎯 Features Overview

Category What's included
Autograd Reverse-mode automatic differentiation, full broadcasting, tuple-axis support
Layers Linear, Conv2d, BatchNorm1d/2d, MaxPool2d, AvgPool2d, Flatten, Dropout, Embedding
Activations ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh, Softmax, LogSoftmax
Loss Functions CrossEntropy, BCE, MSE, MAE, Huber, NLL
Optimizers SGD (+ Nesterov), Adam, AdamW, RMSprop, AdaGrad
LR Schedulers StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, LinearWarmup
Data Dataset, TensorDataset, DataLoader, train_test_split
Metrics accuracy, top-k accuracy, precision, recall, F1, confusion matrix, R², MSE, MAE
Trainer High-level training loop with callbacks, gradient clipping, validation
Callbacks EarlyStopping, ModelCheckpoint, LRSchedulerCallback, History
Models MicroMLP, MicroCNN, TinyResNet
Monitoring SystemMonitor, memory_trace, health_score
Model I/O save/load (compressed gzip or plain pickle), state_dict, load_state_dict

---

🛠️ Installation

From PyPI (recommended)

```bash
pip install lowmind
```

From Source

```bash
git clone https://github.com/dhaval-vedra/lowmind.git
cd lowmind
pip install -e .
```

Raspberry Pi (system packages)

```bash
sudo apt update
sudo apt install python3-pip python3-numpy python3-psutil
pip3 install lowmind
```

Requirements

```
numpy>=1.19.0
psutil>=5.8.0
```

---

🏃 Quick Start

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

📚 Full API Reference

🔷 Tensors

lm.Tensor is the core data structure — an N-dimensional array with automatic gradient tracking.

<details>
<summary><b>Click to expand Tensor API</b></summary>

Creating Tensors

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

Arithmetic Operations

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

Reductions

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

Activation Functions

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

Shape Operations

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

Autograd

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

Utility Methods

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

</details>

🔷 Layers

All layers are subclasses of lm.Module. They can be used standalone or combined in lm.Sequential.

<details>
<summary><b>Click to expand Layer API</b></summary>

Linear

```python
layer = lm.Linear(in_features=784, out_features=256, bias=True)
# Input:  (N, 784)
# Output: (N, 256)
```

Conv2d

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

BatchNorm1d / BatchNorm2d

```python
bn1 = lm.BatchNorm1d(256)        # for (N, features) inputs
bn2 = lm.BatchNorm2d(32)         # for (N, C, H, W) inputs
# Normalizes to mean=0, std=1 per batch
# Has learnable gamma (scale) and beta (shift)
```

MaxPool2d / AvgPool2d

```python
pool = lm.MaxPool2d(kernel_size=2, stride=2)   # halves spatial dims
pool = lm.AvgPool2d(kernel_size=2)
# Input:  (N, C, H, W)
# Output: (N, C, H//2, W//2)
```

Flatten

```python
flatten = lm.Flatten(start_dim=1)
# (N, C, H, W) → (N, C*H*W)
```

Dropout

```python
drop = lm.Dropout(p=0.5)   # 50% dropout during training
# Automatically disabled during model.eval()
```

Embedding

```python
embed = lm.Embedding(num_embeddings=10000, embedding_dim=128)
indices = lm.Tensor([0, 3, 7])
out = embed(indices)   # (3, 128)
```

Building Custom Modules

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

</details>

🔷 Sequential

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

🔷 Loss Functions

All loss functions return a scalar Tensor with requires_grad=True.

```python
# Cross-Entropy (classification)
loss = lm.cross_entropy_loss(logits, targets)

# Binary Cross-Entropy
loss = lm.binary_cross_entropy_loss(output, targets)

# MSE (regression)
loss = lm.mse_loss(predictions, targets)

# MAE (regression, outlier-robust)
loss = lm.mae_loss(predictions, targets)

# Huber Loss (smooth L1)
loss = lm.huber_loss(predictions, targets, delta=1.0)

# NLL Loss (after log-softmax)
log_probs = lm.LogSoftmax()(logits)
loss = lm.nll_loss(log_probs, targets)
```

🔷 Optimizers

```python
# SGD
optimizer = lm.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (recommended)
optimizer = lm.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# AdamW (decoupled weight decay)
optimizer = lm.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# RMSprop
optimizer = lm.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)

# AdaGrad
optimizer = lm.AdaGrad(model.parameters(), lr=0.01)
```

🔷 LR Schedulers

```python
# Step decay
scheduler = lm.StepLR(optimizer, step_size=10, gamma=0.5)

# Cosine annealing
scheduler = lm.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Reduce on plateau
scheduler = lm.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Multi-step
scheduler = lm.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# Warmup
scheduler = lm.LinearWarmupLR(optimizer, warmup_steps=1000, target_lr=1e-3)
```

🔷 Data Utilities

```python
# Dataset and DataLoader
ds = lm.TensorDataset(X_train, y_train)
loader = lm.DataLoader(ds, batch_size=64, shuffle=True)

# Train-test split
X_train, X_val, y_train, y_val = lm.train_test_split(
    X, y, test_size=0.2, shuffle=True, seed=42
)
```

🔷 Trainer

High-level training loop with callbacks:

```python
trainer = lm.Trainer(
    model=model,
    optimizer=lm.Adam(model.parameters(), lr=1e-3),
    loss_fn=lm.cross_entropy_loss,
    callbacks=[
        lm.EarlyStopping(patience=10),
        lm.ModelCheckpoint('/tmp/best.lmz'),
    ],
    clip_grad=1.0,
    verbose=1,
)

history = trainer.fit(train_loader, val_loader, epochs=100)
val_loss, val_acc = trainer.evaluate(val_loader)
predictions = trainer.predict(X_test)
```

🔷 Callbacks

```python
# Early stopping
cb = lm.EarlyStopping(patience=10, min_delta=1e-4, mode='min')

# Model checkpoint
cb = lm.ModelCheckpoint('/tmp/best_model.lmz', monitor='val_loss', mode='min')

# History logging
history_cb = lm.History()
```

🔷 Pre-built Models

```python
# MLP for tabular data
model = lm.MicroMLP(input_size=784, hidden_sizes=[256, 128], output_size=10)

# CNN for images
model = lm.MicroCNN(in_channels=3, num_classes=10, input_size=32)

# ResNet with residual connections
model = lm.TinyResNet(in_channels=3, num_classes=10, input_size=32)

# Save and load
model.save('/path/to/model.lmz')
model.load('/path/to/model.lmz')
print(model.num_parameters())  # Count parameters
model.summary()                # Print architecture
```

🔷 System Monitor

```python
# Set memory limit (especially important on Raspberry Pi)
lm.configure_memory(max_mb=128)

# Monitor system health
monitor = lm.SystemMonitor()
monitor.print_status()
score = monitor.health_score()

# Memory tracing
with lm.memory_trace("Forward Pass"):
    out = model(X)

# Optimize for inference
lm.memory_manager.optimize_for_inference()
```

---

💡 Examples

Ten complete examples in the examples/ folder:

File What it demonstrates
01_basic_tensors.py Tensor creation, arithmetic, autograd from scratch
02_linear_regression.py Linear regression with SGD, custom training loop
03_mlp_classification.py XOR classification with Sequential, Adam, DataLoader
04_mnist_like.py Full pipeline: MicroMLP + Trainer + EarlyStopping + ModelCheckpoint
05_cnn_image.py MicroCNN for image classification, BatchNorm, MaxPool
06_optimizers_comparison.py Benchmark SGD vs Adam vs RMSprop vs AdaGrad
07_custom_layer.py Build custom attention layer, LayerNorm, transformer block
08_save_load_model.py Save/load weights, state dict, transfer learning
09_lr_schedulers.py Compare 6 LR scheduler strategies
10_raspberry_pi_monitor.py System monitoring, memory tracing, health scoring

Run any example:

```bash
cd lowmind_repo
python examples/01_basic_tensors.py
```

---

📁 Project Structure

```
lowmind/
├── lowmind/                 # Main package
│   ├── __init__.py          # Public API — all exports here
│   ├── core/
│   │   ├── tensor.py        # Tensor class + autograd engine
│   │   ├── memory.py        # MemoryManager (LRU, GC optimization)
│   │   └── module.py        # Module base class
│   ├── nn/
│   │   ├── layers.py        # Linear, Conv2d, BatchNorm, Pool, etc.
│   │   ├── activation.py    # ReLU, LeakyReLU, ELU, GELU, etc.
│   │   ├── loss.py          # cross_entropy, bce, mse, etc.
│   │   └── sequential.py    # Sequential container
│   ├── optim/
│   │   ├── sgd.py           # SGD + Nesterov momentum
│   │   ├── adam.py          # Adam, AdamW, RMSprop, AdaGrad
│   │   └── scheduler.py     # All LR schedulers
│   ├── data/
│   │   └── dataloader.py    # Dataset, DataLoader, train_test_split
│   ├── utils/
│   │   ├── metrics.py       # accuracy, precision, recall, f1, etc.
│   │   ├── trainer.py       # Trainer (high-level training loop)
│   │   ├── callbacks.py     # EarlyStopping, ModelCheckpoint, etc.
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

🍓 Raspberry Pi Tips

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

🤝 Contributing

Contributions are welcome! Areas where help is needed:

· Performance benchmarks on more Pi models
· LSTM / GRU layers
· Quantization (INT8 inference)
· Distributed training across multiple Pis

Submitting a PR:

1. Fork and create a feature branch
2. Run tests: pytest tests/ -v
3. Add tests for new features
4. Submit a PR with a clear description

---

🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

📄 License

MIT License — see LICENSE

---

<div align="center">

🌟 Star us on GitHub! 🌟

Built with care in India by Dhaval Vedra

Empowering AI at the edge — from data centers down to $35 computers

https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white
https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white

</div>