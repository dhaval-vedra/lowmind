"""
LowMind — Ultra-Lightweight Deep Learning Framework v2.1.0
==========================================================

Designed for Raspberry Pi, embedded systems, and any resource-constrained device.
Built entirely on NumPy — no heavy dependencies.

Quick Start::

    import lowmind as lm
    import numpy as np

    model = lm.Sequential(
        lm.Linear(784, 128), lm.ReLU(), lm.Dropout(0.3), lm.Linear(128, 10),
    )
    optimizer = lm.Adam(model.parameters(), lr=1e-3)
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = lm.cross_entropy_loss(model(X_b), y_b)
        loss.backward()
        optimizer.step()
"""

__version__ = "2.1.0"
__author__ = "Dhaval Vedra"
__license__ = "MIT"

# ── Core ──────────────────────────────────────────────────────────────────────
from .core.tensor import (
    Tensor,
    zeros, ones, randn, rand, arange, from_numpy,
    cat, stack, clip_grad_norm,
)
from .core.memory import memory_manager, MemoryManager, configure_memory
from .core.module import Module
from .core.no_grad import no_grad, enable_grad

# ── Neural Network ─────────────────────────────────────────────────────────────
from .nn.layers import (
    Linear, Conv2d,
    BatchNorm1d, BatchNorm2d,
    MaxPool2d, AvgPool2d,
    Flatten, Dropout, Embedding,
)
from .nn.activation import (
    ReLU, LeakyReLU, ELU, GELU,
    Sigmoid, Tanh, Softmax, LogSoftmax,
)
from .nn.loss import (
    cross_entropy_loss, binary_cross_entropy_loss,
    mse_loss, mae_loss, huber_loss, nll_loss,
)
from .nn.sequential import Sequential
from .nn.rnn import LSTMCell, LSTM, GRUCell, GRU
from .nn.init import (
    xavier_uniform_, xavier_normal_,
    kaiming_uniform_, kaiming_normal_,
    orthogonal_, normal_, uniform_,
    constant_, zeros_, ones_, eye_,
    init_module,
)

# ── Optimizers ─────────────────────────────────────────────────────────────────
from .optim.sgd import SGD
from .optim.adam import Adam, AdamW, RMSprop, AdaGrad
from .optim.scheduler import (
    StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, ReduceLROnPlateau,
    LinearWarmupLR, CyclicLR,
)

# ── Data ───────────────────────────────────────────────────────────────────────
from .data.dataloader import Dataset, TensorDataset, DataLoader, train_test_split
from .data.transforms import (
    Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip,
    RandomCrop, CenterCrop, GaussianNoise, Cutout, ToTensor,
)

# ── Utilities ──────────────────────────────────────────────────────────────────
from .utils.metrics import (
    accuracy, top_k_accuracy, confusion_matrix,
    precision, recall, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
)
from .utils.trainer import Trainer
from .utils.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint,
    LRSchedulerCallback, History,
)
from .utils.monitor import SystemMonitor, memory_trace, RaspberryPiAdvancedMonitor
from .utils.profiler import ModelProfiler
from .utils.lr_finder import LRFinder

# ── Pre-built Models ───────────────────────────────────────────────────────────
from .models.micro_cnn import MicroMLP, MicroCNN, TinyResNet

# ── Sub-modules ────────────────────────────────────────────────────────────────
from . import nn, optim, data, utils, models

__all__ = [
    # Core
    "Tensor", "Module",
    "zeros", "ones", "randn", "rand", "arange", "from_numpy",
    "cat", "stack", "clip_grad_norm",
    "memory_manager", "MemoryManager", "configure_memory",
    "no_grad", "enable_grad",
    # Layers
    "Linear", "Conv2d", "BatchNorm1d", "BatchNorm2d",
    "MaxPool2d", "AvgPool2d", "Flatten", "Dropout", "Embedding",
    # Activations
    "ReLU", "LeakyReLU", "ELU", "GELU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax",
    # Losses
    "cross_entropy_loss", "binary_cross_entropy_loss",
    "mse_loss", "mae_loss", "huber_loss", "nll_loss",
    # Sequential
    "Sequential",
    # RNNs
    "LSTMCell", "LSTM", "GRUCell", "GRU",
    # Weight init
    "xavier_uniform_", "xavier_normal_",
    "kaiming_uniform_", "kaiming_normal_",
    "orthogonal_", "normal_", "uniform_",
    "constant_", "zeros_", "ones_", "eye_",
    "init_module",
    # Optimizers
    "SGD", "Adam", "AdamW", "RMSprop", "AdaGrad",
    # Schedulers
    "StepLR", "MultiStepLR", "ExponentialLR",
    "CosineAnnealingLR", "ReduceLROnPlateau", "LinearWarmupLR", "CyclicLR",
    # Data
    "Dataset", "TensorDataset", "DataLoader", "train_test_split",
    # Transforms
    "Compose", "Normalize", "RandomHorizontalFlip", "RandomVerticalFlip",
    "RandomCrop", "CenterCrop", "GaussianNoise", "Cutout", "ToTensor",
    # Metrics
    "accuracy", "top_k_accuracy", "confusion_matrix",
    "precision", "recall", "f1_score",
    "r2_score", "mean_squared_error", "mean_absolute_error",
    # Trainer
    "Trainer",
    # Callbacks
    "Callback", "EarlyStopping", "ModelCheckpoint",
    "LRSchedulerCallback", "History",
    # Monitoring
    "SystemMonitor", "memory_trace", "RaspberryPiAdvancedMonitor",
    # Profiler & LR Finder
    "ModelProfiler", "LRFinder",
    # Models
    "MicroMLP", "MicroCNN", "TinyResNet",
    # Sub-modules
    "nn", "optim", "data", "utils", "models",
]
