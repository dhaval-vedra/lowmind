# LowMind Changelog

All notable changes to LowMind are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.1.0] — 2026-05-27

### Added
- **`no_grad` / `enable_grad`** context managers — disable autograd during inference, works as decorator too
- **LSTM & GRU** layers (single-cell + multi-step, multi-layer, dropout support)
- **Weight initialization** module (`xavier_uniform_`, `xavier_normal_`, `kaiming_uniform_`, `kaiming_normal_`, `orthogonal_`, `normal_`, `uniform_`, `constant_`, `zeros_`, `ones_`, `eye_`, `init_module`)
- **Parameter groups** support in all optimizers (SGD, Adam, AdamW, RMSprop, AdaGrad) — different LR/weight-decay per layer
- **Gradient accumulation** in `Trainer` (`grad_accum_steps` argument) — simulate large batches on Pi
- **Model Profiler** (`ModelProfiler`) — count parameters, estimate FLOPs/memory, measure throughput
- **Data Transforms** — `Compose`, `Normalize`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCrop`, `CenterCrop`, `GaussianNoise`, `Cutout`, `ToTensor`
- **LR Finder** (`LRFinder`) — auto-detect optimal learning rate using range test
- **Optimizer `state_dict` / `load_state_dict`** — save and restore optimizer state for resumable training
- **`Trainer.predict_proba`** — return softmax probabilities from the high-level trainer
- 3 new examples: 11 (LSTM sequence), 12 (weight init comparison), 13 (production inference pipeline)
- Integration test suite (`tests/test_integration.py`) — 10 end-to-end tests
- RNN test suite (`tests/test_rnn.py`) — 13 LSTM/GRU/no_grad tests
- Weight init test suite (`tests/test_init.py`) — 14 init tests
- `pyproject.toml` — PEP 621 production packaging config

### Changed
- All optimizers now support **parameter groups** (list of dicts API, same as PyTorch)
- `Trainer` now uses `no_grad` during validation for memory efficiency
- `no_grad` is integrated into `Tensor.__init__` — computed tensors inside `no_grad` skip graph building
- `clip_grad_norm` moved to top-level `lm.clip_grad_norm()`

### Fixed
- `set_lr` / `get_lr` added to all optimizers (needed by `LRFinder` and `ReduceLROnPlateau`)

---

## [2.0.0] — 2026-05-26

### Added
- Full professional package structure (`core/`, `nn/`, `optim/`, `data/`, `utils/`, `models/`)
- **Adam**, **AdamW**, **RMSprop**, **AdaGrad** optimizers
- **7 LR Schedulers**: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `LinearWarmupLR`, `CyclicLR`
- **BatchNorm1d / BatchNorm2d** with running stats and eval mode
- **MaxPool2d / AvgPool2d** with stride and kernel
- **Conv2d** im2col-based backward pass
- **Flatten**, **Dropout**, **Embedding** layers
- **6 Loss functions**: CrossEntropy, BCE, MSE, MAE, Huber, NLL
- **DataLoader** with shuffling, batching, drop_last; `TensorDataset`, `train_test_split`
- **8 Metrics**: accuracy, top-k accuracy, F1, precision, recall, confusion matrix, R², MSE/MAE
- **High-level Trainer** with EarlyStopping, ModelCheckpoint, LRSchedulerCallback, History callbacks
- **SystemMonitor** and `memory_trace` for Pi resource monitoring
- **MicroMLP, MicroCNN, TinyResNet** pre-built models
- Model `save()` / `load()` with gzip compression
- 10 example scripts covering regression, classification, CNN, optimizers, schedulers, save/load
- 88 passing unit tests

### Fixed (from original `lowmind.py`)
- `Conv2d` had no backward pass → full im2col gradient implemented
- `mean(axis=(2,3))` crashed with tuple axis → full tuple-axis support added
- `Tensor.__hash__` missing → set/dict usage failed → `__hash__ = object.__hash__`
- Matmul backward 1-D vector case → was scalar instead of outer product → `np.outer` fix
- `zero_grad()` used `None` assignment instead of `.fill(0)` → fixed
- Memory manager name collision for same-named layers → removed from tracking

---

## [1.0.0] — original

Single-file `lowmind.py` by Dhaval Vedra.
Basic Tensor, Linear, Conv2d, ReLU, MSE loss, SGD optimizer.
