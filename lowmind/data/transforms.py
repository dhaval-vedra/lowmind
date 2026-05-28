"""
LowMind Data Transforms — lightweight image augmentation

Compose         — chain multiple transforms
Normalize       — zero-mean, unit-std normalization
RandomHorizontalFlip
RandomVerticalFlip
RandomCrop
CenterCrop
RandomRotate90
GaussianNoise
Cutout          — randomly mask patches
ToTensor        — numpy HWC → Tensor CHW
"""
import numpy as np
from ..core.tensor import Tensor


class Compose:
    """
    Compose multiple transforms in sequence.

    Args:
        transforms: List of transform callables.

    Example::

        transform = lm.Compose([
            lm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            lm.RandomHorizontalFlip(p=0.5),
        ])
        X_aug = transform(X)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        ts = "\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {ts}\n])"


class Normalize:
    """
    Normalize an array / Tensor to zero mean and unit standard deviation.

    Args:
        mean: Per-channel mean (scalar or list). Applied channel-wise for CHW.
        std:  Per-channel std  (scalar or list).

    Input shapes supported:
        (C, H, W) — single image
        (N, C, H, W) — batch of images
        (N, F)        — flat features

    Example::

        norm = lm.Normalize(mean=0.5, std=0.5)
        X_norm = norm(X)
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, x):
        data = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
        if data.ndim == 4 and self.mean.ndim > 0:  # (N, C, H, W)
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
        elif data.ndim == 3 and self.mean.ndim > 0:  # (C, H, W)
            mean = self.mean.reshape(-1, 1, 1)
            std = self.std.reshape(-1, 1, 1)
        else:
            mean, std = self.mean, self.std
        normed = (data - mean) / (std + 1e-8)
        return Tensor(normed) if isinstance(x, Tensor) else normed

    def __repr__(self):
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()})"


class RandomHorizontalFlip:
    """
    Randomly flip images horizontally with probability p.

    Input: (N, C, H, W) batch or (C, H, W) single image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        if data.ndim == 4:
            mask = np.random.rand(data.shape[0]) < self.p
            out = data.copy()
            out[mask] = out[mask, :, :, ::-1]
        elif data.ndim == 3:
            out = data[:, :, ::-1].copy() if np.random.rand() < self.p else data.copy()
        else:
            out = data.copy()
        return Tensor(out) if isinstance(x, Tensor) else out

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """Randomly flip images vertically with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        if data.ndim == 4:
            mask = np.random.rand(data.shape[0]) < self.p
            out = data.copy()
            out[mask] = out[mask, :, ::-1, :]
        elif data.ndim == 3:
            out = data[:, ::-1, :].copy() if np.random.rand() < self.p else data.copy()
        else:
            out = data.copy()
        return Tensor(out) if isinstance(x, Tensor) else out

    def __repr__(self):
        return f"RandomVerticalFlip(p={self.p})"


class RandomCrop:
    """
    Randomly crop images to (size × size) pixels.

    Args:
        size:    Output spatial size.
        padding: Pad image before cropping (default 0).

    Input: (N, C, H, W) or (C, H, W).
    """

    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        if self.padding > 0:
            p = self.padding
            if data.ndim == 4:
                data = np.pad(data, ((0, 0), (0, 0), (p, p), (p, p)))
            elif data.ndim == 3:
                data = np.pad(data, ((0, 0), (p, p), (p, p)))

        s = self.size
        if data.ndim == 4:
            H, W = data.shape[2], data.shape[3]
            out = np.zeros((data.shape[0], data.shape[1], s, s), dtype=np.float32)
            for i in range(data.shape[0]):
                th = np.random.randint(0, H - s + 1)
                tw = np.random.randint(0, W - s + 1)
                out[i] = data[i, :, th:th + s, tw:tw + s]
        elif data.ndim == 3:
            H, W = data.shape[1], data.shape[2]
            th = np.random.randint(0, H - s + 1)
            tw = np.random.randint(0, W - s + 1)
            out = data[:, th:th + s, tw:tw + s].copy()
        else:
            out = data.copy()

        return Tensor(out) if isinstance(x, Tensor) else out

    def __repr__(self):
        return f"RandomCrop(size={self.size}, padding={self.padding})"


class CenterCrop:
    """Crop the center (size × size) region of an image."""

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        s = self.size
        if data.ndim == 4:
            H, W = data.shape[2], data.shape[3]
            th, tw = (H - s) // 2, (W - s) // 2
            out = data[:, :, th:th + s, tw:tw + s].copy()
        elif data.ndim == 3:
            H, W = data.shape[1], data.shape[2]
            th, tw = (H - s) // 2, (W - s) // 2
            out = data[:, th:th + s, tw:tw + s].copy()
        else:
            out = data.copy()
        return Tensor(out) if isinstance(x, Tensor) else out

    def __repr__(self):
        return f"CenterCrop(size={self.size})"


class GaussianNoise:
    """
    Add Gaussian noise to inputs (good for regularization / robustness).

    Args:
        std: Standard deviation of the noise (default 0.1).
        p:   Probability of applying noise (default 1.0).
    """

    def __init__(self, std=0.1, p=1.0):
        self.std = std
        self.p = p

    def __call__(self, x):
        data = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
        if np.random.rand() < self.p:
            noise = (np.random.randn(*data.shape) * self.std).astype(np.float32)
            data = data + noise
        return Tensor(data) if isinstance(x, Tensor) else data

    def __repr__(self):
        return f"GaussianNoise(std={self.std}, p={self.p})"


class Cutout:
    """
    Randomly mask square patches in images (DeVries & Taylor 2017).

    Args:
        n_holes: Number of patches to cut out.
        length:  Side length of each patch.

    Input: (N, C, H, W) or (C, H, W).
    """

    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, x):
        data = x.data.copy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32).copy()
        if data.ndim == 3:
            H, W = data.shape[1], data.shape[2]
            for _ in range(self.n_holes):
                y = np.random.randint(H)
                xc = np.random.randint(W)
                y1, y2 = max(0, y - self.length // 2), min(H, y + self.length // 2)
                x1, x2 = max(0, xc - self.length // 2), min(W, xc + self.length // 2)
                data[:, y1:y2, x1:x2] = 0
        elif data.ndim == 4:
            H, W = data.shape[2], data.shape[3]
            for i in range(data.shape[0]):
                for _ in range(self.n_holes):
                    y = np.random.randint(H)
                    xc = np.random.randint(W)
                    y1, y2 = max(0, y - self.length // 2), min(H, y + self.length // 2)
                    x1, x2 = max(0, xc - self.length // 2), min(W, xc + self.length // 2)
                    data[i, :, y1:y2, x1:x2] = 0
        return Tensor(data) if isinstance(x, Tensor) else data

    def __repr__(self):
        return f"Cutout(n_holes={self.n_holes}, length={self.length})"


class ToTensor:
    """
    Convert a numpy array (H, W, C) uint8 image to a float32 Tensor (C, H, W)
    and scale to [0, 1].

    Useful for converting PIL / OpenCV images to LowMind format.
    """

    def __call__(self, x):
        data = np.asarray(x, dtype=np.float32)
        if data.ndim == 3 and data.shape[2] <= 4:  # HWC → CHW
            data = data.transpose(2, 0, 1)
        if data.max() > 1.0:
            data /= 255.0
        return Tensor(data)

    def __repr__(self):
        return "ToTensor()"
