"""
LowMind Data Utilities

Dataset     — abstract base class
TensorDataset    — wraps Tensors
DataLoader       — batched + shuffled iteration
train_test_split — split arrays into train/val sets
"""
import numpy as np
from ..core.tensor import Tensor


class Dataset:
    """
    Abstract base class for all datasets.
    Subclass and implement `__len__` and `__getitem__`.

    Example::

        class MyDataset(lm.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(len={len(self)})"


class TensorDataset(Dataset):
    """
    Dataset wrapping multiple numpy arrays or Tensors.
    All arrays must have the same first dimension.

    Args:
        *tensors: Tensors or numpy arrays of the same length.

    Example::

        ds = lm.TensorDataset(X_train, y_train)
    """

    def __init__(self, *tensors):
        lengths = [len(t) for t in tensors]
        assert all(l == lengths[0] for l in lengths), "All tensors must have the same length"
        self.tensors = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)


class DataLoader:
    """
    Iterable over a Dataset in batches.

    Args:
        dataset:    A Dataset instance.
        batch_size: Number of samples per batch (default 32).
        shuffle:    Shuffle before each epoch (default False).
        drop_last:  Drop incomplete last batch (default False).
        collate_fn: Custom collation function (default: numpy-to-Tensor).

    Example::

        loader = lm.DataLoader(ds, batch_size=64, shuffle=True)
        for X_batch, y_batch in loader:
            ...  # X_batch and y_batch are Tensors
    """

    def __init__(self, dataset, batch_size=32, shuffle=False,
                 drop_last=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or self._default_collate

    @staticmethod
    def _default_collate(batch):
        """Stack a list of samples into Tensor batches."""
        if isinstance(batch[0], (tuple, list)):
            return tuple(Tensor(np.stack([s[i] for s in batch], axis=0))
                         for i in range(len(batch[0])))
        return Tensor(np.stack(batch, axis=0))

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        n = len(indices)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n:
                if self.drop_last:
                    break
                end = n
            batch_idx = indices[start:end]
            samples = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(samples)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def train_test_split(*arrays, test_size=0.2, shuffle=True, seed=None, random_state=None):
    """
    Split arrays or Datasets into train and test subsets.

    Args:
        *arrays:      Arrays, Tensors, or a single Dataset of the same length.
        test_size:    Fraction for test set (default 0.2).
        shuffle:      Shuffle before splitting (default True).
        seed:         Random seed (default None).
        random_state: Alias for ``seed`` (for sklearn compatibility).

    Returns:
        If given arrays/Tensors: tuple of (train, test) Tensor pairs.
        If given a single Dataset: (train_dataset, test_dataset).

    Example::

        X_train, X_val, y_train, y_val = lm.train_test_split(X, y, test_size=0.2)
        train_ds, val_ds = lm.train_test_split(dataset, test_size=0.25)
    """
    effective_seed = random_state if random_state is not None else seed
    if effective_seed is not None:
        np.random.seed(effective_seed)

    # Handle single Dataset argument
    if len(arrays) == 1 and isinstance(arrays[0], Dataset):
        ds = arrays[0]
        n = len(ds)
        n_test = int(n * test_size)
        n_train = n - n_test
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        train_idx = indices[:n_train].tolist()
        test_idx  = indices[n_train:].tolist()
        class _Subset(Dataset):
            def __init__(self, dataset, idx): self._ds, self._idx = dataset, idx
            def __len__(self): return len(self._idx)
            def __getitem__(self, i): return self._ds[self._idx[i]]
        return _Subset(ds, train_idx), _Subset(ds, test_idx)

    arrs = [a.data if isinstance(a, Tensor) else np.asarray(a) for a in arrays]
    n = len(arrs[0])
    n_test = int(n * test_size)
    n_train = n - n_test

    indices = np.random.permutation(n) if shuffle else np.arange(n)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    result = []
    for arr in arrs:
        result.append(Tensor(arr[train_idx]))
        result.append(Tensor(arr[test_idx]))
    return tuple(result)
