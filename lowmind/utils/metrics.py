"""
LowMind Metrics

accuracy, top_k_accuracy, precision, recall, f1_score,
confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
"""
import numpy as np
from ..core.tensor import Tensor


def _to_numpy(x):
    return x.data if isinstance(x, Tensor) else np.asarray(x)


def accuracy(predictions, targets):
    """
    Classification accuracy.

    Args:
        predictions: Logits (N, C) or predicted class indices (N,).
        targets:     True class indices (N,).

    Returns:
        Float in [0, 1].
    """
    p = _to_numpy(predictions)
    t = _to_numpy(targets).flatten().astype(int)
    if p.ndim == 2:
        p = p.argmax(axis=1)
    return float((p.astype(int) == t).mean())


def top_k_accuracy(logits, targets, k=5):
    """
    Top-k accuracy: 1 if the true label is in the top-k predictions.

    Args:
        logits:  (N, C) logits or probabilities.
        targets: (N,) true class indices.
        k:       Number of top predictions to consider.

    Returns:
        Float in [0, 1].
    """
    logits = _to_numpy(logits)
    targets = _to_numpy(targets).flatten().astype(int)
    top_k = np.argsort(logits, axis=1)[:, -k:]
    hits = np.array([t in top_k[i] for i, t in enumerate(targets)])
    return float(hits.mean())


def confusion_matrix(predictions, targets, num_classes=None):
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices (N,) or logits (N, C).
        targets:     True class indices (N,).
        num_classes: Total number of classes (inferred if None).

    Returns:
        numpy array of shape (num_classes, num_classes).
        cm[i, j] = # samples with true class i predicted as j.
    """
    p = _to_numpy(predictions)
    t = _to_numpy(targets).flatten().astype(int)
    if p.ndim == 2:
        p = p.argmax(axis=1)
    p = p.astype(int)
    if num_classes is None:
        num_classes = max(p.max(), t.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for ti, pi in zip(t, p):
        cm[ti, pi] += 1
    return cm


def precision(predictions, targets, num_classes=None, average='macro'):
    """
    Precision score.

    Args:
        predictions: Predicted indices or logits.
        targets:     True class indices.
        num_classes: Number of classes.
        average:     'macro', 'micro', or 'none'.

    Returns:
        Float (or array if average='none').
    """
    cm = confusion_matrix(predictions, targets, num_classes)
    prec = np.diag(cm) / (cm.sum(axis=0) + 1e-9)
    if average == 'macro':
        return float(prec.mean())
    elif average == 'micro':
        tp = np.diag(cm).sum()
        fp = cm.sum() - cm.sum(axis=0).sum()
        return float(tp / (tp + fp + 1e-9))
    return prec


def recall(predictions, targets, num_classes=None, average='macro'):
    """
    Recall score.

    Args:
        predictions: Predicted indices or logits.
        targets:     True class indices.
        num_classes: Number of classes.
        average:     'macro', 'micro', or 'none'.

    Returns:
        Float (or array if average='none').
    """
    cm = confusion_matrix(predictions, targets, num_classes)
    rec = np.diag(cm) / (cm.sum(axis=1) + 1e-9)
    if average == 'macro':
        return float(rec.mean())
    elif average == 'micro':
        tp = np.diag(cm).sum()
        fn = cm.sum(axis=1).sum() - tp
        return float(tp / (tp + fn + 1e-9))
    return rec


def f1_score(predictions, targets, num_classes=None, average='macro'):
    """
    F1 score (harmonic mean of precision and recall).

    Args:
        predictions: Predicted indices or logits.
        targets:     True class indices.
        num_classes: Number of classes.
        average:     'macro', 'micro', or 'none'.

    Returns:
        Float (or array if average='none').
    """
    prec = precision(predictions, targets, num_classes, average)
    rec = recall(predictions, targets, num_classes, average)
    if isinstance(prec, np.ndarray):
        return 2 * prec * rec / (prec + rec + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def r2_score(predictions, targets):
    """
    R² coefficient of determination for regression.

    Returns:
        Float — 1.0 is perfect, 0.0 means predicting the mean, <0 is worse.
    """
    p = _to_numpy(predictions).flatten()
    t = _to_numpy(targets).flatten()
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


def mean_squared_error(predictions, targets):
    """MSE between predictions and targets."""
    p = _to_numpy(predictions).flatten()
    t = _to_numpy(targets).flatten()
    return float(np.mean((p - t) ** 2))


def mean_absolute_error(predictions, targets):
    """MAE between predictions and targets."""
    p = _to_numpy(predictions).flatten()
    t = _to_numpy(targets).flatten()
    return float(np.mean(np.abs(p - t)))
