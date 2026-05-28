"""
LowMind Loss Functions

Available:
    cross_entropy_loss  — multi-class classification
    binary_cross_entropy_loss — binary classification (sigmoid output)
    mse_loss            — regression (Mean Squared Error)
    mae_loss            — regression (Mean Absolute Error)
    huber_loss          — regression (smooth L1)
    nll_loss            — Negative Log-Likelihood (use after log-softmax)
"""
import numpy as np
from ..core.tensor import Tensor


def cross_entropy_loss(output: Tensor, target: Tensor, reduction='mean') -> Tensor:
    """
    Cross-Entropy Loss (combines log-softmax + NLL).

    Args:
        output:    Logits tensor of shape (N, C).
        target:    Integer class indices tensor of shape (N,).
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss Tensor.
    """
    N = output.data.shape[0]
    # Numerically stable softmax
    shifted = output.data - output.data.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)

    target_idx = target.data.astype(int).flatten()
    log_probs = -np.log(probs[np.arange(N), target_idx] + 1e-9)
    loss_val = log_probs.mean() if reduction == 'mean' else log_probs.sum()

    loss = Tensor(np.array([loss_val], dtype=np.float32),
                  requires_grad=output.requires_grad,
                  _children=(output,), _op='cross_entropy')

    def _backward():
        if output.requires_grad:
            output._ensure_grad()
            grad = probs.copy()
            grad[np.arange(N), target_idx] -= 1
            grad = grad / N if reduction == 'mean' else grad
            output.grad += grad

    loss._backward = _backward
    return loss


def binary_cross_entropy_loss(output: Tensor, target: Tensor,
                               from_logits=False, reduction='mean') -> Tensor:
    """
    Binary Cross-Entropy Loss.

    Args:
        output:      Predictions — probabilities in [0,1] or raw logits.
        target:      Binary labels (0 or 1), same shape as output.
        from_logits: If True, applies sigmoid first.
        reduction:   'mean' or 'sum'.

    Returns:
        Scalar loss Tensor.
    """
    eps = 1e-9
    if from_logits:
        probs = 1.0 / (1.0 + np.exp(-output.data))
    else:
        probs = output.data

    t = target.data
    loss_val = -(t * np.log(probs + eps) + (1 - t) * np.log(1 - probs + eps))
    loss_scalar = loss_val.mean() if reduction == 'mean' else loss_val.sum()

    loss = Tensor(np.array([loss_scalar], dtype=np.float32),
                  requires_grad=output.requires_grad,
                  _children=(output,), _op='bce')

    def _backward():
        if output.requires_grad:
            output._ensure_grad()
            dprobs = -(t / (probs + eps) - (1 - t) / (1 - probs + eps))
            if from_logits:
                dpred = dprobs * probs * (1 - probs)
            else:
                dpred = dprobs
            n = output.data.size if reduction == 'mean' else 1
            output.grad += dpred / n

    loss._backward = _backward
    return loss


def mse_loss(output: Tensor, target: Tensor, reduction='mean') -> Tensor:
    """
    Mean Squared Error loss.

    Args:
        output:    Predictions.
        target:    Ground-truth values.
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss Tensor.
    """
    diff = output - target
    sq = diff * diff
    return sq.mean() if reduction == 'mean' else sq.sum()


def mae_loss(output: Tensor, target: Tensor, reduction='mean') -> Tensor:
    """
    Mean Absolute Error loss.

    Args:
        output:    Predictions.
        target:    Ground-truth values.
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss Tensor.
    """
    diff = (output - target).abs()
    return diff.mean() if reduction == 'mean' else diff.sum()


def huber_loss(output: Tensor, target: Tensor, delta=1.0, reduction='mean') -> Tensor:
    """
    Huber (smooth L1) loss — less sensitive to outliers than MSE.

    Args:
        output:    Predictions.
        target:    Ground-truth values.
        delta:     Threshold between quadratic and linear region.
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss Tensor.
    """
    diff_data = np.abs(output.data - target.data)
    loss_data = np.where(
        diff_data < delta,
        0.5 * diff_data ** 2,
        delta * (diff_data - 0.5 * delta)
    )
    loss_scalar = loss_data.mean() if reduction == 'mean' else loss_data.sum()
    loss = Tensor(np.array([loss_scalar], dtype=np.float32),
                  requires_grad=output.requires_grad,
                  _children=(output,), _op='huber')

    def _backward():
        if output.requires_grad:
            output._ensure_grad()
            diff_signed = output.data - target.data
            grad = np.where(
                np.abs(diff_signed) < delta,
                diff_signed,
                delta * np.sign(diff_signed)
            )
            n = output.data.size if reduction == 'mean' else 1
            output.grad += grad / n

    loss._backward = _backward
    return loss


def nll_loss(log_probs: Tensor, target: Tensor, reduction='mean') -> Tensor:
    """
    Negative Log-Likelihood loss (use after LogSoftmax).

    Args:
        log_probs: Log-probabilities of shape (N, C).
        target:    Class indices of shape (N,).
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss Tensor.
    """
    N = log_probs.data.shape[0]
    target_idx = target.data.astype(int).flatten()
    selected = log_probs.data[np.arange(N), target_idx]
    loss_val = -selected.mean() if reduction == 'mean' else -selected.sum()

    loss = Tensor(np.array([loss_val], dtype=np.float32),
                  requires_grad=log_probs.requires_grad,
                  _children=(log_probs,), _op='nll')

    def _backward():
        if log_probs.requires_grad:
            log_probs._ensure_grad()
            grad = np.zeros_like(log_probs.data)
            grad[np.arange(N), target_idx] = -1.0 / (N if reduction == 'mean' else 1)
            log_probs.grad += grad

    loss._backward = _backward
    return loss
