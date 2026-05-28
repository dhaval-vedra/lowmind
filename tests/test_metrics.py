"""Tests for metrics and loss functions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import lowmind as lm


class TestLossFunctions:
    def test_cross_entropy_perfect(self):
        logits = lm.Tensor(np.array([[100., -100.], [-100., 100.]], dtype=np.float32))
        target = lm.Tensor(np.array([0, 1]))
        loss = lm.cross_entropy_loss(logits, target)
        assert float(loss.item()) < 0.01

    def test_mse_zero(self):
        x = lm.Tensor([1., 2., 3.], requires_grad=True)
        loss = lm.mse_loss(x, lm.Tensor([1., 2., 3.]))
        assert float(loss.item()) == pytest.approx(0.0)

    def test_mse_gradient(self):
        x = lm.Tensor([3., 4.], requires_grad=True)
        y = lm.Tensor([1., 2.])
        loss = lm.mse_loss(x, y)
        loss.backward()
        # grad = 2*(x-y)/N = [4/2, 4/2] = [2, 2]
        np.testing.assert_allclose(x.grad, [2., 2.], rtol=1e-5)

    def test_mae_loss(self):
        x = lm.Tensor([1., 2., 3.], requires_grad=True)
        y = lm.Tensor([0., 0., 0.])
        loss = lm.mae_loss(x, y)
        assert float(loss.item()) == pytest.approx(2.0)

    def test_huber_loss_small(self):
        x = lm.Tensor([0.5], requires_grad=True)
        y = lm.Tensor([0.])
        loss = lm.huber_loss(x, y, delta=1.0)
        # |0.5| < 1 → 0.5 * 0.5^2 = 0.125
        assert float(loss.item()) == pytest.approx(0.125, rel=1e-4)

    def test_bce_loss(self):
        x = lm.Tensor([0.9, 0.1], requires_grad=True)
        y = lm.Tensor([1.0, 0.0])
        loss = lm.binary_cross_entropy_loss(x, y)
        assert float(loss.item()) < 0.2   # near-perfect predictions

    def test_cross_entropy_backward(self):
        logits = lm.Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
        target = lm.Tensor(np.array([0, 1, 2, 0]))
        loss = lm.cross_entropy_loss(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (4, 3)


class TestMetrics:
    def _preds(self):
        # Perfect predictions for class 0 and 1
        logits = np.array([
            [10., -10.],  # class 0
            [-10., 10.],  # class 1
            [10., -10.],  # class 0
            [-10., 10.],  # class 1
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1])
        return lm.Tensor(logits), lm.Tensor(labels)

    def test_accuracy_perfect(self):
        preds, targets = self._preds()
        assert lm.accuracy(preds, targets) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        preds = lm.Tensor(np.array([[10., -10.], [10., -10.]], dtype=np.float32))
        targets = lm.Tensor(np.array([1, 1]))
        assert lm.accuracy(preds, targets) == pytest.approx(0.0)

    def test_top_k_accuracy(self):
        logits = lm.Tensor(np.eye(5, dtype=np.float32))  # perfect top-1
        targets = lm.Tensor(np.arange(5))
        assert lm.top_k_accuracy(logits, targets, k=1) == pytest.approx(1.0)
        assert lm.top_k_accuracy(logits, targets, k=3) == pytest.approx(1.0)

    def test_confusion_matrix(self):
        preds, targets = self._preds()
        cm = lm.confusion_matrix(preds, targets, num_classes=2)
        # Perfect: [[2,0],[0,2]]
        np.testing.assert_array_equal(cm, [[2, 0], [0, 2]])

    def test_precision_recall_f1_perfect(self):
        preds, targets = self._preds()
        assert lm.precision(preds, targets, num_classes=2) == pytest.approx(1.0, rel=1e-4)
        assert lm.recall(preds, targets, num_classes=2) == pytest.approx(1.0, rel=1e-4)
        assert lm.f1_score(preds, targets, num_classes=2) == pytest.approx(1.0, rel=1e-4)

    def test_r2_score_perfect(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        assert lm.r2_score(lm.Tensor(x), lm.Tensor(x)) == pytest.approx(1.0, rel=1e-4)

    def test_r2_score_mean_predictor(self):
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        mean = np.full_like(x, x.mean())
        assert lm.r2_score(lm.Tensor(mean), lm.Tensor(x)) == pytest.approx(0.0, abs=1e-4)

    def test_mse_metric(self):
        x = lm.Tensor([1., 2., 3.])
        y = lm.Tensor([2., 3., 4.])
        assert lm.mean_squared_error(x, y) == pytest.approx(1.0)

    def test_mae_metric(self):
        x = lm.Tensor([1., 2., 3.])
        y = lm.Tensor([2., 3., 4.])
        assert lm.mean_absolute_error(x, y) == pytest.approx(1.0)
