"""Tests for LowMind Tensor class and autograd."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import lowmind as lm


class TestTensorCreation:
    def test_from_list(self):
        t = lm.Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.dtype == np.float32

    def test_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        t = lm.from_numpy(arr)
        assert t.dtype == np.float32

    def test_factory_zeros(self):
        t = lm.zeros(3, 4)
        assert t.shape == (3, 4)
        assert np.all(t.data == 0)

    def test_factory_ones(self):
        t = lm.ones(2, 3)
        assert np.all(t.data == 1)

    def test_factory_randn(self):
        t = lm.randn(10, 10)
        assert t.shape == (10, 10)

    def test_requires_grad_false_by_default(self):
        t = lm.Tensor([1.0])
        assert not t.requires_grad

    def test_requires_grad_true(self):
        t = lm.Tensor([1.0], requires_grad=True)
        assert t.requires_grad
        assert t.grad is None  # lazy init


class TestArithmetic:
    def test_add(self):
        a = lm.Tensor([1.0, 2.0])
        b = lm.Tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_allclose(c.data, [4.0, 6.0])

    def test_sub(self):
        a = lm.Tensor([5.0, 6.0])
        b = lm.Tensor([1.0, 2.0])
        np.testing.assert_allclose((a - b).data, [4.0, 4.0])

    def test_mul(self):
        a = lm.Tensor([2.0, 3.0])
        b = lm.Tensor([4.0, 5.0])
        np.testing.assert_allclose((a * b).data, [8.0, 15.0])

    def test_div(self):
        a = lm.Tensor([6.0, 8.0])
        b = lm.Tensor([2.0, 4.0])
        np.testing.assert_allclose((a / b).data, [3.0, 2.0], rtol=1e-5)

    def test_pow(self):
        a = lm.Tensor([2.0, 3.0])
        np.testing.assert_allclose((a ** 2).data, [4.0, 9.0])

    def test_neg(self):
        a = lm.Tensor([1.0, -2.0])
        np.testing.assert_allclose((-a).data, [-1.0, 2.0])

    def test_matmul(self):
        A = lm.Tensor(np.eye(3, dtype=np.float32))
        B = lm.Tensor(np.arange(9, dtype=np.float32).reshape(3, 3))
        np.testing.assert_allclose((A @ B).data, B.data)

    def test_scalar_add(self):
        a = lm.Tensor([1.0, 2.0])
        np.testing.assert_allclose((a + 1).data, [2.0, 3.0])

    def test_scalar_mul(self):
        a = lm.Tensor([3.0, 4.0])
        np.testing.assert_allclose((2 * a).data, [6.0, 8.0])


class TestReductions:
    def test_sum(self):
        a = lm.Tensor([[1., 2.], [3., 4.]])
        assert float(a.sum().data) == pytest.approx(10.0)

    def test_sum_axis(self):
        a = lm.Tensor([[1., 2.], [3., 4.]])
        np.testing.assert_allclose(a.sum(axis=0).data, [4., 6.])

    def test_mean(self):
        a = lm.Tensor([[1., 2.], [3., 4.]])
        assert float(a.mean().data) == pytest.approx(2.5)

    def test_mean_tuple_axis(self):
        # FIX: this was broken in original
        a = lm.Tensor(np.ones((2, 3, 4, 4), dtype=np.float32))
        out = a.mean(axis=(2, 3))
        assert out.shape == (2, 3)

    def test_max(self):
        a = lm.Tensor([1., 5., 3.])
        assert float(a.max().data) == pytest.approx(5.0)


class TestAutograd:
    def test_backward_scalar(self):
        x = lm.Tensor(3.0, requires_grad=True)
        y = x * x
        y.backward()
        assert x.grad == pytest.approx(6.0)   # dy/dx = 2x = 6

    def test_backward_add(self):
        a = lm.Tensor([1., 2., 3.], requires_grad=True)
        b = lm.Tensor([4., 5., 6.], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad, [1., 1., 1.])
        np.testing.assert_allclose(b.grad, [1., 1., 1.])

    def test_backward_mul(self):
        a = lm.Tensor([2., 3.], requires_grad=True)
        b = lm.Tensor([4., 5.], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad, b.data)
        np.testing.assert_allclose(b.grad, a.data)

    def test_backward_matmul(self):
        W = lm.Tensor(np.eye(3, dtype=np.float32), requires_grad=True)
        x = lm.Tensor(np.ones((3,), dtype=np.float32), requires_grad=True)
        out = (W @ x).sum()
        out.backward()
        np.testing.assert_allclose(W.grad, np.ones((3, 3), dtype=np.float32))

    def test_backward_relu(self):
        x = lm.Tensor([-1., 0., 2.], requires_grad=True)
        out = x.relu().sum()
        out.backward()
        np.testing.assert_allclose(x.grad, [0., 0., 1.])

    def test_backward_sigmoid(self):
        x = lm.Tensor([0.0], requires_grad=True)
        s = x.sigmoid()
        s.backward()
        # ds/dx at x=0 = 0.25
        assert x.grad[0] == pytest.approx(0.25, rel=1e-4)

    def test_backward_sum_axis(self):
        x = lm.Tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
        out = x.sum(axis=1)   # (3,)
        loss = out.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, np.ones((3, 4)))

    def test_backward_mean_tuple_axis(self):
        # Key regression test for tuple-axis mean
        x = lm.Tensor(np.ones((2, 3, 4, 4), dtype=np.float32), requires_grad=True)
        out = x.mean(axis=(2, 3))   # (2, 3)
        out.sum().backward()
        assert x.grad.shape == (2, 3, 4, 4)
        np.testing.assert_allclose(x.grad, np.full((2, 3, 4, 4), 1.0 / 16))

    def test_no_grad_raises(self):
        x = lm.Tensor([1.0])
        with pytest.raises(RuntimeError):
            x.backward()

    def test_chain_rule(self):
        x = lm.Tensor(2.0, requires_grad=True)
        y = (x ** 3).sum()     # dy/dx = 3x² = 12
        y.backward()
        assert x.grad == pytest.approx(12.0)


class TestShapeOps:
    def test_reshape(self):
        x = lm.Tensor(np.arange(12, dtype=np.float32))
        r = x.reshape(3, 4)
        assert r.shape == (3, 4)

    def test_flatten(self):
        x = lm.Tensor(np.ones((2, 3, 4), dtype=np.float32))
        f = x.flatten(1)
        assert f.shape == (2, 12)

    def test_transpose(self):
        x = lm.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
        t = x.T
        assert t.shape == (3, 2)

    def test_squeeze_unsqueeze(self):
        x = lm.Tensor(np.ones((3, 1, 4)))
        assert x.squeeze(1).shape == (3, 4)
        assert x.unsqueeze(0).shape == (1, 3, 1, 4)

    def test_cat(self):
        a = lm.Tensor(np.ones((2, 3), dtype=np.float32))
        b = lm.Tensor(np.ones((2, 3), dtype=np.float32) * 2)
        c = lm.cat([a, b], axis=0)
        assert c.shape == (4, 3)
