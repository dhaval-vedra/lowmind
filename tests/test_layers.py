"""Tests for neural network layers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import lowmind as lm


class TestLinear:
    def test_output_shape(self):
        layer = lm.Linear(10, 5)
        x = lm.Tensor(np.random.randn(4, 10).astype(np.float32))
        out = layer(x)
        assert out.shape == (4, 5)

    def test_no_bias(self):
        layer = lm.Linear(8, 4, bias=False)
        x = lm.Tensor(np.random.randn(2, 8).astype(np.float32))
        out = layer(x)
        assert out.shape == (2, 4)

    def test_gradient_flows(self):
        layer = lm.Linear(4, 2)
        x = lm.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None


class TestConv2d:
    def test_output_shape(self):
        conv = lm.Conv2d(3, 16, kernel_size=3, padding=1)
        x = lm.Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        out = conv(x)
        assert out.shape == (2, 16, 8, 8)

    def test_output_shape_no_padding(self):
        conv = lm.Conv2d(1, 4, kernel_size=3)
        x = lm.Tensor(np.random.randn(1, 1, 10, 10).astype(np.float32))
        out = conv(x)
        assert out.shape == (1, 4, 8, 8)

    def test_stride(self):
        conv = lm.Conv2d(1, 4, kernel_size=3, stride=2)
        x = lm.Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
        out = conv(x)
        assert out.shape == (1, 4, 3, 3)

    def test_gradient_flows(self):
        conv = lm.Conv2d(1, 2, 3, padding=1)
        x = lm.Tensor(np.random.randn(2, 1, 6, 6).astype(np.float32), requires_grad=True)
        loss = conv(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert conv.weight.grad is not None


class TestBatchNorm:
    def test_bn1d_output_shape(self):
        bn = lm.BatchNorm1d(8)
        x = lm.Tensor(np.random.randn(4, 8).astype(np.float32))
        out = bn(x)
        assert out.shape == (4, 8)

    def test_bn1d_normalizes(self):
        bn = lm.BatchNorm1d(4)
        bn.train()
        x = lm.Tensor(np.random.randn(32, 4).astype(np.float32) * 10 + 5)
        out = bn(x)
        np.testing.assert_allclose(out.data.mean(axis=0), np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(out.data.var(axis=0), np.ones(4), atol=1e-4)

    def test_bn2d_output_shape(self):
        bn = lm.BatchNorm2d(16)
        x = lm.Tensor(np.random.randn(4, 16, 8, 8).astype(np.float32))
        out = bn(x)
        assert out.shape == (4, 16, 8, 8)

    def test_bn_eval_uses_running_stats(self):
        bn = lm.BatchNorm1d(4)
        x = lm.Tensor(np.random.randn(32, 4).astype(np.float32))
        bn.train()
        bn(x)   # update running stats
        bn.eval()
        out = bn(x)
        assert out.shape == (32, 4)


class TestPooling:
    def test_maxpool2d_shape(self):
        pool = lm.MaxPool2d(2)
        x = lm.Tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))
        out = pool(x)
        assert out.shape == (2, 4, 4, 4)

    def test_avgpool2d_shape(self):
        pool = lm.AvgPool2d(2)
        x = lm.Tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))
        out = pool(x)
        assert out.shape == (2, 4, 4, 4)

    def test_maxpool2d_gradient(self):
        pool = lm.MaxPool2d(2)
        x = lm.Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32), requires_grad=True)
        loss = pool(x).sum()
        loss.backward()
        assert x.grad is not None


class TestActivations:
    @pytest.mark.parametrize("act,x_val,expected", [
        (lm.ReLU(),     [-1.0, 1.0], [0.0, 1.0]),
        (lm.Sigmoid(),  [0.0],        [0.5]),
        (lm.Tanh(),     [0.0],        [0.0]),
    ])
    def test_activation_values(self, act, x_val, expected):
        x = lm.Tensor(x_val)
        out = act(x)
        np.testing.assert_allclose(out.data, expected, atol=1e-5)

    def test_leaky_relu_negative(self):
        x = lm.Tensor([-2.0])
        out = lm.LeakyReLU(0.1)(x)
        assert float(out.data[0]) == pytest.approx(-0.2, rel=1e-5)

    def test_softmax_sums_to_one(self):
        x = lm.Tensor(np.random.randn(4, 5).astype(np.float32))
        out = lm.Softmax(axis=-1)(x)
        np.testing.assert_allclose(out.data.sum(axis=-1), np.ones(4), atol=1e-5)


class TestDropout:
    def test_training_mode_zeros_some(self):
        drop = lm.Dropout(0.5)
        drop.train()
        x = lm.Tensor(np.ones((1000,), dtype=np.float32))
        out = drop(x)
        zero_frac = float((out.data == 0).mean())
        assert 0.3 < zero_frac < 0.7

    def test_eval_mode_passthrough(self):
        drop = lm.Dropout(0.9)
        drop.eval()
        x = lm.Tensor(np.ones((100,)))
        out = drop(x)
        np.testing.assert_allclose(out.data, x.data)


class TestSequential:
    def test_forward(self):
        model = lm.Sequential(
            lm.Linear(4, 8),
            lm.ReLU(),
            lm.Linear(8, 2),
        )
        x = lm.Tensor(np.random.randn(3, 4).astype(np.float32))
        out = model(x)
        assert out.shape == (3, 2)

    def test_parameter_count(self):
        model = lm.Sequential(lm.Linear(4, 8), lm.ReLU(), lm.Linear(8, 2))
        # (4*8 + 8) + (8*2 + 2) = 40 + 18 = 58
        assert model.num_parameters() == 58

    def test_train_eval_propagates(self):
        model = lm.Sequential(lm.Linear(2, 4), lm.Dropout(0.5))
        model.eval()
        assert not model.training
