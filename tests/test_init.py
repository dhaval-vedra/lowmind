"""
Tests for weight initialization utilities.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lowmind as lm


def make_tensor(shape):
    return lm.Tensor(np.zeros(shape, dtype=np.float32), requires_grad=True)


class TestXavier:
    def test_xavier_uniform_shape(self):
        t = make_tensor((64, 128))
        lm.xavier_uniform_(t)
        assert t.data.shape == (64, 128)

    def test_xavier_uniform_range(self):
        t = make_tensor((100, 200))
        lm.xavier_uniform_(t)
        a = np.sqrt(6.0 / (200 + 100))   # fan_in=200, fan_out=100
        assert t.data.min() >= -a - 1e-5
        assert t.data.max() <=  a + 1e-5

    def test_xavier_normal_std(self):
        t = make_tensor((256, 512))
        lm.xavier_normal_(t)
        expected_std = np.sqrt(2.0 / (512 + 256))
        np.testing.assert_allclose(t.data.std(), expected_std, rtol=0.15)

    def test_xavier_gain(self):
        t = make_tensor((64, 64))
        lm.xavier_uniform_(t, gain=np.sqrt(2.0))
        a_relu = np.sqrt(2.0) * np.sqrt(6.0 / (64 + 64))
        assert t.data.max() <= a_relu + 1e-5


class TestKaiming:
    def test_kaiming_uniform_shape(self):
        t = make_tensor((32, 64))
        lm.kaiming_uniform_(t)
        assert t.data.shape == (32, 64)

    def test_kaiming_uniform_range(self):
        t = make_tensor((512, 256))
        lm.kaiming_uniform_(t)
        fan_in = 256
        gain = np.sqrt(2.0)   # relu
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std
        assert t.data.min() >= -bound - 1e-4
        assert t.data.max() <=  bound + 1e-4

    def test_kaiming_normal_std(self):
        t = make_tensor((1024, 512))
        lm.kaiming_normal_(t)
        fan_in = 512
        expected_std = np.sqrt(2.0) / np.sqrt(fan_in)
        np.testing.assert_allclose(t.data.std(), expected_std, rtol=0.15)

    def test_kaiming_fan_out(self):
        t = make_tensor((128, 256))
        lm.kaiming_uniform_(t, mode='fan_out')
        fan_out = 128
        gain = np.sqrt(2.0)
        bound = np.sqrt(3.0) * gain / np.sqrt(fan_out)
        assert t.data.max() <= bound + 1e-4


class TestOrthogonal:
    def test_square_matrix_is_orthogonal(self):
        t = make_tensor((64, 64))
        lm.orthogonal_(t)
        Q = t.data
        I = Q @ Q.T
        np.testing.assert_allclose(I, np.eye(64), atol=1e-5)

    def test_tall_matrix(self):
        t = make_tensor((128, 64))
        lm.orthogonal_(t)
        Q = t.data
        I = Q.T @ Q
        np.testing.assert_allclose(I, np.eye(64), atol=1e-5)

    def test_gain(self):
        t = make_tensor((32, 32))
        lm.orthogonal_(t, gain=2.0)
        # Norm of each row should be ~2.0
        row_norms = np.linalg.norm(t.data, axis=1)
        np.testing.assert_allclose(row_norms, 2.0, atol=1e-5)

    def test_1d_raises(self):
        t = make_tensor((64,))
        with pytest.raises(ValueError):
            lm.orthogonal_(t)


class TestSimpleInits:
    def test_normal(self):
        t = make_tensor((1000, 100))
        lm.normal_(t, mean=1.0, std=0.5)
        np.testing.assert_allclose(t.data.mean(), 1.0, atol=0.05)
        np.testing.assert_allclose(t.data.std(), 0.5, atol=0.05)

    def test_uniform(self):
        t = make_tensor((1000, 100))
        lm.uniform_(t, a=-2.0, b=2.0)
        assert t.data.min() >= -2.0 - 1e-6
        assert t.data.max() <=  2.0 + 1e-6

    def test_constant(self):
        t = make_tensor((4, 4))
        lm.constant_(t, 3.14)
        assert np.all(t.data == 3.14)

    def test_zeros(self):
        t = make_tensor((8, 8))
        lm.ones_(t)
        lm.zeros_(t)
        assert np.all(t.data == 0.0)

    def test_ones(self):
        t = make_tensor((4, 4))
        lm.ones_(t)
        assert np.all(t.data == 1.0)

    def test_eye(self):
        t = make_tensor((5, 5))
        lm.eye_(t)
        np.testing.assert_allclose(t.data, np.eye(5, dtype=np.float32))

    def test_eye_non_square_raises(self):
        t = make_tensor((3,))
        with pytest.raises(ValueError):
            lm.eye_(t)


class TestInitModule:
    def test_init_module_all_layers(self):
        model = lm.Sequential(
            lm.Linear(32, 64), lm.ReLU(),
            lm.Linear(64, 32), lm.ReLU(),
            lm.Linear(32, 10),
        )
        # Verify it runs without error
        lm.init_module(model, weight_init='xavier_uniform', bias_init='zeros')
        # Biases should be zero
        for mod_name in [0, 2, 4]:
            mod = model._modules[str(mod_name)]
            np.testing.assert_allclose(mod.bias.data, np.zeros_like(mod.bias.data))

    def test_init_module_kaiming(self):
        model = lm.Sequential(lm.Linear(64, 128), lm.Linear(128, 10))
        lm.init_module(model, weight_init='kaiming_normal', bias_init='zeros')
        w = list(model.parameters())[0].data
        # Should have non-trivial variance
        assert w.std() > 0.01
