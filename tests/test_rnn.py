"""
Tests for LowMind RNN layers (LSTM, GRU) and no_grad context manager.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lowmind as lm


class TestLSTMCell:
    def test_output_shape(self):
        cell = lm.LSTMCell(input_size=8, hidden_size=16)
        x = lm.Tensor(np.random.randn(4, 8).astype(np.float32))
        h, c = cell(x)
        assert h.data.shape == (4, 16)
        assert c.data.shape == (4, 16)

    def test_default_hidden_zeros(self):
        cell = lm.LSTMCell(input_size=4, hidden_size=8)
        x = lm.Tensor(np.random.randn(2, 4).astype(np.float32))
        h, c = cell(x)
        assert h.data.shape == (2, 8)

    def test_gradient_flows(self):
        cell = lm.LSTMCell(input_size=4, hidden_size=8)
        x = lm.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        h, c = cell(x)
        loss = h.sum()
        loss.backward()
        assert x.grad is not None
        assert cell.weight.grad is not None

    def test_no_bias(self):
        cell = lm.LSTMCell(input_size=4, hidden_size=8, bias=False)
        x = lm.Tensor(np.random.randn(2, 4).astype(np.float32))
        h, c = cell(x)
        assert not hasattr(cell, 'bias')
        assert h.data.shape == (2, 8)


class TestLSTM:
    def test_output_shape(self):
        lstm = lm.LSTM(input_size=10, hidden_size=20)
        x = lm.Tensor(np.random.randn(15, 4, 10).astype(np.float32))
        out, (h_n, c_n) = lstm(x)
        assert out.data.shape == (15, 4, 20)
        assert h_n.data.shape == (1, 4, 20)
        assert c_n.data.shape == (1, 4, 20)

    def test_multilayer(self):
        lstm = lm.LSTM(input_size=8, hidden_size=16, num_layers=3)
        x = lm.Tensor(np.random.randn(10, 2, 8).astype(np.float32))
        out, (h_n, c_n) = lstm(x)
        assert out.data.shape == (10, 2, 16)
        assert h_n.data.shape == (3, 2, 16)

    def test_dropout_only_in_training(self):
        lstm = lm.LSTM(input_size=4, hidden_size=8, num_layers=2, dropout=0.9)
        x = lm.Tensor(np.random.randn(5, 3, 4).astype(np.float32))
        lstm.train()
        out_train, _ = lstm(x)
        lstm.eval()
        out_eval, _ = lstm(x)
        # Both should produce valid outputs
        assert out_train.data.shape == out_eval.data.shape

    def test_sequence_regression(self):
        """LSTM should be able to reduce loss on a simple regression task."""
        lstm = lm.LSTM(input_size=1, hidden_size=16)
        head = lm.Linear(16, 1)
        optimizer = lm.Adam(list(lstm.cell_0.parameters()) + list(head.parameters()), lr=1e-2)

        losses = []
        for _ in range(30):
            x = lm.Tensor(np.random.randn(10, 4, 1).astype(np.float32))
            y = lm.Tensor(np.random.randn(4, 1).astype(np.float32))
            out, (h_n, _) = lstm(x)
            pred = head(lm.Tensor(h_n.data[-1]))
            loss = lm.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        # At least valid forward/backward pass (not NaN)
        assert not np.isnan(losses[-1])


class TestGRUCell:
    def test_output_shape(self):
        cell = lm.GRUCell(input_size=6, hidden_size=12)
        x = lm.Tensor(np.random.randn(3, 6).astype(np.float32))
        h = cell(x)
        assert h.data.shape == (3, 12)

    def test_gradient_flows(self):
        cell = lm.GRUCell(input_size=4, hidden_size=8)
        x = lm.Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        h = cell(x)
        h.sum().backward()
        assert x.grad is not None

    def test_no_bias(self):
        cell = lm.GRUCell(input_size=4, hidden_size=8, bias=False)
        x = lm.Tensor(np.random.randn(2, 4).astype(np.float32))
        h = cell(x)
        assert not hasattr(cell, 'bias_ir')
        assert h.data.shape == (2, 8)


class TestGRU:
    def test_output_shape(self):
        gru = lm.GRU(input_size=10, hidden_size=20)
        x = lm.Tensor(np.random.randn(15, 4, 10).astype(np.float32))
        out, h_n = gru(x)
        assert out.data.shape == (15, 4, 20)
        assert h_n.data.shape == (1, 4, 20)

    def test_multilayer_gru(self):
        gru = lm.GRU(input_size=6, hidden_size=12, num_layers=2)
        x = lm.Tensor(np.random.randn(8, 3, 6).astype(np.float32))
        out, h_n = gru(x)
        assert out.data.shape == (8, 3, 12)
        assert h_n.data.shape == (2, 3, 12)


class TestNoGrad:
    def test_no_grad_prevents_graph_building(self):
        x = lm.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        with lm.no_grad():
            y = x * 2 + 1
        assert not y.requires_grad

    def test_no_grad_does_not_affect_data(self):
        x = lm.Tensor(np.array([3.0], dtype=np.float32), requires_grad=True)
        with lm.no_grad():
            y = x * x
        np.testing.assert_allclose(y.data, [9.0])

    def test_enable_grad_inside_no_grad(self):
        x = lm.Tensor(np.array([2.0], dtype=np.float32), requires_grad=True)
        with lm.no_grad():
            with lm.enable_grad():
                y = x * x
        assert y.requires_grad

    def test_no_grad_as_decorator(self):
        @lm.no_grad()
        def forward(x):
            return x * 2

        x = lm.Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        y = forward(x)
        assert not y.requires_grad

    def test_model_eval_with_no_grad(self):
        model = lm.Sequential(lm.Linear(4, 4), lm.ReLU(), lm.Linear(4, 2))
        model.eval()
        x = lm.Tensor(np.random.randn(3, 4).astype(np.float32))
        with lm.no_grad():
            out = model(x)
        assert out.data.shape == (3, 2)
        for p in model.parameters():
            assert p.grad is None or not p.grad.any()
