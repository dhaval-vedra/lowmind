"""
LowMind Recurrent Neural Network Layers

LSTMCell   — single-step LSTM gate computations
LSTM       — multi-step LSTM over a sequence
GRUCell    — single-step GRU
GRU        — multi-step GRU over a sequence
"""
import numpy as np
from ..core.tensor import Tensor
from ..core.module import Module


class LSTMCell(Module):
    """
    Single time-step LSTM cell.

    Gates: i (input), f (forget), g (cell), o (output)

    Args:
        input_size:  Number of input features.
        hidden_size: Number of hidden units.
        bias:        Add bias (default True).

    Inputs:
        x:      (N, input_size)
        h_prev: (N, hidden_size) — previous hidden state (default zeros)
        c_prev: (N, hidden_size) — previous cell state (default zeros)

    Returns:
        (h_next, c_next) both shape (N, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weight matrix for all 4 gates: (input_size + hidden_size) → 4*hidden_size
        k = 1.0 / np.sqrt(hidden_size)
        w_data = np.random.uniform(-k, k, (input_size + hidden_size, 4 * hidden_size)).astype(np.float32)
        self.weight = Tensor(w_data, requires_grad=True)

        if bias:
            b_data = np.zeros(4 * hidden_size, dtype=np.float32)
            self.bias = Tensor(b_data, requires_grad=True)

    def forward(self, x: Tensor, h_prev: Tensor = None, c_prev: Tensor = None):
        N = x.data.shape[0]
        H = self.hidden_size

        if h_prev is None:
            h_prev = Tensor(np.zeros((N, H), dtype=np.float32))
        if c_prev is None:
            c_prev = Tensor(np.zeros((N, H), dtype=np.float32))

        # Concatenate input and hidden: (N, input_size + hidden_size)
        xh_data = np.concatenate([x.data, h_prev.data], axis=1)
        gates_data = xh_data @ self.weight.data
        if hasattr(self, 'bias'):
            gates_data = gates_data + self.bias.data

        # Split into 4 gates
        i = 1.0 / (1.0 + np.exp(-gates_data[:, :H]))          # input gate (sigmoid)
        f = 1.0 / (1.0 + np.exp(-gates_data[:, H:2*H]))       # forget gate (sigmoid)
        g = np.tanh(gates_data[:, 2*H:3*H])                    # cell gate (tanh)
        o = 1.0 / (1.0 + np.exp(-gates_data[:, 3*H:]))        # output gate (sigmoid)

        c_next_data = f * c_prev.data + i * g
        h_next_data = o * np.tanh(c_next_data)

        requires_grad = (x.requires_grad or h_prev.requires_grad or
                         c_prev.requires_grad or self.weight.requires_grad)

        c_next = Tensor(c_next_data, requires_grad=requires_grad,
                        _children=(x, h_prev, c_prev, self.weight), _op='lstm_c')
        h_next = Tensor(h_next_data, requires_grad=requires_grad,
                        _children=(c_next,), _op='lstm_h')

        def _backward_c():
            if c_next.grad is None:
                return
            dc = c_next.grad

            # di = dc * g * i * (1-i)
            di = dc * g * i * (1 - i)
            # df = dc * c_prev * f * (1-f)
            df = dc * c_prev.data * f * (1 - f)
            # dg = dc * i * (1 - g^2)
            dg = dc * i * (1 - g ** 2)
            # do = 0 (o only affects h, not c) → will be accumulated from h backward
            do_ = np.zeros_like(dc)

            dgates = np.concatenate([di, df, dg, do_], axis=1)

            if self.weight.requires_grad:
                self.weight._ensure_grad()
                self.weight.grad += xh_data.T @ dgates
            if hasattr(self, 'bias') and self.bias.requires_grad:
                self.bias._ensure_grad()
                self.bias.grad += dgates.sum(axis=0)

            dxh = dgates @ self.weight.data.T
            if x.requires_grad:
                x._ensure_grad()
                x.grad += dxh[:, :self.input_size]
            if h_prev.requires_grad:
                h_prev._ensure_grad()
                h_prev.grad += dxh[:, self.input_size:]
            if c_prev.requires_grad:
                c_prev._ensure_grad()
                c_prev.grad += dc * f

        def _backward_h():
            if h_next.grad is None:
                return
            dh = h_next.grad
            tanh_c = np.tanh(c_next_data)

            # do = dh * tanh(c) * o * (1-o)
            do_ = dh * tanh_c * o * (1 - o)
            # dc from h: dh * o * (1 - tanh²(c))
            dc_from_h = dh * o * (1 - tanh_c ** 2)

            if c_next.requires_grad:
                c_next._ensure_grad()
                c_next.grad += dc_from_h

            # Accumulate do into gates
            dgates = np.zeros((dh.shape[0], 4 * self.hidden_size), dtype=np.float32)
            dgates[:, 3*self.hidden_size:] = do_

            if self.weight.requires_grad:
                self.weight._ensure_grad()
                self.weight.grad += xh_data.T @ dgates
            if hasattr(self, 'bias') and self.bias.requires_grad:
                self.bias._ensure_grad()
                self.bias.grad += dgates.sum(axis=0)

            dxh = dgates @ self.weight.data.T
            if x.requires_grad:
                x._ensure_grad()
                x.grad += dxh[:, :self.input_size]
            if h_prev.requires_grad:
                h_prev._ensure_grad()
                h_prev.grad += dxh[:, self.input_size:]

        c_next._backward = _backward_c
        h_next._backward = _backward_h
        return h_next, c_next

    def __repr__(self):
        return f"LSTMCell(input={self.input_size}, hidden={self.hidden_size})"


class LSTM(Module):
    """
    Multi-layer LSTM over a full sequence.

    Args:
        input_size:   Number of input features.
        hidden_size:  Number of hidden units.
        num_layers:   Number of stacked LSTM layers (default 1).
        bias:         Add bias (default True).
        dropout:      Dropout between layers (default 0.0, only for num_layers>1).
        bidirectional: Not yet supported — raises if True.

    Inputs:
        x:          (seq_len, N, input_size) — sequence-first format
        h0, c0:     (num_layers, N, hidden_size) — initial states (zeros if None)

    Returns:
        (output, (h_n, c_n)) where
          output: (seq_len, N, hidden_size)
          h_n:    (num_layers, N, hidden_size)
          c_n:    (num_layers, N, hidden_size)

    Example::

        lstm = lm.LSTM(input_size=10, hidden_size=32, num_layers=2, dropout=0.3)
        x = lm.Tensor(np.random.randn(20, 4, 10).astype(np.float32))  # (T=20, N=4, F=10)
        output, (h_n, c_n) = lstm(x)
        # output: (20, 4, 32)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = LSTMCell(in_size, hidden_size, bias)
            setattr(self, f'cell_{layer}', cell)
            self._modules[f'cell_{layer}'] = cell

    def _get_cell(self, layer):
        return getattr(self, f'cell_{layer}')

    def forward(self, x: Tensor, hx=None):
        T, N, _ = x.data.shape
        H = self.hidden_size
        L = self.num_layers

        if hx is None:
            h = [Tensor(np.zeros((N, H), dtype=np.float32)) for _ in range(L)]
            c = [Tensor(np.zeros((N, H), dtype=np.float32)) for _ in range(L)]
        else:
            h0, c0 = hx
            h = [Tensor(h0.data[i]) for i in range(L)]
            c = [Tensor(c0.data[i]) for i in range(L)]

        outputs = []
        for t in range(T):
            x_t = x[t]   # (N, input_size)
            for layer in range(L):
                cell = self._get_cell(layer)
                h[layer], c[layer] = cell(x_t, h[layer], c[layer])
                x_t = h[layer]

                # Dropout between layers (not on last layer)
                if self.dropout_p > 0 and layer < L - 1 and self.training:
                    mask = (np.random.rand(*x_t.data.shape) >= self.dropout_p).astype(np.float32)
                    mask /= (1 - self.dropout_p)
                    x_t = x_t * Tensor(mask)

            outputs.append(h[-1])   # (N, H)

        # Stack outputs: (T, N, H)
        output_data = np.stack([o.data for o in outputs], axis=0)
        output = Tensor(output_data, requires_grad=x.requires_grad)

        h_n_data = np.stack([hi.data for hi in h], axis=0)   # (L, N, H)
        c_n_data = np.stack([ci.data for ci in c], axis=0)   # (L, N, H)

        h_n = Tensor(h_n_data)
        c_n = Tensor(c_n_data)

        return output, (h_n, c_n)

    def __repr__(self):
        return (f"LSTM(input={self.input_size}, hidden={self.hidden_size}, "
                f"layers={self.num_layers}, dropout={self.dropout_p})")


class GRUCell(Module):
    """
    Single time-step GRU cell.

    Gates: z (update), r (reset), h~ (candidate hidden)

    Args:
        input_size:  Number of input features.
        hidden_size: Number of hidden units.
        bias:        Add bias (default True).
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 1.0 / np.sqrt(hidden_size)
        # Weights for [z, r] gates: (input+hidden, 2*hidden)
        self.weight_ir = Tensor(
            np.random.uniform(-k, k, (input_size + hidden_size, 2 * hidden_size)).astype(np.float32),
            requires_grad=True)
        # Weights for candidate: (input+hidden, hidden)
        self.weight_h = Tensor(
            np.random.uniform(-k, k, (input_size + hidden_size, hidden_size)).astype(np.float32),
            requires_grad=True)

        if bias:
            self.bias_ir = Tensor(np.zeros(2 * hidden_size, dtype=np.float32), requires_grad=True)
            self.bias_h  = Tensor(np.zeros(hidden_size, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor, h_prev: Tensor = None):
        N, H = x.data.shape[0], self.hidden_size
        if h_prev is None:
            h_prev = Tensor(np.zeros((N, H), dtype=np.float32))

        xh = np.concatenate([x.data, h_prev.data], axis=1)
        gates = xh @ self.weight_ir.data
        if hasattr(self, 'bias_ir'):
            gates = gates + self.bias_ir.data

        z = 1.0 / (1.0 + np.exp(-gates[:, :H]))      # update gate
        r = 1.0 / (1.0 + np.exp(-gates[:, H:]))      # reset gate

        xrh = np.concatenate([x.data, r * h_prev.data], axis=1)
        h_cand = np.tanh(xrh @ self.weight_h.data + (self.bias_h.data if hasattr(self, 'bias_h') else 0))

        h_next_data = (1 - z) * h_prev.data + z * h_cand

        requires_grad = (x.requires_grad or h_prev.requires_grad or
                         self.weight_ir.requires_grad)
        h_next = Tensor(h_next_data, requires_grad=requires_grad,
                        _children=(x, h_prev, self.weight_ir), _op='gru')

        def _backward():
            if h_next.grad is None:
                return
            dh = h_next.grad

            # Gradient through (1-z)*h_prev + z*h_cand
            dz = dh * (h_cand - h_prev.data) * z * (1 - z)
            dh_cand = dh * z * (1 - h_cand ** 2)
            dh_prev_direct = dh * (1 - z)

            # Gradient through candidate
            dxrh = dh_cand @ self.weight_h.data.T
            if self.weight_h.requires_grad:
                self.weight_h._ensure_grad()
                self.weight_h.grad += xrh.T @ dh_cand
            if hasattr(self, 'bias_h') and self.bias_h.requires_grad:
                self.bias_h._ensure_grad()
                self.bias_h.grad += dh_cand.sum(axis=0)

            dx_from_cand = dxrh[:, :self.input_size]
            dr_h_prev = dxrh[:, self.input_size:]
            dr = dr_h_prev * h_prev.data * r * (1 - r)
            dh_prev_from_cand = dr_h_prev * r

            # Gradient through gates [z, r]
            dgates = np.concatenate([dz, dr], axis=1)
            if self.weight_ir.requires_grad:
                self.weight_ir._ensure_grad()
                self.weight_ir.grad += xh.T @ dgates
            if hasattr(self, 'bias_ir') and self.bias_ir.requires_grad:
                self.bias_ir._ensure_grad()
                self.bias_ir.grad += dgates.sum(axis=0)

            dxh = dgates @ self.weight_ir.data.T
            dx_from_gates = dxh[:, :self.input_size]
            dh_prev_from_gates = dxh[:, self.input_size:]

            if x.requires_grad:
                x._ensure_grad()
                x.grad += dx_from_gates + dx_from_cand
            if h_prev.requires_grad:
                h_prev._ensure_grad()
                h_prev.grad += dh_prev_direct + dh_prev_from_cand + dh_prev_from_gates

        h_next._backward = _backward
        return h_next

    def __repr__(self):
        return f"GRUCell(input={self.input_size}, hidden={self.hidden_size})"


class GRU(Module):
    """
    Multi-step GRU over a full sequence.

    Args:
        input_size:  Number of input features.
        hidden_size: Number of hidden units.
        num_layers:  Stacked layers (default 1).
        bias:        Add bias (default True).
        dropout:     Dropout between layers (default 0.0).

    Inputs:
        x:  (seq_len, N, input_size)
        h0: (num_layers, N, hidden_size) or None

    Returns:
        (output, h_n) where output: (seq_len, N, hidden_size)

    Example::

        gru = lm.GRU(input_size=10, hidden_size=32)
        x = lm.Tensor(np.random.randn(20, 4, 10).astype(np.float32))
        output, h_n = gru(x)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = GRUCell(in_size, hidden_size, bias)
            setattr(self, f'cell_{layer}', cell)
            self._modules[f'cell_{layer}'] = cell

    def forward(self, x: Tensor, h0: Tensor = None):
        T, N, _ = x.data.shape
        H = self.hidden_size
        L = self.num_layers

        if h0 is None:
            h = [Tensor(np.zeros((N, H), dtype=np.float32)) for _ in range(L)]
        else:
            h = [Tensor(h0.data[i]) for i in range(L)]

        outputs = []
        for t in range(T):
            x_t = x[t]
            for layer in range(L):
                cell = getattr(self, f'cell_{layer}')
                h[layer] = cell(x_t, h[layer])
                x_t = h[layer]
                if self.dropout_p > 0 and layer < L - 1 and self.training:
                    mask = (np.random.rand(*x_t.data.shape) >= self.dropout_p).astype(np.float32)
                    mask /= (1 - self.dropout_p)
                    x_t = x_t * Tensor(mask)
            outputs.append(h[-1])

        output_data = np.stack([o.data for o in outputs], axis=0)
        output = Tensor(output_data, requires_grad=x.requires_grad)
        h_n_data = np.stack([hi.data for hi in h], axis=0)
        return output, Tensor(h_n_data)

    def __repr__(self):
        return (f"GRU(input={self.input_size}, hidden={self.hidden_size}, "
                f"layers={self.num_layers})")
