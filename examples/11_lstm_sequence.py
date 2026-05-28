"""
LowMind Example 11 — LSTM for Sequence Prediction
=================================================
Train an LSTM to learn a sine wave.
Next-step prediction: given the last N points, predict the next one.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lowmind as lm

print("=" * 55)
print("  LowMind Example 11 — LSTM Sequence Prediction")
print("=" * 55)

np.random.seed(42)

# ── Generate sine wave data ──────────────────────────────
SEQ_LEN = 20    # look-back window
N_TRAIN = 800
N_VAL   = 200
HIDDEN  = 32
EPOCHS  = 40
LR      = 3e-3

t = np.linspace(0, 8 * np.pi, N_TRAIN + N_VAL + SEQ_LEN).astype(np.float32)
signal = np.sin(t) + 0.05 * np.random.randn(len(t)).astype(np.float32)

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(signal, SEQ_LEN)
# Shape: X=(N, SEQ_LEN), y=(N,)
X_train, y_train = X_all[:N_TRAIN], y_all[:N_TRAIN]
X_val,   y_val   = X_all[N_TRAIN:], y_all[N_TRAIN:]

print(f"\n  Sequence length: {SEQ_LEN}")
print(f"  Train samples : {N_TRAIN}")
print(f"  Val   samples : {N_VAL}")
print(f"  Hidden size   : {HIDDEN}")

# ── Build LSTM model ─────────────────────────────────────
class LSTMRegressor(lm.Module):
    """Single-layer LSTM + Linear head for next-step prediction."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = lm.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=0.2)
        self.head = lm.Linear(hidden_size, 1)
        lm.init_module(self.head, weight_init='xavier_normal')

    def forward(self, x):
        # x: (N, SEQ_LEN)  →  reshape to (SEQ_LEN, N, 1)
        batch = x.data.shape[0]
        x_seq = lm.Tensor(x.data.T.reshape(SEQ_LEN, batch, 1))
        output, (h_n, _) = self.lstm(x_seq)
        # Use the last hidden state: h_n[-1] = (N, hidden)
        last_h = lm.Tensor(h_n.data[-1])
        return self.head(last_h)  # (N, 1)

model = LSTMRegressor(input_size=1, hidden_size=HIDDEN)
optimizer = lm.Adam(model.parameters(), lr=LR)
scheduler = lm.CosineAnnealingLR(optimizer, T_max=EPOCHS)

params = sum(p.data.size for p in model.parameters())
print(f"  Parameters    : {params:,}\n")

# ── Training loop ─────────────────────────────────────────
BATCH = 64
print(f"  {'Epoch':>5} | {'Train MSE':>10} | {'Val MSE':>10}")
print("  " + "-" * 34)

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = np.random.permutation(N_TRAIN)
    train_loss = 0.0
    n_batches = 0

    for i in range(0, N_TRAIN, BATCH):
        bi = idx[i:i + BATCH]
        X_b = lm.Tensor(X_train[bi])
        y_b = lm.Tensor(y_train[bi].reshape(-1, 1))

        optimizer.zero_grad()
        pred = model(X_b)
        loss = lm.mse_loss(pred, y_b)
        loss.backward()
        lm.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.item())
        n_batches += 1

    scheduler.step()

    if epoch % 10 == 0:
        model.eval()
        X_v = lm.Tensor(X_val)
        y_v = lm.Tensor(y_val.reshape(-1, 1))
        with lm.no_grad():
            pred_v = model(X_v)
        val_mse = float(((pred_v.data - y_v.data) ** 2).mean())
        print(f"  {epoch:>5} | {train_loss/n_batches:>10.6f} | {val_mse:>10.6f}")

# ── Final evaluation ──────────────────────────────────────
model.eval()
with lm.no_grad():
    pred_all = model(lm.Tensor(X_val)).data.flatten()

true_vals = y_val[:10]
pred_vals = pred_all[:10]
print("\n  First 10 predictions vs actual:")
print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>10}")
for a, p in zip(true_vals, pred_vals):
    print(f"  {a:>10.4f}  {p:>10.4f}  {abs(a-p):>10.4f}")

final_mse = float(((pred_all - y_val) ** 2).mean())
print(f"\n  Final Val MSE : {final_mse:.6f}")
print(f"  Final Val RMSE: {final_mse**0.5:.6f}")
print("\n  LSTM sequence prediction complete!")
