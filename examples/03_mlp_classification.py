"""
Example 03 — MLP Binary Classification
========================================
Classify XOR pattern using a 2-layer MLP.
Demonstrates: Sequential, cross_entropy_loss, Adam, accuracy metric.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 55)
print("  LowMind Example 03 — MLP Classification (XOR)")
print("=" * 55)

# ── XOR dataset ───────────────────────────────────────────
np.random.seed(0)
N = 400
X_np = np.random.randn(N, 2).astype(np.float32)
y_np = ((X_np[:, 0] * X_np[:, 1]) > 0).astype(np.int32)   # XOR pattern

X_train_np, X_val_np = X_np[:300], X_np[300:]
y_train_np, y_val_np = y_np[:300], y_np[300:]

print(f"\n  Train samples: {len(X_train_np)}  |  Val samples: {len(X_val_np)}")
print(f"  Classes: {np.unique(y_np)}")

# ── Model ─────────────────────────────────────────────────
model = lm.Sequential(
    lm.Linear(2, 32),
    lm.ReLU(),
    lm.Linear(32, 32),
    lm.ReLU(),
    lm.Linear(32, 2),
)
print(f"\n  Parameters: {model.num_parameters():,}")

optimizer = lm.Adam(model.parameters(), lr=5e-3)
scheduler = lm.StepLR(optimizer, step_size=30, gamma=0.5)

# ── DataLoader ────────────────────────────────────────────
ds_train = lm.TensorDataset(X_train_np, y_train_np)
ds_val   = lm.TensorDataset(X_val_np,   y_val_np)
train_loader = lm.DataLoader(ds_train, batch_size=32, shuffle=True)
val_loader   = lm.DataLoader(ds_val,   batch_size=64)

# ── Training loop ─────────────────────────────────────────
print("\n  Epoch | Train Loss | Val Acc")
print("  " + "-" * 35)

for epoch in range(1, 81):
    model.train()
    total_loss = 0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        out = model(X_b)
        loss = lm.cross_entropy_loss(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    scheduler.step()

    if epoch % 20 == 0:
        model.eval()
        X_val = lm.Tensor(X_val_np)
        y_val = lm.Tensor(y_val_np)
        val_out = model(X_val)
        acc = lm.accuracy(val_out, y_val)
        avg_loss = total_loss / len(train_loader)
        print(f"  {epoch:>5} | {avg_loss:.4f}     | {acc:.3f}")

# ── Final evaluation ──────────────────────────────────────
model.eval()
val_out = model(lm.Tensor(X_val_np))
print(f"\n  Final val accuracy : {lm.accuracy(val_out, lm.Tensor(y_val_np)):.4f}")
print(f"  F1 score           : {lm.f1_score(val_out, lm.Tensor(y_val_np), num_classes=2):.4f}")
print("\n  XOR classification complete!")
