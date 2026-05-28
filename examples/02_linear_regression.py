"""
Example 02 — Linear Regression from Scratch
=============================================
Train y = 2x + 3 with gradient descent using LowMind.
Demonstrates: Tensor, mse_loss, SGD, custom training loop.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 55)
print("  LowMind Example 02 — Linear Regression")
print("=" * 55)

# ── Generate data: y = 2x + 3 + noise ────────────────────
np.random.seed(42)
N = 100
X_np = np.random.randn(N, 1).astype(np.float32)
y_np = 2.0 * X_np + 3.0 + 0.1 * np.random.randn(N, 1).astype(np.float32)

# ── Model: a single Linear(1 → 1) ─────────────────────────
model = lm.Linear(1, 1)
optimizer = lm.SGD(model.parameters(), lr=0.1)

print(f"\n  Initial weight: {model.weight.data[0][0]:.4f}")
print(f"  Initial bias  : {model.bias.data[0]:.4f}")
print(f"  Target   w=2.0, b=3.0\n")

# ── Training loop ──────────────────────────────────────────
losses = []
for epoch in range(200):
    X = lm.Tensor(X_np)
    y = lm.Tensor(y_np)

    optimizer.zero_grad()
    pred = model(X)
    loss = lm.mse_loss(pred, y)
    loss.backward()
    optimizer.step()

    losses.append(float(loss.item()))
    if (epoch + 1) % 40 == 0:
        print(f"  Epoch {epoch+1:>3} | loss={loss.item():.6f} | "
              f"w={model.weight.data[0][0]:.4f} | b={model.bias.data[0]:.4f}")

print(f"\n  Final weight: {model.weight.data[0][0]:.4f}  (target 2.0)")
print(f"  Final bias  : {model.bias.data[0]:.4f}  (target 3.0)")

# ── Prediction ─────────────────────────────────────────────
x_test = lm.Tensor([[1.0], [2.0], [5.0]])
preds = model(x_test)
print("\n  Predictions (y = 2x + 3):")
for xi, yi in zip([1, 2, 5], preds.data.flatten()):
    print(f"    x={xi} → pred={yi:.3f}  expected={2*xi+3}")

print("\n  Linear regression complete!")
