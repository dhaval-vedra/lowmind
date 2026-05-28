"""
Example 06 — Optimizers Comparison
====================================
Compare SGD, Adam, RMSprop, AdaGrad on the same task.
Demonstrates: all optimizers, how to benchmark them.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np
import time

print("=" * 60)
print("  LowMind Example 06 — Optimizer Comparison")
print("=" * 60)

# ── Reproducible dataset ──────────────────────────────────
np.random.seed(123)
N = 600
X_np = np.random.randn(N, 20).astype(np.float32)
W_true = np.random.randn(20, 5).astype(np.float32)
y_np = (X_np @ W_true).argmax(axis=1)  # 5-class labels

X_t = lm.Tensor(X_np)
y_t = lm.Tensor(y_np)

# ── Optimizer configurations ──────────────────────────────
configs = [
    ("SGD       ", lambda p: lm.SGD(p, lr=0.05, momentum=0.9)),
    ("Adam      ", lambda p: lm.Adam(p, lr=1e-3)),
    ("AdamW     ", lambda p: lm.AdamW(p, lr=1e-3, weight_decay=1e-4)),
    ("RMSprop   ", lambda p: lm.RMSprop(p, lr=1e-3, alpha=0.9)),
    ("AdaGrad   ", lambda p: lm.AdaGrad(p, lr=0.05)),
]

def make_model():
    return lm.Sequential(
        lm.Linear(20, 64),
        lm.ReLU(),
        lm.Linear(64, 32),
        lm.ReLU(),
        lm.Linear(32, 5),
    )

print(f"\n  {'Optimizer':<12} {'Final Loss':>12} {'Final Acc':>10} {'Time (s)':>10}")
print("  " + "-" * 48)

EPOCHS = 100
BATCH  = 64

results = {}
for name, opt_fn in configs:
    model = make_model()
    opt = opt_fn(model.parameters())
    loader = lm.DataLoader(
        lm.TensorDataset(X_np, y_np), batch_size=BATCH, shuffle=True)

    t0 = time.time()
    for _ in range(EPOCHS):
        for X_b, y_b in loader:
            opt.zero_grad()
            out = model(X_b)
            loss = lm.cross_entropy_loss(out, y_b)
            loss.backward()
            opt.step()

    elapsed = time.time() - t0
    model.eval()
    final_out = model(X_t)
    final_loss = float(lm.cross_entropy_loss(final_out, y_t).item())
    final_acc  = lm.accuracy(final_out, y_t)
    results[name] = (final_loss, final_acc, elapsed)
    print(f"  {name} {final_loss:>12.4f} {final_acc:>10.4f} {elapsed:>10.2f}s")

best = min(results, key=lambda k: results[k][0])
print(f"\n  Best optimizer: {best.strip()} "
      f"(loss={results[best][0]:.4f}, acc={results[best][1]:.4f})")
print("\n  Optimizer comparison complete!")
