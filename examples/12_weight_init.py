"""
LowMind Example 12 — Weight Initialization Comparison
======================================================
Compare Xavier, He, Orthogonal initialization on a deep network.
Shows how proper init leads to faster convergence.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lowmind as lm

print("=" * 58)
print("  LowMind Example 12 — Weight Initialization Comparison")
print("=" * 58)

np.random.seed(42)

# ── Dataset: synthetic 4-class classification ─────────────
N, D, C = 500, 64, 4
X = np.random.randn(N, D).astype(np.float32)
W_true = np.random.randn(D, C).astype(np.float32)
logits = X @ W_true
y_np = logits.argmax(axis=1)

X_t = lm.Tensor(X)
y_t = lm.Tensor(y_np.astype(np.int64))

def build_deep_model():
    return lm.Sequential(
        lm.Linear(64, 128), lm.ReLU(),
        lm.Linear(128, 128), lm.ReLU(),
        lm.Linear(128, 128), lm.ReLU(),
        lm.Linear(128, 64),  lm.ReLU(),
        lm.Linear(64, C),
    )

def train_with_init(init_name, n_epochs=60, lr=1e-3):
    model = build_deep_model()

    if init_name == 'default':
        pass   # layers already use He-like uniform by default
    elif init_name == 'xavier_uniform':
        lm.init_module(model, weight_init='xavier_uniform', bias_init='zeros')
    elif init_name == 'kaiming_normal':
        lm.init_module(model, weight_init='kaiming_normal', bias_init='zeros')
    elif init_name == 'orthogonal':
        lm.init_module(model, weight_init='orthogonal', bias_init='zeros')
    elif init_name == 'all_zeros':
        for p in model.parameters():
            if p.data.ndim > 1:
                p.data.fill(0.0)

    optimizer = lm.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = lm.cross_entropy_loss(out, y_t)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    final_acc = lm.accuracy(model(X_t).data.argmax(axis=1), y_np)
    return losses, final_acc

# ── Compare initializations ───────────────────────────────
inits = ['default', 'xavier_uniform', 'kaiming_normal', 'orthogonal', 'all_zeros']
print(f"\n  {'Init Method':<20} {'Init Loss':>10} {'Final Loss':>10} {'Final Acc':>10}")
print("  " + "-" * 54)

for init_name in inits:
    losses, acc = train_with_init(init_name)
    init_loss = losses[0]
    final_loss = losses[-1]
    warn = " ← WARNING: dead network" if acc < 0.5 else ""
    print(f"  {init_name:<20} {init_loss:>10.4f} {final_loss:>10.4f} {acc:>10.4f}{warn}")

# ── Demo individual init functions ────────────────────────
print("\n  Individual Init Function Demo:")
print("  " + "-" * 44)

demos = [
    ('xavier_uniform_', lambda t: lm.xavier_uniform_(t, gain=1.0)),
    ('kaiming_uniform_', lambda t: lm.kaiming_uniform_(t, nonlinearity='relu')),
    ('kaiming_normal_',  lambda t: lm.kaiming_normal_(t)),
    ('orthogonal_',      lambda t: lm.orthogonal_(t, gain=1.0)),
    ('normal_(0,0.01)',  lambda t: lm.normal_(t, mean=0.0, std=0.01)),
]

for name, init_fn in demos:
    t = lm.Tensor(np.zeros((64, 128), dtype=np.float32), requires_grad=True)
    init_fn(t)
    print(f"  {name:<22} mean={t.data.mean():>7.4f}  std={t.data.std():>7.4f}")

print("\n  Weight initialization comparison complete!")
print("  Conclusion: 'kaiming_normal' or 'xavier_uniform' recommended for ReLU nets.")
print("  'all_zeros' breaks gradient flow — always use proper init!")
