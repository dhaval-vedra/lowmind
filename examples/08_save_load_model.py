"""
Example 08 — Save, Load and Export Models
==========================================
Save model weights, reload them, and verify predictions match.
Demonstrates: model.save(), model.load(), state_dict.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 55)
print("  LowMind Example 08 — Save & Load Models")
print("=" * 55)

# ── Train a small model ───────────────────────────────────
np.random.seed(42)
X_np = np.random.randn(400, 16).astype(np.float32)
y_np = np.random.randint(0, 4, 400)

model = lm.Sequential(
    lm.Linear(16, 64),
    lm.ReLU(),
    lm.Dropout(0.2),
    lm.Linear(64, 32),
    lm.ReLU(),
    lm.Linear(32, 4),
)

opt = lm.Adam(model.parameters(), lr=3e-3)
loader = lm.DataLoader(lm.TensorDataset(X_np, y_np), batch_size=32, shuffle=True)

print("\n  Training model...")
for epoch in range(50):
    for X_b, y_b in loader:
        opt.zero_grad()
        loss = lm.cross_entropy_loss(model(X_b), y_b)
        loss.backward()
        opt.step()

model.eval()
X_all = lm.Tensor(X_np)
before_preds = model(X_all).data.argmax(axis=1).copy()
acc_before = lm.accuracy(model(X_all), lm.Tensor(y_np))
print(f"  Accuracy before save: {acc_before:.4f}")

# ── Save (compressed) ─────────────────────────────────────
path_gz = '/tmp/my_model.lmz'
model.save(path_gz, compress=True)
size = os.path.getsize(path_gz) / 1024
print(f"\n  Saved compressed model: {path_gz} ({size:.1f} KB)")

# ── Save (plain) ──────────────────────────────────────────
path_plain = '/tmp/my_model.lm'
model.save(path_plain, compress=False)
size_plain = os.path.getsize(path_plain) / 1024
print(f"  Saved plain model    : {path_plain} ({size_plain:.1f} KB)")

# ── Load into a fresh model ───────────────────────────────
model2 = lm.Sequential(
    lm.Linear(16, 64),
    lm.ReLU(),
    lm.Dropout(0.2),
    lm.Linear(64, 32),
    lm.ReLU(),
    lm.Linear(32, 4),
)
model2.eval()
model2.load(path_gz)

after_preds = model2(X_all).data.argmax(axis=1)
acc_after = lm.accuracy(model2(X_all), lm.Tensor(y_np))
print(f"\n  Accuracy after reload: {acc_after:.4f}")
print(f"  Predictions match    : {np.all(before_preds == after_preds)}")

# ── state_dict inspection ─────────────────────────────────
print("\n  State dict keys:")
sd = model.state_dict()
for key, arr in sd.items():
    print(f"    {key:25s}  shape={arr.shape}  dtype={arr.dtype}")

# ── Partial load / transfer learning ─────────────────────
print("\n  Transfer learning demo: freeze first layer")
model3 = lm.Sequential(
    lm.Linear(16, 64),
    lm.ReLU(),
    lm.Dropout(0.2),
    lm.Linear(64, 32),
    lm.ReLU(),
    lm.Linear(32, 8),   # different output size (8 classes now)
)
# Load only the shared layers
shared_sd = {k: v for k, v in sd.items() if not k.startswith('5')}
model3.load_state_dict(shared_sd, strict=False)
print("  Loaded shared weights into model with 8 output classes")

# Freeze first linear layer
first_param = list(model3.parameters())[0]
first_param.requires_grad = False
trainable = sum(p.data.size for p in model3.parameters() if p.requires_grad)
print(f"  Trainable params after freeze: {trainable:,}")

print("\n  Save / Load example complete!")
