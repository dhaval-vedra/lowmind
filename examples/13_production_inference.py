"""
LowMind Example 13 — Production Inference Pipeline
===================================================
Demonstrates the full production workflow:
  1. no_grad context → fast inference, no memory waste
  2. Model profiler → FLOPs, params, throughput
  3. Batch inference with transforms
  4. Model export / reload check
  5. Parameter groups — different LR per layer
"""
import numpy as np
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lowmind as lm

print("=" * 58)
print("  LowMind Example 13 — Production Inference Pipeline")
print("=" * 58)

np.random.seed(0)

# ── Build and train a quick model ─────────────────────────
N, D, C = 400, 32, 5
X = np.random.randn(N, D).astype(np.float32)
y = (X @ np.random.randn(D, C).astype(np.float32)).argmax(axis=1)

model = lm.Sequential(
    lm.Linear(D, 64), lm.BatchNorm1d(64), lm.ReLU(), lm.Dropout(0.2),
    lm.Linear(64, 32), lm.ReLU(),
    lm.Linear(32, C),
)
lm.init_module(model, weight_init='kaiming_normal', bias_init='zeros')

# ── [1] Parameter Groups: freeze BN, different LR for head ─
print("\n[1] Parameter Groups — different LR per layer")
head_params = list(model[-1].parameters())
body_params  = [p for p in model.parameters() if not any(p is hp for hp in head_params)]

optimizer = lm.Adam([
    {'params': body_params, 'lr': 1e-4, 'weight_decay': 1e-5},
    {'params': head_params, 'lr': 1e-3},
])
print(f"    Body params : {sum(p.data.size for p in body_params):,}  (lr=1e-4)")
print(f"    Head params : {sum(p.data.size for p in head_params):,}  (lr=1e-3)")

# Quick train
X_t = lm.Tensor(X[:320])
y_t = lm.Tensor(y[:320].astype(np.int64))
for _ in range(50):
    optimizer.zero_grad()
    loss = lm.cross_entropy_loss(model(X_t), y_t)
    loss.backward()
    optimizer.step()
print(f"    Loss after 50 steps: {float(loss.item()):.4f}")

# ── [2] no_grad — inference mode ──────────────────────────
print("\n[2] no_grad — Zero-overhead inference")
X_test = lm.Tensor(X[320:])

model.eval()
# Zero grads from training before inference check
optimizer.zero_grad()

with lm.no_grad():
    logits = model(X_test)
    probs  = lm.Tensor(
        np.exp(logits.data - logits.data.max(axis=1, keepdims=True))
    )
    preds = logits.data.argmax(axis=1)

print(f"    Inference batch: {X_test.data.shape}")
print(f"    Logits computed: {logits.data.shape}")
print(f"    Max logit (first 5): {logits.data.max(axis=1)[:5].round(3)}")
print(f"    Predicted classes  : {preds[:10]}")

# Verify no new gradients were accumulated during inference
has_new_grad = any(p.grad is not None and p.grad.any()
                   for p in model.parameters())
print(f"    New gradients from inference: {has_new_grad}  (expected: False)")

# ── [3] Batch inference with transforms ───────────────────
print("\n[3] Batch inference with data transforms")
transform = lm.Compose([
    lm.Normalize(mean=X.mean(axis=0), std=X.std(axis=0) + 1e-8),
    lm.GaussianNoise(std=0.01, p=1.0),
])

X_transformed = transform(lm.Tensor(X[320:]))
print(f"    After Normalize → mean={X_transformed.data.mean():.4f}, std={X_transformed.data.std():.4f}")

BATCH = 16
all_preds = []
model.eval()
for i in range(0, len(X_transformed.data), BATCH):
    batch = lm.Tensor(X_transformed.data[i:i+BATCH])
    with lm.no_grad():
        out = model(batch)
    all_preds.extend(out.data.argmax(axis=1).tolist())

acc = lm.accuracy(np.array(all_preds), y[320:])
print(f"    Batched inference accuracy: {acc:.4f}")

# ── [4] Model Profiler ────────────────────────────────────
print("\n[4] Model Profiler")
profiler = lm.ModelProfiler(model)
report = profiler.profile(input_shape=(D,), batch_size=32, n_runs=30)
lm.ModelProfiler.print_report(report)

# ── [5] Save → reload → verify identical predictions ──────
print("\n[5] Save / Reload Consistency Check")
with tempfile.NamedTemporaryFile(suffix='.lmz', delete=False) as f:
    path = f.name

model.save(path)
model2 = lm.Sequential(
    lm.Linear(D, 64), lm.BatchNorm1d(64), lm.ReLU(), lm.Dropout(0.2),
    lm.Linear(64, 32), lm.ReLU(),
    lm.Linear(32, C),
)
model2.load(path)
model2.eval()

with lm.no_grad():
    preds2 = model2(X_test).data.argmax(axis=1)

match = np.all(preds == preds2)
print(f"    Saved  model → {os.path.getsize(path) / 1024:.1f} KB")
print(f"    Reload predictions match: {match}  (expected: True)")
os.unlink(path)

print("\n  Production inference pipeline complete!")
