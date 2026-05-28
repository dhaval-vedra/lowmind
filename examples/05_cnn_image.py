"""
Example 05 — CNN for Image Classification
==========================================
Train a MicroCNN on synthetic 32×32 colour images.
Demonstrates: Conv2d, BatchNorm2d, MaxPool2d, MicroCNN, DataLoader.

Works on Raspberry Pi with default memory settings.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 55)
print("  LowMind Example 05 — CNN Image Classification")
print("=" * 55)

# ── Synthetic image dataset ───────────────────────────────
np.random.seed(7)
N_TRAIN, N_VAL = 500, 100
NUM_CLASSES = 5
H, W, C_IN = 32, 32, 3

def make_dataset(n, num_classes):
    X = np.random.randn(n, C_IN, H, W).astype(np.float32) * 0.3
    y = np.random.randint(0, num_classes, n)
    for c in range(num_classes):
        mask = y == c
        X[mask, c % C_IN, :, :] += 1.5  # channel-specific signal
    return X, y

X_train, y_train = make_dataset(N_TRAIN, NUM_CLASSES)
X_val,   y_val   = make_dataset(N_VAL,   NUM_CLASSES)

print(f"\n  Images : {C_IN}×{H}×{W}  |  Classes: {NUM_CLASSES}")
print(f"  Train  : {N_TRAIN}   |  Val: {N_VAL}")

# ── Model ─────────────────────────────────────────────────
model = lm.MicroCNN(
    in_channels=C_IN,
    num_classes=NUM_CLASSES,
    input_size=H,
    dropout=0.25,
)
print(f"\n  Model: MicroCNN  |  Parameters: {model.num_parameters():,}")
print(f"  Architecture:\n{model}")

# ── DataLoaders ───────────────────────────────────────────
train_loader = lm.DataLoader(
    lm.TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = lm.DataLoader(
    lm.TensorDataset(X_val, y_val), batch_size=32)

# ── Training ──────────────────────────────────────────────
optimizer = lm.Adam(model.parameters(), lr=5e-3)
scheduler = lm.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

print("\n  Training...")
print("  Epoch | Train Loss | Val Acc")
print("  " + "-" * 35)

for epoch in range(1, 31):
    # Train
    model.train()
    total_loss = 0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        out = model(X_b)
        loss = lm.cross_entropy_loss(out, y_b)
        loss.backward()
        lm.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())

    scheduler.step()

    # Validate
    if epoch % 5 == 0:
        model.eval()
        X_v = lm.Tensor(X_val)
        val_out = model(X_v)
        acc = lm.accuracy(val_out, lm.Tensor(y_val))
        print(f"  {epoch:>5} | {total_loss/len(train_loader):.4f}     | {acc:.4f}")

# ── System monitor ────────────────────────────────────────
monitor = lm.SystemMonitor()
monitor.print_status()

print("\n  CNN training complete!")
