"""
Example 04 — MNIST-like Digit Classification
=============================================
Simulates training on MNIST-style 28x28 images using MicroMLP.
Demonstrates: MicroMLP, DataLoader, EarlyStopping, ModelCheckpoint, Trainer.

To run on real MNIST, install `python-mnist` and replace the data generation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 55)
print("  LowMind Example 04 — MNIST-like Classification")
print("=" * 55)

# ── Synthetic 28×28 "digit" dataset ───────────────────────
np.random.seed(42)
N_TRAIN, N_VAL = 2000, 400
NUM_CLASSES = 10
IMG_DIM = 784  # 28*28

X_train = np.random.randn(N_TRAIN, IMG_DIM).astype(np.float32) * 0.3
y_train = np.random.randint(0, NUM_CLASSES, N_TRAIN)

# Add class-specific signal
for c in range(NUM_CLASSES):
    mask = y_train == c
    X_train[mask, c * 78:(c + 1) * 78] += 2.0  # discriminative feature

X_val = np.random.randn(N_VAL, IMG_DIM).astype(np.float32) * 0.3
y_val = np.random.randint(0, NUM_CLASSES, N_VAL)
for c in range(NUM_CLASSES):
    mask = y_val == c
    X_val[mask, c * 78:(c + 1) * 78] += 2.0

print(f"\n  Train: {N_TRAIN} samples  |  Val: {N_VAL} samples")
print(f"  Input: {IMG_DIM} features  |  Classes: {NUM_CLASSES}")

# ── Model ─────────────────────────────────────────────────
model = lm.MicroMLP(
    input_size=IMG_DIM,
    hidden_sizes=[256, 128],
    output_size=NUM_CLASSES,
    dropout=0.3,
)
print(f"\n  Model parameters: {model.num_parameters():,}")

# ── DataLoaders ───────────────────────────────────────────
train_loader = lm.DataLoader(
    lm.TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = lm.DataLoader(
    lm.TensorDataset(X_val, y_val), batch_size=128)

# ── Optimizer & Scheduler ─────────────────────────────────
optimizer = lm.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = lm.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

# ── Callbacks ─────────────────────────────────────────────
save_path = '/tmp/best_mnist_model.lmz'
callbacks = [
    lm.EarlyStopping(patience=10, verbose=True),
    lm.ModelCheckpoint(save_path, verbose=True),
    lm.LRSchedulerCallback(scheduler, monitor='val_loss'),
]

# ── Trainer ───────────────────────────────────────────────
trainer = lm.Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=lm.cross_entropy_loss,
    callbacks=callbacks,
    clip_grad=1.0,
    verbose=5,
)
print("\n  Starting training...")
history = trainer.fit(train_loader, val_loader, epochs=60)

# ── Evaluate ──────────────────────────────────────────────
val_loss, val_acc = trainer.evaluate(val_loader)
print(f"\n  Final val loss     : {val_loss:.4f}")
print(f"  Final val accuracy : {val_acc:.4f}")

# ── Save and reload ──────────────────────────────────────
model.save('/tmp/mnist_final.lmz')
model2 = lm.MicroMLP(IMG_DIM, [256, 128], NUM_CLASSES, dropout=0.3)
model2.load('/tmp/mnist_final.lmz')
print(f"\n  Model saved and reloaded successfully!")
print("\n  MNIST-like training complete!")
