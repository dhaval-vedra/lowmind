"""
Example 10 — Raspberry Pi System Monitor
==========================================
Monitor CPU temperature, RAM usage, and framework memory during training.
Demonstrates: SystemMonitor, memory_trace, configure_memory.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np
import time

print("=" * 60)
print("  LowMind Example 10 — System Monitor")
print("=" * 60)

# ── Configure memory for Raspberry Pi ─────────────────────
# Uncomment on Raspberry Pi:
# lm.configure_memory(max_mb=128, low_memory_mode=True)

monitor = lm.SystemMonitor()

print("\n  Initial system status:")
monitor.print_status()

# ── Measure memory usage during training ─────────────────
model = lm.Sequential(
    lm.Linear(128, 64),
    lm.ReLU(),
    lm.Linear(64, 32),
    lm.ReLU(),
    lm.Linear(32, 10),
)

np.random.seed(0)
X_np = np.random.randn(256, 128).astype(np.float32)
y_np = np.random.randint(0, 10, 256)
loader = lm.DataLoader(lm.TensorDataset(X_np, y_np), batch_size=32, shuffle=True)
optimizer = lm.Adam(model.parameters(), lr=1e-3)

print("\n  Training with memory tracing...\n")

with lm.memory_trace("Full Training (20 epochs)"):
    for epoch in range(20):
        model.train()
        with lm.memory_trace(f"Epoch {epoch+1:>2}"):
            epoch_loss = 0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                out = model(X_b)
                loss = lm.cross_entropy_loss(out, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
        stats = monitor.update()

print("\n  Post-training system status:")
monitor.print_status()

# ── Memory info ───────────────────────────────────────────
mem = lm.memory_manager.get_memory_info()
print(f"\n  Framework memory allocated : {mem['allocated_mb']:.2f} MB")
print(f"  Peak memory during session : {mem['peak_memory_mb']:.2f} MB")
print(f"  Total tensor count         : {mem['tensors_count']}")

# ── Optimize for inference ────────────────────────────────
print("\n  Optimizing memory for inference...")
lm.memory_manager.optimize_for_inference()
mem_after = lm.memory_manager.get_memory_info()
print(f"  Memory after optimization  : {mem_after['allocated_mb']:.2f} MB")

# ── Health score ──────────────────────────────────────────
score = monitor.health_score()
print(f"\n  System health score: {score:.1f} / 100")
if score >= 80:
    print("  Status: Excellent — running optimally")
elif score >= 60:
    print("  Status: Good — some constraints present")
elif score >= 40:
    print("  Status: Fair — consider reducing batch size")
else:
    print("  Status: Poor — reduce model size immediately")

# ── Memory history ────────────────────────────────────────
if monitor.history:
    alloc_trend = [h.get('allocated_mb', 0) for h in monitor.history]
    print(f"\n  Memory trend over {len(alloc_trend)} samples:")
    print(f"    Min: {min(alloc_trend):.2f} MB")
    print(f"    Max: {max(alloc_trend):.2f} MB")
    print(f"    Avg: {sum(alloc_trend)/len(alloc_trend):.2f} MB")

print("\n  System monitoring example complete!")
print("\n  Raspberry Pi tips:")
print("    - Use lm.configure_memory(max_mb=128) on Raspberry Pi")
print("    - Use batch_size=8-16 for 1GB Pi devices")
print("    - Use MicroMLP or MicroCNN for Pi-optimized architectures")
print("    - Call lm.memory_manager.optimize_for_inference() before deployment")
