"""
Example 01 — Basic Tensors and Autograd
========================================
Learn how LowMind tensors work: creation, arithmetic,
automatic differentiation, and gradient computation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 55)
print("  LowMind Example 01 — Basic Tensors & Autograd")
print("=" * 55)

# ── 1. Tensor creation ─────────────────────────────────────
print("\n[1] Tensor Creation")
a = lm.Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = lm.Tensor([4.0, 5.0, 6.0], requires_grad=True)
print(f"  a = {a.data}")
print(f"  b = {b.data}")

# Factory functions
z = lm.zeros(3, 4)
o = lm.ones(2, 2)
r = lm.randn(3, 3)
print(f"  zeros(3,4) shape : {z.shape}")
print(f"  ones(2,2)  shape : {o.shape}")
print(f"  randn(3,3) shape : {r.shape}")

# ── 2. Arithmetic ─────────────────────────────────────────
print("\n[2] Arithmetic Operations")
c = a + b
d = a * b
e = a ** 2
print(f"  a + b = {c.data}")
print(f"  a * b = {d.data}")
print(f"  a**2  = {e.data}")

# ── 3. Autograd — single variable ─────────────────────────
print("\n[3] Automatic Differentiation")
x = lm.Tensor(3.0, requires_grad=True)
y = x * x + 2 * x + 1         # y = x² + 2x + 1  → dy/dx = 2x + 2
y.backward()
print(f"  y  = x^2 + 2x + 1  at x=3.0")
print(f"  y  = {y.item():.1f}   (expected 16.0)")
print(f"  dy/dx = {x.grad:.1f}   (expected 8.0)")

# ── 4. Autograd — vector ──────────────────────────────────
print("\n[4] Vector Autograd")
x = lm.Tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()           # loss = sum(x²)  → grad = 2x
loss.backward()
print(f"  x.grad = {x.grad}   (expected [2, 4, 6])")

# ── 5. Matrix multiplication ──────────────────────────────
print("\n[5] Matrix Multiplication")
W = lm.Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
x = lm.Tensor(np.random.randn(3).astype(np.float32), requires_grad=True)
out = W @ x
loss = out.sum()
loss.backward()
print(f"  W shape   : {W.shape}")
print(f"  out shape : {out.shape}")
print(f"  W.grad shape : {W.grad.shape}")

# ── 6. Activations ────────────────────────────────────────
print("\n[6] Activation Functions")
x = lm.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"  relu     : {x.relu().data}")
print(f"  sigmoid  : {np.round(x.sigmoid().data, 3)}")
print(f"  tanh     : {np.round(x.tanh().data, 3)}")
print(f"  leaky    : {x.leaky_relu(0.1).data}")
print(f"  softmax  : {np.round(x.softmax().data, 3)}")

print("\n  Done! Tensors and autograd working correctly.")
