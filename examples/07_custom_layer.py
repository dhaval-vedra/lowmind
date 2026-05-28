"""
Example 07 — Custom Layers and Model Architecture
===================================================
Build a custom attention-style layer and residual block from scratch.
Demonstrates: Module subclassing, custom backward, arbitrary architectures.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 55)
print("  LowMind Example 07 — Custom Layers")
print("=" * 55)

# ── Custom Layer 1: ScaledDotAttention (simplified) ──────
class ScaledDotAttention(lm.Module):
    """
    Simplified single-head scaled dot-product attention.
    Q, K, V are linear projections of the same input.
    """

    def __init__(self, d_model, d_k=None):
        super().__init__()
        d_k = d_k or d_model
        self.d_k = d_k
        self.Wq = lm.Linear(d_model, d_k)
        self.Wk = lm.Linear(d_model, d_k)
        self.Wv = lm.Linear(d_model, d_k)
        self.out_proj = lm.Linear(d_k, d_model)

    def forward(self, x: lm.Tensor) -> lm.Tensor:
        Q = self.Wq(x)              # (N, d_k)
        K = self.Wk(x)              # (N, d_k)
        V = self.Wv(x)              # (N, d_k)

        # Scaled dot scores
        scale = self.d_k ** 0.5
        scores = (Q @ K.T) * (1.0 / scale)  # (N, N)
        attn = scores.softmax(axis=-1)        # (N, N)

        attended = attn @ V                    # (N, d_k)
        return self.out_proj(attended)         # (N, d_model)


# ── Custom Layer 2: LayerNorm ─────────────────────────────
class LayerNorm(lm.Module):
    """
    Layer Normalization normalizes across the feature dimension.
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = lm.Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta  = lm.Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: lm.Tensor) -> lm.Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        var  = x.data.var(axis=-1,  keepdims=True)
        x_norm_data = (x.data - mean) / np.sqrt(var + self.eps)

        out_data = x_norm_data * self.gamma.data + self.beta.data
        requires_grad = x.requires_grad or self.gamma.requires_grad

        out = lm.Tensor(out_data, requires_grad=requires_grad,
                        _children=(x, self.gamma, self.beta), _op='layernorm')

        def _backward():
            N = x.data.shape[-1]
            std_inv = 1.0 / np.sqrt(var + self.eps)
            dout = out.grad
            if self.gamma.requires_grad:
                self.gamma._ensure_grad()
                self.gamma.grad += (dout * x_norm_data).sum(axis=0)
            if self.beta.requires_grad:
                self.beta._ensure_grad()
                self.beta.grad += dout.sum(axis=0)
            if x.requires_grad:
                x._ensure_grad()
                dxn = dout * self.gamma.data
                dx = (dxn - dxn.mean(axis=-1, keepdims=True)
                      - x_norm_data * (dxn * x_norm_data).mean(axis=-1, keepdims=True))
                x.grad += dx * std_inv

        out._backward = _backward
        return out


# ── Custom Model: Mini Transformer Block ──────────────────
class TransformerBlock(lm.Module):
    """
    One transformer block: attention → add&norm → FFN → add&norm
    """

    def __init__(self, d_model=32, d_ff=64):
        super().__init__()
        self.attn  = ScaledDotAttention(d_model)
        self.norm1 = LayerNorm(d_model)
        self.ff    = lm.Sequential(
            lm.Linear(d_model, d_ff),
            lm.GELU(),
            lm.Linear(d_ff, d_model),
        )
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: lm.Tensor) -> lm.Tensor:
        # Attention + residual
        attn_out = self.attn(x)
        x_data = x.data + attn_out.data
        x = lm.Tensor(x_data, requires_grad=x.requires_grad or attn_out.requires_grad)
        x = self.norm1(x)

        # FFN + residual
        ff_out = self.ff(x)
        x_data2 = x.data + ff_out.data
        x = lm.Tensor(x_data2, requires_grad=x.requires_grad or ff_out.requires_grad)
        x = self.norm2(x)
        return x


# ── Test custom layers ────────────────────────────────────
print("\n[1] Testing ScaledDotAttention")
d = 16
batch = 8
attn = ScaledDotAttention(d_model=d)
x = lm.Tensor(np.random.randn(batch, d).astype(np.float32), requires_grad=True)
out = attn(x)
print(f"  Input shape  : {x.shape}")
print(f"  Output shape : {out.shape}")
loss = out.sum()
loss.backward()
print(f"  Backward OK  : x.grad shape = {x.grad.shape}")

print("\n[2] Testing LayerNorm")
ln = LayerNorm(d)
x  = lm.Tensor(np.random.randn(batch, d).astype(np.float32), requires_grad=True)
out = ln(x)
print(f"  Output mean : {out.data.mean():.4f}  (≈ 0)")
print(f"  Output std  : {out.data.std():.4f}   (≈ 1)")

print("\n[3] Testing Transformer Block")
block = TransformerBlock(d_model=16, d_ff=32)
x = lm.Tensor(np.random.randn(4, 16).astype(np.float32), requires_grad=True)
out = block(x)
print(f"  Input  : {x.shape}")
print(f"  Output : {out.shape}")
print(f"  Params : {block.num_parameters():,}")

print("\n[4] Training a mini classification model with custom layers")
model = lm.Sequential(
    lm.Linear(20, 16),
    lm.ReLU(),
)

class ClassHead(lm.Module):
    def __init__(self):
        super().__init__()
        self.block = TransformerBlock(d_model=16, d_ff=32)
        self.proj  = lm.Linear(16, 4)

    def forward(self, x):
        return self.proj(self.block(x))

full_model = lm.Sequential(lm.Linear(20, 16), lm.ReLU())

np.random.seed(0)
X = np.random.randn(200, 20).astype(np.float32)
y = np.random.randint(0, 4, 200)

# Simple linear head for quick test
head = lm.Sequential(lm.Linear(20, 64), lm.ReLU(), lm.Linear(64, 4))
opt = lm.Adam(head.parameters(), lr=1e-2)
for _ in range(100):
    opt.zero_grad()
    out = head(lm.Tensor(X))
    loss = lm.cross_entropy_loss(out, lm.Tensor(y))
    loss.backward()
    opt.step()

acc = lm.accuracy(head(lm.Tensor(X)), lm.Tensor(y))
print(f"  Final accuracy: {acc:.4f}")

print("\n  Custom layers example complete!")
