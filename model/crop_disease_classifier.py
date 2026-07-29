"""
Crop Disease Leaf Classifier using LowMind MicroCNN
Optimized for Raspberry Pi deployment on field edge devices.
"""
import lowmind as lm
import numpy as np


class LeafDiseaseCNN(lm.Module):
    """
    轻量级卷积神经网络 (CNN) for leaf crop disease image classification.
    Input: (N, 3, 32, 32)
    Output: (N, 3) -> ["HEALTHY", "BACTERIAL_ROT", "RUST_FUNGI"]
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = lm.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = lm.BatchNorm2d(8)
        self.pool1 = lm.MaxPool2d(2, 2)

        self.conv2 = lm.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = lm.BatchNorm2d(16)
        self.pool2 = lm.MaxPool2d(2, 2)

        self.flatten = lm.Flatten(start_dim=1)
        self.fc1 = lm.Linear(16 * 8 * 8, 32)
        self.fc2 = lm.Linear(32, num_classes)
        self.dropout = lm.Dropout(0.2)

    def forward(self, x):
        x = self.pool1(self.bn1(self.conv1(x)).relu())
        x = self.pool2(self.bn2(self.conv2(x)).relu())
        x = self.flatten(x)
        x = self.dropout(self.fc1(x).relu())
        x = self.fc2(x)
        return x


def train_disease_classifier():
    print("=" * 60)
    print("🌱 Training Leaf Crop Disease CNN Classifier")
    print("=" * 60)

    # Generate mock leaf disease image dataset (N, C, H, W)
    # Class 0: Healthy, Class 1: Bacterial Rot, Class 2: Rust Fungi
    np.random.seed(42)
    num_samples = 150
    X = np.random.randn(num_samples, 3, 32, 32).astype(np.float32)
    # Add fake distinctive features based on class to allow model to learn
    y = np.random.randint(0, 3, num_samples)
    for i in range(num_samples):
        if y[i] == 0:
            X[i, 1] += 1.5  # More green (Healthy)
        elif y[i] == 1:
            X[i, 0] += 1.5  # More red/yellow (Rot)
        else:
            X[i, 2] += 1.5  # More blue/brown (Rust)

    # Train / Val Split
    X_train, X_val = X[:120], X[120:]
    y_train, y_val = y[:120], y[120:]

    train_loader = lm.DataLoader(lm.TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = lm.DataLoader(lm.TensorDataset(X_val, y_val), batch_size=16)

    # Initialize Model, Optimizer and Trainer
    model = LeafDiseaseCNN(num_classes=3)
    optimizer = lm.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    trainer = lm.Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lm.cross_entropy_loss,
        verbose=1
    )

    # Train for 5 epochs
    print("Starting training...")
    trainer.fit(train_loader, val_loader, epochs=5)

    # Evaluate model
    loss, acc = trainer.evaluate(val_loader)
    print(f"\n✅ Validation Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}")

    # --- Advanced Optimization / Quantization & Export ---
    print("\n📦 Production-level Edge Optimization:")

    # 1. Prune 30% of lowest magnitude weights to optimize footprint
    pruner = lm.Pruner(model)
    pruner.prune_model(sparsity_ratio=0.3)
    print(f"   ✂️ Pruned model weight sparsity: {pruner.calculate_sparsity():.2f}%")

    # 2. Simulate INT8 Quantization
    model.quantize()
    print("   🎯 Weights quantized to INT8 simulation.")

    # 3. Export to standard ONNX for edge deployment (e.g. ONNX Runtime on Pi Zero)
    dummy_in = np.random.randn(1, 3, 32, 32).astype(np.float32)
    lm.export_to_onnx(model, dummy_in, "model/crop_disease_model.onnx")

    print("=" * 60)


if __name__ == "__main__":
    train_disease_classifier()
