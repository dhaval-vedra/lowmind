"""
Heart Disease Predictor (Binary Classification) using LowMind MLP and QAT
"""
import lowmind as lm
import numpy as np


class HeartDiseaseMLP(lm.Module):
    """
    Multi-Layer Perceptron for Clinical Tabular Patient Data.
    Input features: 8 (Age, Cholesterol, Blood Pressure, Heart Rate, etc.)
    Output: 1 (Heart Risk / No Risk)
    """
    def __init__(self, in_dim=8):
        super().__init__()
        self.fc1 = lm.Linear(in_dim, 16)
        self.fc2 = lm.Linear(16, 8)
        self.fc3 = lm.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x


def train_heart_predictor():
    print("=" * 60)
    print("❤️ Training Tabular Clinical Heart Disease Predictor")
    print("=" * 60)

    # Generate mock heart disease tabular clinical dataset
    np.random.seed(42)
    num_samples = 200
    X = np.random.randn(num_samples, 8).astype(np.float32)
    # Target label: 0 (no risk), 1 (high risk)
    y = np.random.randint(0, 2, num_samples).astype(np.float32).reshape(-1, 1)

    # Inject fake risk features (e.g., if feature 2 & 4 are high, risk is higher)
    for i in range(num_samples):
        score = X[i, 2] * 2.0 + X[i, 4] * 1.5
        if score > 1.0:
            y[i, 0] = 1.0
        else:
            y[i, 0] = 0.0

    # Train / Val Split
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]

    train_loader = lm.DataLoader(lm.TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = lm.DataLoader(lm.TensorDataset(X_val, y_val), batch_size=8)

    model = HeartDiseaseMLP(in_dim=8)

    # --- Quantization Aware Training (QAT) ---
    print("Enabling Quantization Aware Training (QAT with STE)...")
    lm.prepare_qat(model, enabled=True)

    optimizer = lm.Adam(model.parameters(), lr=0.01)

    trainer = lm.Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lambda out, target: lm.binary_cross_entropy_loss(out, target, from_logits=True),
        verbose=1
    )

    # Train QAT model
    print("Starting QAT training...")
    trainer.fit(train_loader, val_loader, epochs=5)

    # Disable QAT for final deployment
    lm.prepare_qat(model, enabled=False)

    # Perform in-place INT8 Quantization
    model.quantize()
    print("In-place INT8 quantization successfully completed!")

    # Run evaluation
    loss, acc = trainer.evaluate(val_loader)
    print(f"\n✅ Validation Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}")

    # Export optimized model to ONNX format
    dummy_in = np.random.randn(1, 8).astype(np.float32)
    lm.export_to_onnx(model, dummy_in, "model/heart_disease_model.onnx")

    print("=" * 60)


if __name__ == "__main__":
    train_heart_predictor()
