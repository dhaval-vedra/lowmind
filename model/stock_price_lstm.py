"""
Stock Price Timeseries Forecaster using LowMind LSTM Recurrent Network
"""
import lowmind as lm
import numpy as np


class StockForecasterLSTM(lm.Module):
    """
    LSTM Recurrent Neural Network for financial timeseries stock price forecasting.
    Input features: 1 (previous day close price)
    Hidden dim: 16
    Output: 1 (next day price prediction)
    """
    def __init__(self, in_dim=1, hidden_dim=16, out_dim=1):
        super().__init__()
        self.lstm = lm.LSTM(in_dim, hidden_dim, num_layers=1)
        self.fc = lm.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, in_dim)
        lstm_out, _ = self.lstm(x)
        # We only take the output of the last sequence step for prediction
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out


def train_stock_forecaster():
    print("=" * 60)
    print("📈 Training LSTM Financial Stock Price Forecaster")
    print("=" * 60)

    # Generate mock stock timeseries dataset
    # 100 sequences of length 5 (previous 5 days) to predict next close
    np.random.seed(42)
    num_samples = 100
    seq_len = 5
    X = np.zeros((num_samples, seq_len, 1), dtype=np.float32)
    y = np.zeros((num_samples, 1), dtype=np.float32)

    # Mock a sine-wave like stock price pattern with noise
    for i in range(num_samples):
        start_phase = i * 0.1
        seq_values = np.sin(start_phase + np.arange(seq_len) * 0.2) + np.random.randn(seq_len) * 0.05
        X[i, :, 0] = seq_values
        y[i, 0] = np.sin(start_phase + seq_len * 0.2)

    # Train / Val Split
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    train_loader = lm.DataLoader(lm.TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    val_loader = lm.DataLoader(lm.TensorDataset(X_val, y_val), batch_size=4)

    model = StockForecasterLSTM(in_dim=1, hidden_dim=16, out_dim=1)
    optimizer = lm.Adam(model.parameters(), lr=0.01)

    trainer = lm.Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lm.mse_loss,  # regression uses Mean Squared Error
        verbose=1
    )

    # Train LSTM for 5 epochs
    print("Starting LSTM training...")
    trainer.fit(train_loader, val_loader, epochs=5)

    # Run evaluation
    loss, accuracy_or_metrics = trainer.evaluate(val_loader)
    print(f"\n✅ Validation MSE Loss: {loss:.6f}")

    # --- Edge Optimization & Quantization ---
    print("\n📦 Production-level Edge Optimization:")

    # In-place INT8 Quantization
    model.quantize()
    print("   🎯 LSTM weights quantized to INT8 simulation.")

    # Export LSTM model to standard ONNX format
    dummy_in = np.random.randn(1, 5, 1).astype(np.float32)
    lm.export_to_onnx(model, dummy_in, "model/stock_lstm_model.onnx")

    print("=" * 60)


if __name__ == "__main__":
    train_stock_forecaster()
