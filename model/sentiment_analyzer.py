"""
NLP Sentiment Analyzer (Binary Classification) using LowMind Embedding, GRU and Knowledge Distillation
"""
import lowmind as lm
import numpy as np


class SentimentAnalyzerGRU(lm.Module):
    """
    Lightweight GRU-based Sentiment Classifier.
    Input shape: (batch_size, sequence_length) -> integer word indices
    Output: (batch_size, 2) -> logits for [NEGATIVE, POSITIVE]
    """
    def __init__(self, vocab_size=50, embed_dim=8, hidden_dim=8, num_classes=2):
        super().__init__()
        self.embedding = lm.Embedding(vocab_size, embed_dim)
        self.gru = lm.GRU(embed_dim, hidden_dim, num_layers=1)
        self.fc = lm.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is word indices shape (N, seq_len)
        embeds = self.embedding(x)  # (N, seq_len, embed_dim)
        gru_out, _ = self.gru(embeds)  # (N, seq_len, hidden_dim)
        last_step = gru_out[:, -1, :]  # (N, hidden_dim)
        out = self.fc(last_step)
        return out


def train_sentiment_analyzer():
    print("=" * 60)
    print("🎭 Training NLP Sentiment Analyzer with Knowledge Distillation")
    print("=" * 60)

    # Vocabulary size and sequence length
    vocab_size = 50
    seq_len = 6
    num_samples = 100

    # Generate mock review sentiment dataset
    np.random.seed(42)
    X = np.random.randint(0, vocab_size, (num_samples, seq_len))
    # Target label: 0 (negative), 1 (positive)
    y = np.random.randint(0, 2, num_samples)

    # Set positive words to trigger positive label
    for i in range(num_samples):
        # If words indices 5 or 12 or 22 are in the sequence, class is highly likely positive
        if 5 in X[i] or 12 in X[i] or 22 in X[i]:
            y[i] = 1
        else:
            y[i] = 0

    # Train / Val Split
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    train_loader = lm.DataLoader(lm.TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = lm.DataLoader(lm.TensorDataset(X_val, y_val), batch_size=8)

    # 1. Define Teacher Model (Slightly larger, pre-trained-like)
    print("Initializing Teacher and Student models...")
    teacher = SentimentAnalyzerGRU(vocab_size=vocab_size, embed_dim=16, hidden_dim=16)
    # Fit teacher first as teacher baseline
    teacher_optimizer = lm.Adam(teacher.parameters(), lr=0.01)
    teacher_trainer = lm.Trainer(model=teacher, optimizer=teacher_optimizer, loss_fn=lm.cross_entropy_loss, verbose=0)
    teacher_trainer.fit(train_loader, epochs=3)
    print("   ✅ Teacher pre-training complete.")

    # 2. Define Student Model (Tiny, ultra-lightweight)
    student = SentimentAnalyzerGRU(vocab_size=vocab_size, embed_dim=8, hidden_dim=8)
    student_optimizer = lm.Adam(student.parameters(), lr=0.01)

    # 3. Setup DistillationTrainer (combines student loss with scaled soft teacher logits)
    print("Starting Knowledge Distillation training...")
    distill_trainer = lm.DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        optimizer=student_optimizer,
        loss_fn=lm.cross_entropy_loss,
        temperature=3.0,
        alpha=0.6,
        verbose=1
    )

    # Distill teacher's knowledge into student
    distill_trainer.fit(train_loader, val_loader, epochs=5)

    # Run evaluation
    loss, acc = distill_trainer.evaluate(val_loader)
    print(f"\n✅ Student Validation Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}")

    # --- Edge Optimization & Quantization ---
    print("\n📦 Production-level Edge Optimization:")

    # 1. Prune student model
    pruner = lm.Pruner(student)
    pruner.prune_model(sparsity_ratio=0.2)
    print(f"   ✂️ Pruned Student weight sparsity: {pruner.calculate_sparsity():.2f}%")

    # 2. In-place INT8 Quantization
    student.quantize()
    print("   🎯 Student weights quantized to INT8 simulation.")

    # 3. Export student model to standard ONNX format
    dummy_in = np.random.randint(0, vocab_size, (1, 6))
    lm.export_to_onnx(student, dummy_in, "model/sentiment_gru_model.onnx")

    print("=" * 60)


if __name__ == "__main__":
    train_sentiment_analyzer()
