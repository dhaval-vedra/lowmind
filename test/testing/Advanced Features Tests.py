"""
Comprehensive Tests for LowMind Advanced Optimizations: Weight Pruning and Knowledge Distillation
"""
import unittest
import numpy as np
import lowmind as lm


class TestAdvancedOptimizations(unittest.TestCase):

    def test_weight_pruning(self):
        """Test Weight Pruning and Sparsity API"""
        print("Testing Weight Pruning...")

        # Create a simple MLP model
        model = lm.Sequential(
            lm.Linear(10, 100),
            lm.ReLU(),
            lm.Linear(100, 5)
        )

        pruner = lm.Pruner(model)

        # Ensure initial sparsity is very low/zero
        initial_sparsity = pruner.calculate_sparsity()
        self.assertLess(initial_sparsity, 10.0)

        # Prune weights of the model by 50%
        pruner.prune_model(sparsity_ratio=0.5)

        # Verify sparsity has increased significantly
        sparsity = pruner.calculate_sparsity()
        self.assertGreaterEqual(sparsity, 40.0)  # skip biases, so not exactly 50% of the entire model, but close

        # Make a prediction with pruned model to ensure it still works
        dummy_input = lm.Tensor(np.random.randn(2, 10))
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 5))

        # Re-apply masks
        pruner.apply_masks()

        # Verify that weights are indeed zeroed where masks are zero
        for name, param in model.named_parameters():
            if name in pruner.masks:
                mask = pruner.masks[name]
                np.testing.assert_array_equal(param.data * (1 - mask), 0.0)

    def test_knowledge_distillation(self):
        """Test Knowledge Distillation Trainer"""
        print("Testing Knowledge Distillation...")

        # Teacher model (bigger/pretrained-like)
        teacher = lm.Sequential(
            lm.Linear(10, 20),
            lm.ReLU(),
            lm.Linear(20, 3)
        )

        # Student model (smaller/lightweight)
        student = lm.Sequential(
            lm.Linear(10, 5),
            lm.ReLU(),
            lm.Linear(5, 3)
        )

        # Create dummy classification dataset
        X = np.random.randn(10, 10).astype(np.float32)
        y = np.random.randint(0, 3, 10)
        dataset = lm.TensorDataset(X, y)
        loader = lm.DataLoader(dataset, batch_size=2)

        # Setup optimizer and KD trainer
        optimizer = lm.SGD(student.parameters(), lr=0.1)
        trainer = lm.DistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            optimizer=optimizer,
            loss_fn=lm.cross_entropy_loss,
            temperature=2.0,
            alpha=0.6,
            verbose=0
        )

        # Run 1 epoch of distillation training
        history = trainer.fit(loader, epochs=1)

        self.assertIn('train_loss', history)
        self.assertEqual(len(history['train_loss']), 1)
        print("Knowledge Distillation epoch completed successfully!")

    def test_model_quantization(self):
        """Test Model Quantization (INT8) simulation"""
        print("Testing INT8 Quantization...")

        # Create model with float32 weights
        model = lm.Sequential(
            lm.Linear(10, 5)
        )

        # Save original weights to compare
        orig_weights = model[0].weight.data.copy()

        # Quantize using in-place helper
        model.quantize()

        # Ensure weights are slightly changed due to quantization loss but are close
        quantized_weights = model[0].weight.data
        np.testing.assert_allclose(orig_weights, quantized_weights, atol=0.05)

        # Verify quantize_model works
        model_2 = lm.Sequential(
            lm.Linear(10, 5)
        )
        orig_w2 = model_2[0].weight.data.copy()
        scales = lm.quantize_model(model_2)

        self.assertIn("0.weight", scales)
        np.testing.assert_allclose(orig_w2, model_2[0].weight.data, atol=0.05)

        # Verify QuantizedTensor dequantization
        q_tensor, scale = lm.quantize_weight(orig_w2)
        qt = lm.QuantizedTensor(q_tensor, scale)
        dequant_w2 = qt.dequantize()
        np.testing.assert_allclose(orig_w2, dequant_w2, atol=0.05)

    def test_gradient_accumulation(self):
        """Test Gradient Accumulation under trainer"""
        print("Testing Gradient Accumulation...")

        model = lm.Sequential(
            lm.Linear(10, 3)
        )

        X = np.random.randn(8, 10).astype(np.float32)
        y = np.random.randint(0, 3, 8)
        dataset = lm.TensorDataset(X, y)
        loader = lm.DataLoader(dataset, batch_size=2)

        optimizer = lm.SGD(model.parameters(), lr=0.01)
        trainer = lm.Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=lm.cross_entropy_loss,
            grad_accum_steps=4,  # Accumulate over 4 steps (entire dataset of 8 samples with batch_size=2)
            verbose=0
        )

        history = trainer.fit(loader, epochs=1)
        self.assertIn('train_loss', history)


def run_tests():
    print("🧪 Running LowMind Advanced Optimization Tests...")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAdvancedOptimizations)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ADVANCED OPTIMIZATIONS PASSED! ✅")
    else:
        print("❌ SOME ADVANCED OPTIMIZATIONS FAILED!")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
