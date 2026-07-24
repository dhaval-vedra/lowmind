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
