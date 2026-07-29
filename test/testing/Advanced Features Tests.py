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

    def test_gradient_checkpointing(self):
        """Test Gradient Checkpointing function"""
        print("Testing Gradient Checkpointing...")

        # Define a simple function/layer block (e.g. Sequential block)
        block = lm.Sequential(
            lm.Linear(10, 20),
            lm.ReLU(),
            lm.Linear(20, 5)
        )

        # Run standard forward and backward pass
        x1 = lm.Tensor(np.random.randn(2, 10), requires_grad=True)
        out1 = block(x1)
        loss1 = out1.sum()

        optimizer1 = lm.SGD(block.parameters(), lr=0.1)
        optimizer1.zero_grad()
        loss1.backward()

        # Save standard gradients
        grad_x1 = x1.grad.copy() if x1.grad is not None else None
        grads_block = [p.grad.copy() for p in block.parameters() if p.grad is not None]

        # Reset gradients
        optimizer1.zero_grad()
        if x1.grad is not None:
            x1.grad.fill(0.0)

        # Run with checkpointing
        x2 = lm.Tensor(x1.data.copy(), requires_grad=True)
        out2 = lm.checkpoint(block, x2)
        loss2 = out2.sum()

        loss2.backward()

        # Verify that output values match
        np.testing.assert_allclose(out1.data, out2.data, rtol=1e-5)

        # Verify that input gradients match exactly
        if grad_x1 is not None:
            np.testing.assert_allclose(grad_x1, x2.grad, rtol=1e-5)

        # Verify that parameters gradients match exactly
        grads_checkpointed = [p.grad.copy() for p in block.parameters() if p.grad is not None]
        for g1, g2 in zip(grads_block, grads_checkpointed):
            np.testing.assert_allclose(g1, g2, rtol=1e-5)

        print("Gradient Checkpointing gradients match standard backpropagation exactly!")

    def test_quantization_aware_training(self):
        """Test Quantization Aware Training (QAT) with STE"""
        print("Testing Quantization Aware Training (QAT)...")

        # Create a simple MLP model
        model = lm.Sequential(
            lm.Linear(10, 5)
        )

        # Enable QAT
        lm.prepare_qat(model, enabled=True)
        self.assertTrue(getattr(model[0], 'qat_mode', False))

        # Generate dummy input and target
        x = lm.Tensor(np.random.randn(2, 10))
        y = lm.Tensor([1, 2])

        # Run forward pass
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

        # Run backward pass
        loss = lm.cross_entropy_loss(out, y)
        loss.backward()

        # Ensure gradients are propagated back to weights through STE
        self.assertIsNotNone(model[0].weight.grad)
        self.assertGreater(np.sum(np.abs(model[0].weight.grad)), 0.0)

        # Turn off QAT
        lm.prepare_qat(model, enabled=False)
        self.assertFalse(getattr(model[0], 'qat_mode', True))

    def test_onnx_exporter(self):
        """Test exporting LowMind model to standard ONNX format"""
        print("Testing ONNX Exporter...")
        import os

        # Create a Sequential model
        model = lm.Sequential(
            lm.Linear(10, 20),
            lm.ReLU(),
            lm.Linear(20, 3)
        )

        # Define dummy input
        dummy_input = np.random.randn(2, 10).astype(np.float32)

        # Target filepath
        filepath = "test_model.onnx"

        # Export to ONNX
        onnx_model = lm.export_to_onnx(model, dummy_input, filepath)

        # Check if file exists and has size > 0
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)

        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_auto_tuning_batch_size(self):
        """Test Auto-tuning Dynamic Batch Size on MemoryError"""
        print("Testing Auto-tuning Dynamic Batch Size...")

        # Create a simple MLP model
        model = lm.Sequential(
            lm.Linear(10, 5)
        )

        # Generate dummy input and target
        X = np.random.randn(10, 10).astype(np.float32)
        y = np.random.randint(0, 5, 10)

        # Mock forward to raise MemoryError if batch size is >= 8
        def mock_forward(x):
            if x.shape[0] >= 8:
                raise MemoryError("Out of Memory simulated for batch size >= 8")
            # Call original layer 0 forward
            return model[0].forward(x)

        model.forward = mock_forward

        loader = lm.DataLoader(lm.TensorDataset(X, y), batch_size=8)
        optimizer = lm.SGD(model.parameters(), lr=0.01)

        trainer = lm.Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=lm.cross_entropy_loss,
            grad_accum_steps=1,
            auto_tune_batch_size=True,
            verbose=0
        )

        # Running fit should hit MemoryError at batch_size=8,
        # auto-tune to batch_size=4 and complete successfully!
        history = trainer.fit(loader, epochs=1)
        self.assertIn('train_loss', history)
        self.assertLess(loader.batch_size, 8)
        self.assertGreater(trainer.grad_accum_steps, 1)
        print(f"Auto-tuning verified: new batch_size={loader.batch_size}, grad_accum_steps={trainer.grad_accum_steps}!")


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
