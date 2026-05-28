"""
Integration tests — full training pipelines, edge cases, production readiness.
Tests the entire system end-to-end.
"""
import numpy as np
import pytest
import tempfile, os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lowmind as lm


np.random.seed(42)


def make_classification_data(N=200, D=16, C=4):
    X = np.random.randn(N, D).astype(np.float32)
    W = np.random.randn(D, C).astype(np.float32)
    y = (X @ W).argmax(axis=1)
    return X, y


def make_regression_data(N=200, D=8):
    X = np.random.randn(N, D).astype(np.float32)
    w_true = np.random.randn(D, 1).astype(np.float32)
    y = (X @ w_true) + np.random.randn(N, 1).astype(np.float32) * 0.1
    return X, y.astype(np.float32)


class TestFullTrainingPipeline:
    """End-to-end: build model → train → evaluate."""

    def test_mlp_classification(self):
        X, y = make_classification_data(N=200, D=16, C=4)
        X_t = lm.Tensor(X)
        y_t = lm.Tensor(y.astype(np.int64))

        model = lm.Sequential(
            lm.Linear(16, 32), lm.ReLU(), lm.Linear(32, 4)
        )
        opt = lm.Adam(model.parameters(), lr=3e-3)

        initial_loss = float(lm.cross_entropy_loss(model(X_t), y_t).item())
        for _ in range(100):
            opt.zero_grad()
            loss = lm.cross_entropy_loss(model(X_t), y_t)
            loss.backward()
            opt.step()
        final_loss = float(lm.cross_entropy_loss(model(X_t), y_t).item())

        assert final_loss < initial_loss, "Loss did not decrease during training"
        acc = lm.accuracy(model(X_t).data.argmax(axis=1), y)
        assert acc > 0.70, f"Accuracy too low: {acc}"

    def test_regression_mse(self):
        X, y = make_regression_data(N=200, D=8)
        X_t = lm.Tensor(X)
        y_t = lm.Tensor(y)

        model = lm.Sequential(
            lm.Linear(8, 32), lm.ReLU(), lm.Linear(32, 1)
        )
        opt = lm.SGD(model.parameters(), lr=0.01, momentum=0.9)

        for _ in range(150):
            opt.zero_grad()
            pred = model(X_t)
            loss = lm.mse_loss(pred, y_t)
            loss.backward()
            opt.step()
        final_mse = float(lm.mse_loss(model(X_t), y_t).item())
        assert final_mse < 0.5, f"MSE too high: {final_mse}"

    def test_dataloader_trainer(self):
        X, y = make_classification_data(N=160, D=12, C=3)
        ds = lm.TensorDataset(lm.Tensor(X), lm.Tensor(y.astype(np.int64)))
        train_ds, val_ds = lm.train_test_split(ds, test_size=0.25, random_state=0)
        train_loader = lm.DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader   = lm.DataLoader(val_ds,   batch_size=16, shuffle=False)

        model = lm.Sequential(lm.Linear(12, 32), lm.ReLU(), lm.Linear(32, 3))
        trainer = lm.Trainer(
            model=model,
            optimizer=lm.Adam(model.parameters(), lr=5e-3),
            loss_fn=lm.cross_entropy_loss,
            verbose=0,
        )
        history = trainer.fit(train_loader, val_loader, epochs=30)
        assert len(history['train_loss']) == 30
        assert history['val_loss'][-1] < history['train_loss'][0]

    def test_gradient_accumulation(self):
        X, y = make_classification_data(N=120, D=8, C=3)
        ds = lm.TensorDataset(lm.Tensor(X), lm.Tensor(y.astype(np.int64)))
        loader = lm.DataLoader(ds, batch_size=8, shuffle=True)
        model = lm.Sequential(lm.Linear(8, 16), lm.ReLU(), lm.Linear(16, 3))

        trainer = lm.Trainer(
            model=model,
            optimizer=lm.Adam(model.parameters(), lr=3e-3),
            loss_fn=lm.cross_entropy_loss,
            grad_accum_steps=4,
            verbose=0,
        )
        history = trainer.fit(loader, epochs=20)
        assert not np.isnan(history['train_loss'][-1])


class TestSaveLoadRoundtrip:
    def test_save_load_predictions_match(self):
        model = lm.Sequential(lm.Linear(8, 16), lm.ReLU(), lm.Linear(16, 3))
        X = lm.Tensor(np.random.randn(10, 8).astype(np.float32))
        model.eval()
        pred1 = model(X).data

        with tempfile.NamedTemporaryFile(suffix='.lmz', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model2 = lm.Sequential(lm.Linear(8, 16), lm.ReLU(), lm.Linear(16, 3))
            model2.load(path)
            model2.eval()
            pred2 = model2(X).data
            np.testing.assert_allclose(pred1, pred2, atol=1e-6)
        finally:
            os.unlink(path)

    def test_state_dict_roundtrip(self):
        model = lm.Sequential(lm.Linear(4, 8), lm.Linear(8, 2))
        sd = model.state_dict()
        model2 = lm.Sequential(lm.Linear(4, 8), lm.Linear(8, 2))
        model2.load_state_dict(sd)
        for k in sd:
            np.testing.assert_allclose(sd[k], model2.state_dict()[k])


class TestParameterGroups:
    def test_adam_parameter_groups(self):
        model = lm.Sequential(lm.Linear(8, 16), lm.ReLU(), lm.Linear(16, 4))
        backbone = list(model._modules['0'].parameters())
        head     = list(model._modules['2'].parameters())

        opt = lm.Adam([
            {'params': backbone, 'lr': 1e-4},
            {'params': head,     'lr': 1e-2},
        ])
        assert len(opt._param_groups) == 2
        assert opt._param_groups[0]['lr'] == 1e-4
        assert opt._param_groups[1]['lr'] == 1e-2

        X = lm.Tensor(np.random.randn(5, 8).astype(np.float32))
        y = lm.Tensor(np.array([0, 1, 2, 3, 0], dtype=np.int64))
        loss = lm.cross_entropy_loss(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_sgd_parameter_groups(self):
        model = lm.Sequential(lm.Linear(4, 8), lm.Linear(8, 2))
        opt = lm.SGD([
            {'params': list(model._modules['0'].parameters()), 'lr': 0.001},
            {'params': list(model._modules['1'].parameters()), 'lr': 0.01, 'momentum': 0.9},
        ], lr=0.005)
        assert opt._param_groups[0]['lr'] == 0.001
        assert opt._param_groups[1]['momentum'] == 0.9


class TestGradientClipping:
    def test_clip_grad_norm(self):
        model = lm.Sequential(lm.Linear(4, 8), lm.ReLU(), lm.Linear(8, 2))
        X = lm.Tensor(np.ones((3, 4), dtype=np.float32) * 1000)  # large input
        y = lm.Tensor(np.array([0, 1, 0], dtype=np.int64))
        loss = lm.cross_entropy_loss(model(X), y)
        loss.backward()

        # Clip to small norm
        lm.clip_grad_norm(model.parameters(), max_norm=1.0)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += float((p.grad ** 2).sum())
        total_norm = total_norm ** 0.5
        assert total_norm <= 1.01, f"Gradient norm too large after clipping: {total_norm}"


class TestTransforms:
    def test_normalize(self):
        x = lm.Tensor(np.random.randn(100, 16).astype(np.float32) * 5 + 3)
        norm = lm.Normalize(mean=3.0, std=5.0)
        out = norm(x)
        np.testing.assert_allclose(out.data.mean(), 0.0, atol=0.1)
        np.testing.assert_allclose(out.data.std(), 1.0, atol=0.15)

    def test_horizontal_flip(self):
        x = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)
        flip = lm.RandomHorizontalFlip(p=1.0)
        out = flip(x)
        np.testing.assert_allclose(out[:, :, :, ::-1], x)

    def test_gaussian_noise_shape(self):
        x = lm.Tensor(np.zeros((8, 16), dtype=np.float32))
        noise = lm.GaussianNoise(std=1.0, p=1.0)
        out = noise(x)
        assert out.data.shape == (8, 16)
        assert not np.all(out.data == 0)

    def test_compose(self):
        x = lm.Tensor((np.random.randn(5, 32) * 3 + 1).astype(np.float32))
        compose = lm.Compose([
            lm.Normalize(mean=1.0, std=3.0),
            lm.GaussianNoise(std=0.01, p=1.0),
        ])
        out = compose(x)
        assert out.data.shape == (5, 32)


class TestNoGradProductionPattern:
    def test_inference_loop_no_grad_accumulation(self):
        """Verify no gradient memory leak during repeated inference."""
        model = lm.Sequential(lm.Linear(16, 32), lm.ReLU(), lm.Linear(32, 4))
        model.eval()

        for _ in range(20):
            x = lm.Tensor(np.random.randn(8, 16).astype(np.float32))
            with lm.no_grad():
                out = model(x)

        # Parameters should have no gradients from inference
        for p in model.parameters():
            assert p.grad is None or not p.grad.any(), \
                "Gradient was unexpectedly accumulated during no_grad inference"

    def test_trainer_predict_proba(self):
        X, y = make_classification_data(N=80, D=8, C=3)
        model = lm.Sequential(lm.Linear(8, 16), lm.ReLU(), lm.Linear(16, 3))
        trainer = lm.Trainer(model, lm.Adam(model.parameters()), lm.cross_entropy_loss, verbose=0)
        probs = trainer.predict_proba(X)
        assert probs.shape == (80, 3)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(80), atol=1e-5)
