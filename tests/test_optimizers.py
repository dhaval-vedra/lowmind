"""Tests for optimizers and learning rate schedulers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import lowmind as lm


def simple_model():
    return lm.Linear(4, 2)


def run_steps(model, optimizer, n=5):
    X = lm.Tensor(np.random.randn(8, 4).astype(np.float32))
    y = lm.Tensor(np.random.randint(0, 2, 8))
    for _ in range(n):
        optimizer.zero_grad()
        loss = lm.cross_entropy_loss(model(X), y)
        loss.backward()
        optimizer.step()
    return loss


class TestSGD:
    def test_loss_decreases(self):
        np.random.seed(0)
        model = simple_model()
        opt = lm.SGD(model.parameters(), lr=0.1)
        X = lm.Tensor(np.random.randn(32, 4).astype(np.float32))
        y = lm.Tensor(np.random.randint(0, 2, 32))
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = lm.cross_entropy_loss(model(X), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0]

    def test_momentum(self):
        model = simple_model()
        opt = lm.SGD(model.parameters(), lr=0.01, momentum=0.9)
        run_steps(model, opt)

    def test_weight_decay(self):
        model = simple_model()
        opt = lm.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
        run_steps(model, opt)

    def test_zero_grad_resets(self):
        model = simple_model()
        opt = lm.SGD(model.parameters(), lr=0.01)
        X = lm.Tensor(np.random.randn(4, 4).astype(np.float32))
        y = lm.Tensor(np.array([0, 1, 0, 1]))
        lm.cross_entropy_loss(model(X), y).backward()
        opt.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(p.grad == 0)


class TestAdam:
    def test_loss_decreases(self):
        np.random.seed(1)
        model = simple_model()
        opt = lm.Adam(model.parameters(), lr=1e-2)
        X = lm.Tensor(np.random.randn(32, 4).astype(np.float32))
        y = lm.Tensor(np.random.randint(0, 2, 32))
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = lm.cross_entropy_loss(model(X), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0]

    def test_adamw_weight_decay(self):
        model = simple_model()
        opt = lm.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        run_steps(model, opt)

    def test_amsgrad(self):
        model = simple_model()
        opt = lm.Adam(model.parameters(), lr=1e-3, amsgrad=True)
        run_steps(model, opt)


class TestRMSprop:
    def test_loss_decreases(self):
        np.random.seed(2)
        model = simple_model()
        opt = lm.RMSprop(model.parameters(), lr=1e-3)
        X = lm.Tensor(np.random.randn(32, 4).astype(np.float32))
        y = lm.Tensor(np.random.randint(0, 2, 32))
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = lm.cross_entropy_loss(model(X), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0]


class TestSchedulers:
    def _make_opt(self, lr=0.1):
        param = lm.Tensor([0.0], requires_grad=True)
        return lm.SGD([param], lr=lr)

    def test_step_lr_decays(self):
        opt = self._make_opt(0.1)
        sched = lm.StepLR(opt, step_size=5, gamma=0.5)
        for _ in range(5):
            sched.step()
        assert opt.get_lr() == pytest.approx(0.05, rel=1e-4)

    def test_cosine_lr(self):
        opt = self._make_opt(0.1)
        sched = lm.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.get_lr())
        assert lrs[-1] == pytest.approx(0.0, abs=1e-5)

    def test_reduce_on_plateau(self):
        opt = self._make_opt(0.1)
        sched = lm.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        for _ in range(4):   # 4 non-improving steps
            sched.step(metric=1.0)
        assert opt.get_lr() == pytest.approx(0.05, rel=1e-4)

    def test_linear_warmup(self):
        opt = self._make_opt(0.0)
        sched = lm.LinearWarmupLR(opt, warmup_steps=5, target_lr=0.1)
        lrs = []
        for _ in range(5):
            lrs.append(opt.get_lr())
            sched.step()
        assert lrs[0] < lrs[-1]
        assert opt.get_lr() == pytest.approx(0.1, rel=1e-4)
