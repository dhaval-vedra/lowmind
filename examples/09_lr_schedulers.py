"""
Example 09 — Learning Rate Schedulers
=======================================
Visualize how different LR schedulers change the learning rate over epochs.
Demonstrates: StepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR,
              LinearWarmupLR, ExponentialLR.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lowmind as lm
import numpy as np

print("=" * 60)
print("  LowMind Example 09 — LR Schedulers")
print("=" * 60)

EPOCHS = 50
BASE_LR = 0.1

def dummy_optimizer(lr=BASE_LR):
    """Create a single-param optimizer as a scheduler target."""
    param = lm.Tensor([0.0], requires_grad=True)
    return lm.SGD([param], lr=lr)

def track_lr(scheduler_name, scheduler, n_epochs, metric_fn=None):
    lrs = [scheduler.optimizer.get_lr() if hasattr(scheduler, 'optimizer')
           else scheduler.optimizer.get_lr()]
    for e in range(n_epochs):
        if metric_fn is not None:
            metric = metric_fn(e)
            scheduler.step(metric)
        else:
            scheduler.step()
        cur_lr = (scheduler.optimizer.get_lr() if hasattr(scheduler, 'optimizer')
                  else scheduler.optimizer.get_lr())
        lrs.append(cur_lr)
    return lrs

schedulers = {
    "StepLR(step=10, gamma=0.5)": lambda: lm.StepLR(dummy_optimizer(), step_size=10, gamma=0.5),
    "ExponentialLR(gamma=0.95) ": lambda: lm.ExponentialLR(dummy_optimizer(), gamma=0.95),
    "CosineAnnealingLR(T=25)  ": lambda: lm.CosineAnnealingLR(dummy_optimizer(), T_max=25, eta_min=1e-4),
    "MultiStepLR([15,30,45])  ": lambda: lm.MultiStepLR(dummy_optimizer(), milestones=[15, 30, 45], gamma=0.3),
    "LinearWarmup(warmup=10)  ": lambda: lm.LinearWarmupLR(dummy_optimizer(0.0), warmup_steps=10, target_lr=BASE_LR),
}

print(f"\n  Base LR: {BASE_LR}  |  Epochs: {EPOCHS}")
print(f"\n  {'Scheduler':<35} {'LR at 0':>9} {'LR at 25':>9} {'LR at 50':>9}")
print("  " + "-" * 65)

all_lrs = {}
for name, ctor in schedulers.items():
    s = ctor()
    lrs = track_lr(name, s, EPOCHS)
    all_lrs[name] = lrs
    print(f"  {name} {lrs[0]:>9.5f} {lrs[25]:>9.5f} {lrs[50]:>9.5f}")

# ReduceLROnPlateau (separate — needs metric)
print("\n  ReduceLROnPlateau (patience=5, factor=0.5):")
opt = dummy_optimizer()
sched = lm.ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=True)
plateau_lrs = [opt.get_lr()]
# Simulate a slowly improving then stagnating loss
for e in range(EPOCHS):
    simulated_loss = max(0.5, 2.0 - 0.05 * e + 0.2 * (e // 10))
    sched.step(simulated_loss)
    plateau_lrs.append(opt.get_lr())
print(f"  LR at 0  : {plateau_lrs[0]:.5f}")
print(f"  LR at 25 : {plateau_lrs[25]:.5f}")
print(f"  LR at 50 : {plateau_lrs[50]:.5f}")

# CyclicLR
print("\n  CyclicLR (base=1e-4, max=1e-1, step_size=10):")
opt2 = dummy_optimizer(lr=1e-4)
cyclic = lm.CyclicLR(opt2, base_lr=1e-4, max_lr=1e-1, step_size=10)
cyclic_lrs = [opt2.get_lr()]
for _ in range(EPOCHS):
    cyclic.step()
    cyclic_lrs.append(opt2.get_lr())
print(f"  LR at 0  : {cyclic_lrs[0]:.5f}")
print(f"  LR at 10 : {cyclic_lrs[10]:.5f}")
print(f"  LR at 20 : {cyclic_lrs[20]:.5f}")
print(f"  LR at 30 : {cyclic_lrs[30]:.5f}")

print("\n  LR Schedulers example complete!")
print("  Tip: plot the 'lrs' lists with matplotlib for visualization.")
