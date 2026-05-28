"""
LowMind SGD Optimizer — with momentum, weight decay, Nesterov, and parameter groups.
"""
import numpy as np
from ..core.tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimizer with full production features.

    Supports **parameter groups** so you can apply different learning rates,
    weight decays, or momentum to different parts of your model.

    Args:
        params:       Iterable of Tensors OR list of param-group dicts.
        lr:           Default learning rate.
        momentum:     Default momentum (0 = plain SGD).
        weight_decay: Default L2 regularization coefficient.
        nesterov:     Enable Nesterov momentum (requires momentum > 0).

    Parameter groups example::

        optimizer = lm.SGD([
            {'params': model.encoder.parameters(), 'lr': 1e-4},
            {'params': model.head.parameters(),    'lr': 1e-3, 'weight_decay': 1e-4},
        ], lr=1e-3)

    Standard example::

        optimizer = lm.SGD(model.parameters(), lr=0.01, momentum=0.9)
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        if nesterov and momentum == 0:
            raise ValueError("Nesterov requires momentum > 0")
        self._defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        self._step = 0
        self._param_groups = self._parse_params(params)

    def _parse_params(self, params):
        groups = []
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for pg in params:
                group = dict(self._defaults)
                group.update({k: v for k, v in pg.items() if k != 'params'})
                param_list = list(pg['params'])
                group['params'] = param_list
                group['velocities'] = [np.zeros_like(p.data) for p in param_list]
                groups.append(group)
        else:
            param_list = list(params)
            group = dict(self._defaults)
            group['params'] = param_list
            group['velocities'] = [np.zeros_like(p.data) for p in param_list]
            groups.append(group)
        return groups

    def parameters(self):
        for g in self._param_groups:
            yield from g['params']

    def zero_grad(self):
        for g in self._param_groups:
            for p in g['params']:
                if p.grad is not None:
                    p.grad.fill(0.0)

    def step(self):
        self._step += 1
        for g in self._param_groups:
            lr = g['lr']
            momentum = g['momentum']
            wd = g['weight_decay']
            nesterov = g['nesterov']
            for p, v in zip(g['params'], g['velocities']):
                if p.grad is None:
                    continue
                grad = p.grad.copy()
                if wd != 0:
                    grad += wd * p.data
                if momentum != 0:
                    v[:] = momentum * v + grad
                    grad = grad + momentum * v if nesterov else v
                p.data -= lr * grad

    def get_lr(self):
        return self._param_groups[0]['lr']

    def set_lr(self, lr):
        for g in self._param_groups:
            g['lr'] = lr

    def add_param_group(self, group):
        """Add a new parameter group at runtime."""
        pg = dict(self._defaults)
        pg.update({k: v for k, v in group.items() if k != 'params'})
        param_list = list(group['params'])
        pg['params'] = param_list
        pg['velocities'] = [np.zeros_like(p.data) for p in param_list]
        self._param_groups.append(pg)

    def state_dict(self):
        return {
            'step': self._step,
            'defaults': self._defaults,
            'groups': [
                {k: v for k, v in g.items() if k not in ('params', 'velocities')}
                | {'velocities': [v.copy() for v in g['velocities']]}
                for g in self._param_groups
            ],
        }

    def load_state_dict(self, state):
        self._step = state['step']
        for g, sg in zip(self._param_groups, state['groups']):
            for k, v in sg.items():
                if k == 'velocities':
                    for vi, sv in zip(g['velocities'], v):
                        vi[:] = sv
                else:
                    g[k] = v

    def __repr__(self):
        d = self._defaults
        return (f"SGD(lr={d['lr']}, momentum={d['momentum']}, "
                f"weight_decay={d['weight_decay']}, nesterov={d['nesterov']})")
