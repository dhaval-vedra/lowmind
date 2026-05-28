"""
LowMind Adam, AdamW, RMSprop, AdaGrad — all with parameter group support.
"""
import numpy as np


def _parse_groups(params, defaults):
    groups = []
    if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
        for pg in params:
            group = dict(defaults)
            group.update({k: v for k, v in pg.items() if k != 'params'})
            group['params'] = list(pg['params'])
            groups.append(group)
    else:
        group = dict(defaults)
        group['params'] = list(params)
        groups.append(group)
    return groups


class Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).

    Supports **parameter groups** for per-layer learning rates.

    Args:
        params:       Iterable of Tensors OR list of param-group dicts.
        lr:           Learning rate (default 1e-3).
        betas:        (beta1, beta2) — default (0.9, 0.999).
        eps:          Numerical stability (default 1e-8).
        weight_decay: L2 regularization (default 0).
        amsgrad:      Use AMSGrad variant (default False).

    Parameter groups example::

        optimizer = lm.Adam([
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': model.head.parameters(),     'lr': 1e-3},
        ])

    Standard example::

        optimizer = lm.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        self._defaults = dict(lr=lr, betas=betas, eps=eps,
                              weight_decay=weight_decay, amsgrad=amsgrad)
        self._step = 0
        self._param_groups = _parse_groups(params, self._defaults)
        for g in self._param_groups:
            n = len(g['params'])
            g['m'] = [np.zeros_like(p.data) for p in g['params']]
            g['v'] = [np.zeros_like(p.data) for p in g['params']]
            g['v_max'] = [np.zeros_like(p.data) for p in g['params']] if g.get('amsgrad') else None

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
        t = self._step
        for g in self._param_groups:
            lr = g['lr']
            b1, b2 = g['betas']
            eps = g['eps']
            wd = g['weight_decay']
            amsgrad = g.get('amsgrad', False)
            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
            for i, p in enumerate(g['params']):
                if p.grad is None:
                    continue
                grad = p.grad.copy()
                if wd != 0:
                    grad += wd * p.data
                g['m'][i] = b1 * g['m'][i] + (1 - b1) * grad
                g['v'][i] = b2 * g['v'][i] + (1 - b2) * (grad ** 2)
                m_hat = g['m'][i] / bc1
                v_hat = g['v'][i] / bc2
                if amsgrad:
                    g['v_max'][i] = np.maximum(g['v_max'][i], v_hat)
                    denom = np.sqrt(g['v_max'][i]) + eps
                else:
                    denom = np.sqrt(v_hat) + eps
                p.data -= lr * m_hat / denom

    def get_lr(self):
        return self._param_groups[0]['lr']

    def set_lr(self, lr):
        for g in self._param_groups:
            g['lr'] = lr

    def add_param_group(self, group):
        pg = dict(self._defaults)
        pg.update({k: v for k, v in group.items() if k != 'params'})
        pg['params'] = list(group['params'])
        n = len(pg['params'])
        pg['m'] = [np.zeros_like(p.data) for p in pg['params']]
        pg['v'] = [np.zeros_like(p.data) for p in pg['params']]
        pg['v_max'] = [np.zeros_like(p.data) for p in pg['params']] if pg.get('amsgrad') else None
        self._param_groups.append(pg)

    def state_dict(self):
        return {
            'step': self._step,
            'defaults': self._defaults,
            'groups': [
                {k: v for k, v in g.items() if k not in ('params', 'm', 'v', 'v_max')}
                | {'m': [m.copy() for m in g['m']],
                   'v': [v.copy() for v in g['v']]}
                for g in self._param_groups
            ],
        }

    def load_state_dict(self, state):
        self._step = state['step']
        for g, sg in zip(self._param_groups, state['groups']):
            for i, (m, v) in enumerate(zip(sg['m'], sg['v'])):
                g['m'][i][:] = m
                g['v'][i][:] = v
            for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad'):
                if k in sg:
                    g[k] = sg[k]

    def __repr__(self):
        d = self._defaults
        return (f"Adam(lr={d['lr']}, betas={d['betas']}, "
                f"eps={d['eps']}, weight_decay={d['weight_decay']})")


class AdamW(Adam):
    """
    AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Applies weight decay directly to parameters (not gradients),
    which is theoretically cleaner and often gives better generalization.
    """

    def step(self):
        self._step += 1
        t = self._step
        for g in self._param_groups:
            lr = g['lr']
            b1, b2 = g['betas']
            eps = g['eps']
            wd = g['weight_decay']
            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
            for i, p in enumerate(g['params']):
                if p.grad is None:
                    continue
                grad = p.grad.copy()
                if wd != 0:
                    p.data *= (1 - lr * wd)      # decoupled decay
                g['m'][i] = b1 * g['m'][i] + (1 - b1) * grad
                g['v'][i] = b2 * g['v'][i] + (1 - b2) * (grad ** 2)
                m_hat = g['m'][i] / bc1
                v_hat = g['v'][i] / bc2
                p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def __repr__(self):
        d = self._defaults
        return f"AdamW(lr={d['lr']}, weight_decay={d['weight_decay']})"


class RMSprop:
    """
    RMSprop optimizer with parameter group support.

    Args:
        params:       Iterable of Tensors OR list of param-group dicts.
        lr:           Learning rate (default 1e-2).
        alpha:        Smoothing constant (default 0.99).
        eps:          Numerical stability (default 1e-8).
        weight_decay: L2 regularization (default 0).
        momentum:     Momentum factor (default 0).
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8,
                 weight_decay=0.0, momentum=0.0):
        self._defaults = dict(lr=lr, alpha=alpha, eps=eps,
                              weight_decay=weight_decay, momentum=momentum)
        self._step = 0
        self._param_groups = _parse_groups(params, self._defaults)
        for g in self._param_groups:
            g['v'] = [np.zeros_like(p.data) for p in g['params']]
            g['buf'] = [np.zeros_like(p.data) for p in g['params']]

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
            lr, alpha, eps = g['lr'], g['alpha'], g['eps']
            wd, mom = g['weight_decay'], g['momentum']
            for i, p in enumerate(g['params']):
                if p.grad is None:
                    continue
                grad = p.grad.copy()
                if wd != 0:
                    grad += wd * p.data
                g['v'][i] = alpha * g['v'][i] + (1 - alpha) * grad ** 2
                update = grad / (np.sqrt(g['v'][i]) + eps)
                if mom != 0:
                    g['buf'][i] = mom * g['buf'][i] + update
                    update = g['buf'][i]
                p.data -= lr * update

    def get_lr(self):
        return self._param_groups[0]['lr']

    def set_lr(self, lr):
        for g in self._param_groups:
            g['lr'] = lr

    def __repr__(self):
        d = self._defaults
        return f"RMSprop(lr={d['lr']}, alpha={d['alpha']}, momentum={d['momentum']})"


class AdaGrad:
    """
    AdaGrad optimizer with parameter group support.

    Args:
        params:       Iterable of Tensors OR list of param-group dicts.
        lr:           Learning rate (default 1e-2).
        eps:          Numerical stability (default 1e-10).
        weight_decay: L2 regularization (default 0).
    """

    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        self._defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        self._step = 0
        self._param_groups = _parse_groups(params, self._defaults)
        for g in self._param_groups:
            g['accum'] = [np.zeros_like(p.data) for p in g['params']]

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
            lr, eps, wd = g['lr'], g['eps'], g['weight_decay']
            for i, p in enumerate(g['params']):
                if p.grad is None:
                    continue
                grad = p.grad.copy()
                if wd != 0:
                    grad += wd * p.data
                g['accum'][i] += grad ** 2
                p.data -= lr * grad / (np.sqrt(g['accum'][i]) + eps)

    def get_lr(self):
        return self._param_groups[0]['lr']

    def set_lr(self, lr):
        for g in self._param_groups:
            g['lr'] = lr

    def __repr__(self):
        d = self._defaults
        return f"AdaGrad(lr={d['lr']}, eps={d['eps']})"
