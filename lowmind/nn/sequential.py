"""
LowMind Sequential — stack layers in order
"""
from collections import OrderedDict
from ..core.module import Module
from ..core.tensor import Tensor


class Sequential(Module):
    """
    A sequential container. Modules are added in the order they are passed
    and each module is called in that order on the input.

    Example::

        model = lm.Sequential(
            lm.Linear(784, 256),
            lm.ReLU(),
            lm.Dropout(0.3),
            lm.Linear(256, 10),
        )

        # Or using an OrderedDict for named layers:
        model = lm.Sequential(OrderedDict([
            ('fc1', lm.Linear(784, 256)),
            ('act1', lm.ReLU()),
            ('fc2', lm.Linear(256, 10)),
        ]))
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self._modules[name] = module
                object.__setattr__(self, name, module)
        else:
            for idx, module in enumerate(args):
                self._modules[str(idx)] = module
                object.__setattr__(self, str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def append(self, module: Module):
        """Append a module to the end of the sequence."""
        idx = str(len(self._modules))
        self._modules[idx] = module
        object.__setattr__(self, idx, module)
        return self

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            mods = list(self._modules.values())[idx]
            return Sequential(*mods)
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def __repr__(self):
        lines = ["Sequential("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {repr(module)}")
        lines.append(")")
        return "\n".join(lines)
