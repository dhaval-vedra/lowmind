"""
LowMind Gradient Checkpointing — trade compute for massive memory savings on edge devices
"""
from ..core.tensor import Tensor
from ..core.no_grad import no_grad, enable_grad


def checkpoint(function, *args):
    """
    Run a function/layer with Gradient Checkpointing.

    During the forward pass, the function is evaluated with gradient tracking disabled,
    saving substantial memory by not storing intermediate activations.
    During the backward pass, the function is re-evaluated on the inputs with gradient tracking enabled
    to compute and propagate gradients, and the intermediate activations are immediately discarded.

    Args:
        function: The function/layer module to evaluate (e.g. a Sequential block or activation).
        args: Input Tensor(s) to the function.

    Returns:
        Tensor output.
    """
    # Convert all inputs to Tensors
    inputs = [a if isinstance(a, Tensor) else Tensor(a) for a in args]
    requires_grad = any(i.requires_grad for i in inputs)

    # 1. Forward pass under no_grad to save activation memory
    with no_grad():
        out_no_grad = function(*inputs)

    if not requires_grad:
        return out_no_grad

    # 2. Create the checkpointed output tensor
    out = Tensor(out_no_grad.data,
                 requires_grad=True,
                 _children=tuple(inputs),
                 _op='checkpoint')

    def _backward():
        if out.requires_grad:
            # 3. Re-evaluate in grad mode to build the micro-graph
            with enable_grad():
                # Make sure inputs have requires_grad set for recompute
                for i in inputs:
                    i._ensure_grad()

                out_recompute = function(*inputs)

            # 4. Backward pass through the micro-graph with the incoming gradient
            out_recompute.backward(out.grad)

            # 5. The micro-graph activations are discarded automatically when out_recompute goes out of scope

    out._backward = _backward
    return out
