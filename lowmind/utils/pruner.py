"""
LowMind Weight Pruner — optimized for edge devices and sparsity
"""
import numpy as np
from ..core.tensor import Tensor


class Pruner:
    """
    Weight Pruning API to zero out low-magnitude weights and introduce sparsity.

    Supports:
    - Magnitude-based global or local pruning
    - Sparsity calculation
    - Pruning mask tracking and re-application
    """

    def __init__(self, model):
        self.model = model
        self.masks = {}  # parameter_name -> binary numpy mask

    def prune_module_weight(self, param_name, sparsity_ratio):
        """
        Prune a specific weight parameter of the model by a certain ratio (0.0 to 1.0).
        """
        # Find parameter
        target_param = None
        for name, param in self.model.named_parameters():
            if name == param_name or name.endswith("." + param_name):
                target_param = param
                param_name = name
                break

        if target_param is None:
            raise ValueError(f"Parameter '{param_name}' not found in the model.")

        weights = target_param.data
        if sparsity_ratio <= 0.0:
            return
        if sparsity_ratio >= 1.0:
            target_param.data.fill(0.0)
            self.masks[param_name] = np.zeros_like(weights)
            return

        # Magnitude-based thresholding
        abs_weights = np.abs(weights)
        threshold = np.percentile(abs_weights, sparsity_ratio * 100)

        # Create and apply mask
        mask = (abs_weights >= threshold).astype(np.float32)
        target_param.data *= mask
        self.masks[param_name] = mask

    def prune_model(self, sparsity_ratio, param_names=None):
        """
        Prune multiple weight parameters (or all trainable parameters by default)
        using magnitude-based pruning.
        """
        for name, param in self.model.named_parameters():
            # Prune only weight matrices, skipping biases by default unless specified
            if param_names is not None:
                if name not in param_names and not any(name.endswith("." + p) for p in param_names):
                    continue
            elif "bias" in name:
                continue

            self.prune_module_weight(name, sparsity_ratio)

    def apply_masks(self):
        """
        Re-apply stored pruning masks to weights. Useful after a gradient step/training iteration.
        """
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
                if param.grad is not None:
                    param.grad *= self.masks[name]

    def calculate_sparsity(self):
        """
        Calculate total sparsity of the model weights as a percentage of zero elements.
        """
        total_elements = 0
        total_zeros = 0

        for name, param in self.model.named_parameters():
            total_elements += param.data.size
            total_zeros += np.sum(param.data == 0.0)

        if total_elements == 0:
            return 0.0

        return (total_zeros / total_elements) * 100.0
