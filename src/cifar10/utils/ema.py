"""Exponential Moving Average (EMA) for model parameters."""

import torch
import torch.nn as nn


class EMA:
    """Maintains a shadow copy of model parameters averaged over time.

    During evaluation the EMA weights are loaded into the model for smoother,
    more accurate predictions.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow copy with the current model parameters."""
        for name, tensor in model.state_dict().items():
            if tensor.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(tensor.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name] = tensor.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        """Load the EMA shadow weights into the model."""
        model.load_state_dict(self.shadow)