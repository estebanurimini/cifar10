"""TVConvNeXt — ConvNeXt Tiny wrapper with torchvision pretrained weights.

Uses ``torchvision.models.convnext_tiny(weights='DEFAULT')`` and replaces the
classifier head for CIFAR10 (10 classes). Designed for transfer learning with
a staged fine-tuning strategy (freeze backbone → unfreeze gradually).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)


def _find_norm_parameter_names(model: nn.Module) -> set[str]:
    """Collect full parameter names that belong to normalization layers.

    Inspects the module hierarchy to identify LayerNorm/BatchNorm parameters
    by module type, not just by name string.
    """
    norm_names: set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for p_name in module.state_dict():
                full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                norm_names.add(full_name)
    return norm_names


class TVConvNeXt(nn.Module):
    """ConvNeXt Tiny with a CIFAR10 classification head (torchvision pretrained).

    Args:
        num_classes: Number of output classes (default 10).

    .. note::

        The pretrained classifier has 1000 ImageNet classes. It is replaced
        with a single ``nn.Linear(768, num_classes)`` head.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Load pretrained ConvNeXt Tiny
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        # Store original feature dimension
        in_features = self.backbone.classifier[2].in_features  # 768

        # Replace classifier: original is [LayerNorm(eps=1e-6), Flatten(1), Linear(768, 1000)]
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

        # Build norm parameter set for use by _get_param_groups
        self._norm_param_names = _find_norm_parameter_names(self.backbone)

        # Freeze all backbone parameters initially (will be unfrozen gradually)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all parameters of the backbone (not the head)."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _is_norm_param(self, name: str) -> bool:
        """Check if a parameter name belongs to a normalization layer."""
        return name in self._norm_param_names

    def _get_param_groups(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        norm_weight_decay: float = 0.0,
        backbone_lr_scale: float = 0.1,
    ) -> list[dict]:
        """Build parameter groups for AdamW with separate LR for backbone/head
        and no weight decay on normalization layers.

        This should be called after unfreezing for full fine-tuning.

        Args:
            lr: Learning rate for the classification head.
            weight_decay: Weight decay for non-norm parameters.
            norm_weight_decay: Weight decay for norm parameters (typically 0.0).
            backbone_lr_scale: Multiplier for backbone LR relative to head LR.

        Returns:
            List of param groups suitable for ``torch.optim.AdamW``.
        """
        backbone_params = []
        backbone_norm_params = []
        head_params = []
        head_norm_params = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                if self._is_norm_param(name):
                    head_norm_params.append(param)
                else:
                    head_params.append(param)
            else:
                if self._is_norm_param(name):
                    backbone_norm_params.append(param)
                else:
                    backbone_params.append(param)

        return [
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
            {"params": head_norm_params, "lr": lr, "weight_decay": norm_weight_decay},
            {"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay},
            {"params": backbone_norm_params, "lr": lr * backbone_lr_scale, "weight_decay": norm_weight_decay},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)