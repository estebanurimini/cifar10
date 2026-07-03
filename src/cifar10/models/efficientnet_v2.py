"""EfficientNet-V2 wrapper with torchvision pretrained weights.

Uses ``torchvision.models.efficientnet_v2_s(weights='DEFAULT')`` and replaces
the classifier head for CIFAR10 (10 classes). Designed for transfer learning
with a staged fine-tuning strategy (freeze backbone → unfreeze gradually).

Supports S, M, and L variants via the ``variant`` argument.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)

# ---------------------------------------------------------------------------
# Registry: variant → (builder, weights, in_features)
# ---------------------------------------------------------------------------
_EFFICIENTNET_V2_REGISTRY: dict[str, tuple] = {
    "s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT, 1280),
    "m": (efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT, 1280),
    "l": (efficientnet_v2_l, EfficientNet_V2_L_Weights.DEFAULT, 1280),
}


def _find_norm_parameter_names(model: nn.Module) -> set[str]:
    """Collect full parameter names belonging to normalisation layers.

    Inspects the module hierarchy to identify BatchNorm parameters by type.
    EfficientNet-V2 uses BatchNorm2d extensively in its MBConv blocks.
    """
    norm_names: set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.LayerNorm, nn.GroupNorm)):
            for p_name in module.state_dict():
                full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                norm_names.add(full_name)
    return norm_names


class EfficientNetV2CIFAR10(nn.Module):
    """EfficientNet-V2 with a CIFAR10 classification head.

    Args:
        num_classes: Number of output classes (default 10).
        variant: Model variant — ``'s'``, ``'m'``, or ``'l'`` (default ``'s'``).

    .. note::

        The pretrained classifier has 1000 ImageNet classes. It is replaced
        with a single ``nn.Linear(in_features, num_classes)`` head.
    """

    def __init__(self, num_classes: int = 10, variant: str = "s") -> None:
        super().__init__()

        if variant not in _EFFICIENTNET_V2_REGISTRY:
            msg = f"Unknown variant '{variant}'. Choose from {list(_EFFICIENTNET_V2_REGISTRY)}."
            raise ValueError(msg)

        builder, weights, in_features = _EFFICIENTNET_V2_REGISTRY[variant]
        self.variant = variant

        # Load pretrained model
        self.backbone = builder(weights=weights)

        # Replace classifier: original is [Dropout(p=0.2), Linear(1280, 1000)]
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

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
        """Check if a parameter name belongs to a normalisation layer."""
        return name in self._norm_param_names

    def _get_param_groups(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
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
        backbone_params: list[nn.Parameter] = []
        backbone_norm_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        head_norm_params: list[nn.Parameter] = []

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

    def _get_backbone_param_groups(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        norm_weight_decay: float = 0.0,
        backbone_lr_scale: float = 0.1,
    ) -> list[dict]:
        """Return param groups for the backbone only (excludes classifier head).

        Used when unfreezing the backbone mid-training to add new groups
        to the existing optimizer without rebuilding it. The model's own
        knowledge of which parameters are norms and which belong to the
        classifier keeps this logic self-contained.

        Args:
            lr: Base learning rate (head LR). Backbone LR is ``lr * scale``.
            weight_decay: Weight decay for non-norm parameters.
            norm_weight_decay: Weight decay for norm parameters (typically 0.0).
            backbone_lr_scale: Multiplier for backbone LR relative to head LR.

        Returns:
            List of param groups suitable for ``torch.optim.Optimizer.add_param_group``.
        """
        backbone_params: list[nn.Parameter] = []
        backbone_norm_params: list[nn.Parameter] = []

        for name, param in self.backbone.named_parameters():
            if "classifier" in name or not param.requires_grad:
                continue
            if self._is_norm_param(name):
                backbone_norm_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay},
            {"params": backbone_norm_params, "lr": lr * backbone_lr_scale, "weight_decay": norm_weight_decay},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
