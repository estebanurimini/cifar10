"""TimmResNet — ResNet wrapper using TIMM (PyTorch Image Models) pretrained weights.

Uses ``timm.create_model('resnet50', pretrained=True)`` and replaces the
classifier head for CIFAR10 (10 classes). Designed for transfer learning with
a staged fine-tuning strategy (freeze backbone → unfreeze gradually).

Usage:
    >>> model = TimmResNet(num_classes=10, model_name='resnet50')
    >>> model.to(device)

Available TIMM ResNet models:
    - 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimmResNet(nn.Module):
    """ResNet from TIMM with a CIFAR10 classification head (pretrained on ImageNet).

    Args:
        num_classes: Number of output classes (default 10).
        model_name: TIMM model name to use (default ``'resnet18'``).
        pretrained: Whether to load ImageNet-pretrained weights (default True).
        freeze_backbone: Whether to freeze backbone initially (default True).

    .. note::

        The pretrained classifier has 1000 ImageNet classes. It is replaced
        with a single ``nn.Linear(in_features, num_classes)`` head. Input images
        should be resized to at least 224×224 for ResNet models (TIMM default).
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        import timm

        # Load model from TIMM
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
        )

        # Determine in_features from the model
        in_features = self.backbone.num_features

        # New classification head
        self.head = nn.Linear(in_features, num_classes)

        # Build norm parameter set
        self._norm_param_names = self._find_norm_parameter_names()

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _find_norm_parameter_names(self) -> set[str]:
        """Collect full parameter names that belong to normalization layers."""
        norm_names: set[str] = set()
        for mod_name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                    nn.LayerNorm, nn.GroupNorm)):
                for p_name in module.state_dict():
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    norm_names.add(full_name)
        return norm_names

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters (head remains trainable)."""
        for param in self.backbone.parameters():
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
            if "head" in name:
                if self._is_norm_param(name):
                    head_norm_params.append(param)
                else:
                    head_params.append(param)
            else:
                if self._is_norm_param(name):
                    backbone_norm_params.append(param)
                else:
                    backbone_params.append(param)

        # Head parameters (the linear layer)
        for name, param in self.head.named_parameters():
            head_params.append(param)

        return [
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
            {"params": head_norm_params, "lr": lr, "weight_decay": norm_weight_decay},
            {"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay},
            {"params": backbone_norm_params, "lr": lr * backbone_lr_scale, "weight_decay": norm_weight_decay},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
