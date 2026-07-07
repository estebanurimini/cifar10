"""TimmResNet-specific training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class TimmResNetConfig(TrainerConfig):
    """TimmResNet-specific configuration."""
    model_name: str = "resnet18"
    image_size: int = 224
    source: str = "timm"
    architecture: str = "resnet"
    pretrained: bool = True
    pretrained_source: str = "imagenet1k"
    input_size: int = 224
    data_norm: str = "imagenet"
    lr: float = 1e-3
    weight_decay: float = 0.05
    norm_weight_decay: float = 0.0
    backbone_lr_scale: float = 0.1
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0
    epochs: int = 300
    freeze_backbone_epochs: int = 10
    ema_decay: float = 0.9999
    run_dir: Path = field(default_factory=lambda: Path("./.runs/timm_resnet"))

    def _arch_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "image_size": self.image_size}