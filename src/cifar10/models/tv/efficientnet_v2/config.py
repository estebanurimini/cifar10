"""EfficientNet-V2 (torchvision pretrained) training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class EfficientNetV2Config(TrainerConfig):
    """EfficientNet-V2 (torchvision pretrained) configuration."""
    variant: str = "s"
    image_size: int = 128
    source: str = "tv"
    architecture: str = "efficientnetv2"
    pretrained: bool = True
    pretrained_source: str = "imagenet1k"
    input_size: int = 128
    data_norm: str = "imagenet"
    lr: float = 0.0625
    weight_decay: float = 2e-5
    norm_weight_decay: float = 0.0
    backbone_lr_scale: float = 0.01
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0
    epochs: int = 300
    freeze_backbone_epochs: int = 16
    backbone_warmup_epochs: int = 5
    ema_decay: float = 0.9999
    run_dir: Path = field(default_factory=lambda: Path("./.runs/tv_efficientnetv2"))

    def _arch_params(self) -> dict[str, Any]:
        return {"variant": self.variant, "image_size": self.image_size}