"""ResNet-specific training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class ResNetConfig(TrainerConfig):
    """ResNet-specific configuration."""
    variant: str = "resnet56"
    source: str = "own"
    architecture: str = "resnet"
    input_size: int = 32
    data_norm: str = "cifar10"
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    min_lr: float = 0.0
    momentum: float = 0.9
    epochs: int = 200
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    optimizer: str = "sgd"
    run_dir: Path = field(default_factory=lambda: Path("./.runs/own_resnet"))

    def _arch_params(self) -> dict[str, Any]:
        return {"variant": self.variant}