"""VGG-specific training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class VGGConfig(TrainerConfig):
    """VGG-specific configuration."""
    variant: str = "vgg16_bn"
    vgg_dropout: float = 0.5
    source: str = "own"
    architecture: str = "vgg"
    input_size: int = 32
    data_norm: str = "cifar10"
    lr: float = 0.05
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    min_lr: float = 0.0
    momentum: float = 0.9
    epochs: int = 200
    optimizer: str = "sgd"
    run_dir: Path = field(default_factory=lambda: Path("./.runs/own_vgg"))

    def to_run_params(self) -> dict[str, Any]:
        params = super().to_run_params()
        params["dropout"] = self.vgg_dropout
        return params

    def _arch_params(self) -> dict[str, Any]:
        return {"variant": self.variant}