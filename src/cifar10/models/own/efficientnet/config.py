"""EfficientNet-V2-CIFAR (from scratch) training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class EfficientNetConfig(TrainerConfig):
    """EfficientNet-V2-CIFAR (from scratch) configuration."""
    source: str = "own"
    architecture: str = "efficientnetv2"
    input_size: int = 32
    data_norm: str = "cifar10"
    lr: float = 3e-3
    weight_decay: float = 5e-2
    norm_weight_decay: float = 0.0
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0
    epochs: int = 300
    ema_decay: float = 0.9999
    stochastic_depth_prob: float = 0.1
    run_dir: Path = field(default_factory=lambda: Path("./.runs/own_efficientnetv2"))

    def to_run_params(self) -> dict[str, Any]:
        params = super().to_run_params()
        params["dropout"] = 0.2  # match original script
        return params

    def _arch_params(self) -> dict[str, Any]:
        return {"stochastic_depth_prob": self.stochastic_depth_prob}