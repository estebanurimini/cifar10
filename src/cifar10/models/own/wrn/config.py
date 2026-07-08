"""WideResNet-specific training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class WRNConfig(TrainerConfig):
    """WideResNet-specific configuration."""
    depth: int = 16
    widen_factor: int = 4
    wrn_dropout: float = 0.0
    source: str = "own"
    architecture: str = "wrn"
    input_size: int = 32
    data_norm: str = "cifar10"
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = True
    ema_decay: float = 0.995
    epochs: int = 300
    optimizer: str = "sgd"
    augment: str = "mid"
    run_dir: Path = field(default_factory=lambda: Path("./.runs/own_wrn"))

    def _arch_params(self) -> dict[str, Any]:
        return {"depth": self.depth, "widen_factor": self.widen_factor}