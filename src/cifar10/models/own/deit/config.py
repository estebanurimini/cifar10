"""DeiT-specific training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class DeiTConfig(TrainerConfig):
    """DeiT-specific configuration."""
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    source: str = "own"
    architecture: str = "deit"
    input_size: int = 32
    data_norm: str = "cifar10"
    # teacher model (WRN)
    wrn_depth: int = 16
    wrn_width: int = 4
    wrn_dropout: float = 0.0
    # distillation
    teacher_reliance: float = 0.7
    teacher_ckpt: Path = field(
        default_factory=lambda: Path("./.runs/own_wrn/checkpoints/best.pt")
    )
    run_dir: Path = field(default_factory=lambda: Path("./.runs/own_deit"))

    def _arch_params(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "drop_path_rate": self.drop_path_rate,
            "teacher_reliance": self.teacher_reliance,
            "wrn_depth": self.wrn_depth,
            "wrn_width": self.wrn_width,
        }

    def _teacher_params(self) -> dict[str, Any] | None:
        return {
            "source_run": "own_wrn",
            "local_path": "teacher/teacher_best.pt",
            "original_path": str(self.teacher_ckpt),
        }