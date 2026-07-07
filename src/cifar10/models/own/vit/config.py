"""ViT-specific training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cifar10.training.trainer import TrainerConfig


@dataclass
class ViTConfig(TrainerConfig):
    """ViT-specific configuration."""
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4
    dropout: float = 0.1
    source: str = "own"
    architecture: str = "vit"
    input_size: int = 32
    data_norm: str = "cifar10"
    run_dir: Path = field(default_factory=lambda: Path("./.runs/own_vit"))

    def _arch_params(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
        }