"""Training script for ViT on CIFAR10.

Usage:
    python -m cifar10.scripts.train_vit
"""

from dataclasses import dataclass, field
from pathlib import Path

from cifar10.data import build_cifar10_loaders, build_mixup_cutmix
from cifar10.models import ViT
from cifar10.training import StandardTrainer
from cifar10.training.trainer import TrainerConfig
from cifar10.utils import set_seed, get_device


@dataclass
class ViTConfig(TrainerConfig):
    """ViT-specific configuration."""
    # model
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4
    dropout: float = 0.1
    # paths
    run_dir: Path = field(default_factory=lambda: Path("./.runs/vit_cifar10"))


class ViTTrainer(StandardTrainer):
    """ViT trainer with MixUp/CutMix augmentation."""

    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.mixup_cutmix = build_mixup_cutmix(
            num_classes=10,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
        )

    def _augment_batch(self, images, labels):
        return self.mixup_cutmix(images, labels)


def main():
    cfg = ViTConfig()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, _ = build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    model = ViT(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_classes=10,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(device)

    trainer = ViTTrainer(model, cfg, device)
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()