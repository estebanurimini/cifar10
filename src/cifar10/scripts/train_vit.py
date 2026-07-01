"""Training script for ViT on CIFAR10.

Uses a validation split from the training set (like ResNet/WRN/VGG). Trains with
MixUp/CutMix augmentation and reports final held-out test accuracy.

Usage:
    python -m cifar10.scripts.train_vit
    python -m cifar10.scripts.train_vit --resume .runs/vit/checkpoints/last.pt
"""

from dataclasses import dataclass, field
from pathlib import Path

from cifar10.data import build_cifar10_loaders, build_mixup_cutmix
from cifar10.models import ViT
from cifar10.training import StandardTrainer, evaluate
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
    run_dir: Path = field(default_factory=lambda: Path("./.runs/vit"))


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
    import argparse

    parser = argparse.ArgumentParser(description="Train ViT on CIFAR10.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from "
             "(e.g., .runs/vit/checkpoints/last.pt).",
    )
    args = parser.parse_args()

    cfg = ViTConfig()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ViT uses a validation split from training data (like WRN/VGG)
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        with_validation_split=True,
        use_randaugment=True,
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
    best_acc = trainer.train(train_loader, val_loader, resume_from=args.resume)

    # Final evaluation on held-out test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()