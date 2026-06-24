"""Training script for VGG on CIFAR10.

Uses SGD with momentum (standard for VGG) and cosine annealing. Trains with
a validation split from the training set and RandAugment. Writes checkpoints
and logs to ``./.runs/vgg/``.

Usage:
    python -m cifar10.scripts.train_vgg
    python -m cifar10.scripts.train_vgg --variant vgg11_bn
    python -m cifar10.scripts.train_vgg --resume .runs/vgg/checkpoints/last.pt
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from cifar10.data import build_cifar10_loaders
from cifar10.models import VGG
from cifar10.training import BaseTrainer, evaluate
from cifar10.training.trainer import TrainerConfig
from cifar10.utils import set_seed, get_device


@dataclass
class VGGConfig(TrainerConfig):
    """VGG-specific configuration."""
    # model
    variant: str = "vgg16_bn"  # "vgg11_bn" or "vgg16_bn"
    vgg_dropout: float = 0.5

    # optimization (SGD with momentum is standard for VGG)
    lr: float = 0.05
    weight_decay: float = 5e-4
    warmup_epochs: int = 0  # no warmup
    min_lr: float = 0.0
    momentum: float = 0.9

    # training
    epochs: int = 200

    # paths
    run_dir: Path = field(default_factory=lambda: Path(".runs/vgg"))


class VGGTrainer(BaseTrainer):
    """VGG trainer using SGD with momentum and cosine annealing."""

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
        )

    def _compute_loss(self, images, labels):
        logits = self.model(images)
        return self.criterion(logits, labels)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train VGG on CIFAR10.")
    parser.add_argument(
        "--variant",
        type=str,
        default="vgg16_bn",
        choices=["vgg11_bn", "vgg16_bn"],
        help="VGG variant (default: vgg16_bn).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from "
             "(e.g., .runs/vgg/checkpoints/last.pt).",
    )
    args = parser.parse_args()

    cfg = VGGConfig(variant=args.variant)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # VGG uses a validation split from training data (like WRN)
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        with_validation_split=True,
        use_randaugment=True,
    )

    model = VGG(
        variant=cfg.variant,
        num_classes=10,
        dropout=cfg.vgg_dropout,
    ).to(device)

    print(f"VGG variant: {cfg.variant}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.3f}M")

    trainer = VGGTrainer(model, cfg, device)
    best_acc = trainer.train(train_loader, val_loader, resume_from=args.resume)

    # Final evaluation on held-out test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()