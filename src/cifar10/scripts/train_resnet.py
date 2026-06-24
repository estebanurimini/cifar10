"""Training script for ResNet on CIFAR10.

Uses SGD with momentum (standard for ResNet) and warmup + cosine annealing.
Trains with a validation split from the training set, RandAugment, and
MixUp/CutMix augmentation. Writes checkpoints and logs to ``./.runs/resnet/``.

Usage:
    python -m cifar10.scripts.train_resnet
    python -m cifar10.scripts.train_resnet --variant resnet56
    python -m cifar10.scripts.train_resnet --resume .runs/resnet/checkpoints/last.pt
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from cifar10.data import build_cifar10_loaders, build_mixup_cutmix
# from cifar10.models import ResNetCIFAR
from cifar10.training import BaseTrainer, evaluate
from cifar10.training.trainer import TrainerConfig
from cifar10.utils import set_seed, get_device


@dataclass
class ResNetConfig(TrainerConfig):
    """ResNet-specific configuration."""
    # model
    variant: str = "resnet20"  # "resnet20" or "resnet56"

    # optimization (SGD with momentum is standard for ResNet)
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    min_lr: float = 0.0
    momentum: float = 0.9

    # training
    epochs: int = 200

    # augmentation (MixUp/CutMix helps ResNet on CIFAR10)
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    # paths
    run_dir: Path = field(default_factory=lambda: Path("./.runs/resnet"))


class ResNetTrainer(BaseTrainer):
    """ResNet trainer using SGD with momentum, warmup + cosine, and MixUp/CutMix."""

    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.mixup_cutmix = build_mixup_cutmix(
            num_classes=10,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.SequentialLR:
        from cifar10.training.scheduler import build_scheduler
        return build_scheduler(
            self.optimizer,
            epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            min_lr=self.config.min_lr,
        )

    def _augment_batch(self, images, labels):
        return self.mixup_cutmix(images, labels)

    def _compute_loss(self, images, labels):
        logits = self.model(images)
        return self.criterion(logits, labels)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR10.")
    parser.add_argument(
        "--variant",
        type=str,
        default="resnet56",
        choices=["resnet20", "resnet56"],
        help="ResNet variant (default: resnet56).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from "
             "(e.g., .runs/resnet/checkpoints/last.pt).",
    )
    args = parser.parse_args()

    cfg = ResNetConfig(variant=args.variant)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ResNet uses a validation split from training data
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        with_validation_split=True,
        use_randaugment=True,
    )

    model = ResNetCIFAR(
        variant=cfg.variant,
        num_classes=10,
    ).to(device)

    print(f"ResNet variant: {cfg.variant}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.3f}M")

    trainer = ResNetTrainer(model, cfg, device)
    best_acc = trainer.train(train_loader, val_loader, resume_from=args.resume)

    # Final evaluation on held-out test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()