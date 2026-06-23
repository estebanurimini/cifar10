"""Training script for WideResNet on CIFAR10.

Uses SGD optimizer (as is standard for WRN) and a validation split from the
training set. Writes checkpoints and logs to ``./.runs/wrn/``.

Usage:
    python -m cifar10.scripts.train_wrn
    python -m cifar10.scripts.train_wrn --resume .runs/wrn/checkpoints/last.pt
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from cifar10.data import build_cifar10_loaders
from cifar10.models import WideResNet
from cifar10.training import BaseTrainer, evaluate
from cifar10.training.trainer import TrainerConfig
from cifar10.utils import set_seed, get_device


@dataclass
class WRNConfig(TrainerConfig):
    """WideResNet-specific configuration."""
    # model
    depth: int = 28
    widen_factor: int = 10
    wrn_dropout: float = 0.0

    # optimization (SGD is standard for WRN)
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epochs: int = 0  # no warmup for WRN
    min_lr: float = 0.0
    momentum: float = 0.9

    # training
    epochs: int = 200

    # paths
    run_dir: Path = field(default_factory=lambda: Path("./.runs/wrn"))


class WRNTrainer(BaseTrainer):
    """WideResNet trainer using SGD with momentum and cosine annealing."""

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

    parser = argparse.ArgumentParser(description="Train WideResNet on CIFAR10.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from "
             "(e.g., .runs/wrn/checkpoints/last.pt).",
    )
    args = parser.parse_args()

    cfg = WRNConfig()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # WRN uses a validation split from training data
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        with_validation_split=True,
        use_randaugment=True,
    )

    model = WideResNet(
        depth=cfg.depth,
        widen_factor=cfg.widen_factor,
        dropout_rate=cfg.wrn_dropout,
        num_classes=10,
    ).to(device)

    trainer = WRNTrainer(model, cfg, device)
    best_acc = trainer.train(train_loader, val_loader, resume_from=args.resume)

    # Final evaluation on held-out test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
