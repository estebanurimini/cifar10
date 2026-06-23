"""Training script for DeiT on CIFAR10 with knowledge distillation.

Requires a pre-trained WideResNet teacher checkpoint at the configured path.
First run ``python -m cifar10.scripts.train_wrn`` to generate the teacher.

Usage:
    python -m cifar10.scripts.train_deit
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast

from cifar10.data import build_cifar10_loaders, build_mixup_cutmix
from cifar10.models import DeiT, WideResNet
from cifar10.training import BaseTrainer
from cifar10.training.trainer import TrainerConfig
from cifar10.utils import set_seed, get_device, load_checkpoint


@dataclass
class DeiTConfig(TrainerConfig):
    """DeiT-specific configuration."""
    # student model
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4
    dropout: float = 0.1
    drop_path_rate: float = 0.1

    # teacher model (WRN)
    wrn_depth: int = 16
    wrn_width: int = 4
    wrn_dropout: float = 0.0

    # distillation
    teacher_reliance: float = 0.7
    teacher_ckpt: Path = field(
        default_factory=lambda: Path("./.runs/wrn_cifar10/checkpoints/best.pt")
    )

    # paths
    run_dir: Path = field(default_factory=lambda: Path("./.runs/deit_cifar10"))


class DistillationTrainer(BaseTrainer):
    """Trainer with hard distillation loss from a teacher model."""

    def __init__(self, model, config, device, teacher):
        super().__init__(model, config, device)
        self.teacher = teacher
        self.mixup_cutmix = build_mixup_cutmix(
            num_classes=10,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
        )

    def _augment_batch(self, images, labels):
        return self.mixup_cutmix(images, labels)

    def _compute_loss(self, images, labels):
        # Teacher forward pass (no grad)
        with torch.no_grad():
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                teacher_logits = self.teacher(images)
                teacher_labels = teacher_logits.argmax(dim=-1)

        # Student forward pass
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            cls_logits, dist_logits = self.model(images)

            base_loss = self.criterion(cls_logits, labels)
            dist_loss = F.cross_entropy(dist_logits, teacher_labels)

            cfg = self.config
            return (1 - cfg.teacher_reliance) * base_loss + cfg.teacher_reliance * dist_loss


def build_teacher(config: DeiTConfig) -> WideResNet:
    return WideResNet(
        depth=config.wrn_depth,
        widen_factor=config.wrn_width,
        dropout_rate=config.wrn_dropout,
        num_classes=10,
    )


def main():
    cfg = DeiTConfig()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, _ = build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    # Load teacher
    teacher = build_teacher(cfg).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    if cfg.teacher_ckpt.exists():
        print(f"Loading teacher weights from {cfg.teacher_ckpt}...")
        ckpt = load_checkpoint(cfg.teacher_ckpt, device)
        if "ema" in ckpt:
            teacher.load_state_dict(ckpt["ema"])
        else:
            teacher.load_state_dict(ckpt["model"])
    else:
        print(
            f"Warning: Teacher checkpoint not found at {cfg.teacher_ckpt}. "
            "Teacher will have random weights. Run train_wrn.py first!"
        )

    # Build student
    model = DeiT(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_classes=10,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        drop_path_rate=cfg.drop_path_rate,
    ).to(device)

    trainer = DistillationTrainer(model, cfg, device, teacher)
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()