"""Training script for EfficientNet-V2 on CIFAR10 with transfer learning.

Uses torchvision's ImageNet-pretrained EfficientNet-V2 (S/M/L), upscales
CIFAR10 images to 128×128 with bicubic interpolation, and fine-tunes with AdamW.

Recipe (based on TorchVision's official EfficientNet-V2 training recipe):
    - ImageNet preprocessing (bicubic upscale to image_size)
    - TrivialAugmentWide + RandomErasing
    - AdamW (lr=6.25e-2, weight_decay=2e-5, no WD on norm layers)
    - Cosine LR schedule with 5-epoch linear warmup
    - Label smoothing (0.1)
    - MixUp (α=0.2) + CutMix (α=1.0)
    - EMA (decay=0.9999)
    - Gradient clipping (max_norm=1.0)
    - Staged fine-tuning: 10 epochs head-only, then full fine-tune

Usage:
    python -m cifar10.scripts.train_efficientnet_v2
    python -m cifar10.scripts.train_efficientnet_v2 --resume .runs/efficientnet_v2/checkpoints/last.pt
    python -m cifar10.scripts.train_efficientnet_v2 --variant m
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from cifar10.data import build_cifar10_imagenet_loaders, build_mixup_cutmix
from cifar10.models import EfficientNetV2CIFAR10
from cifar10.training import BaseTrainer, evaluate
from cifar10.training.trainer import TrainerConfig
from cifar10.utils import set_seed, get_device


@dataclass
class EfficientNetV2Config(TrainerConfig):
    """EfficientNet-V2-specific configuration.

    Overrides defaults to match the TorchVision EfficientNet-V2 training recipe.
    The LR 0.0625 is scaled from the recipe (0.5 for batch 1024 across 8 GPUs)
    to single-GPU batch 128: 0.5 * (128 / 1024) = 0.0625.
    """
    # model
    variant: str = "s"
    image_size: int = 128

    # optimization (AdamW per recipe)
    lr: float = 0.0625
    weight_decay: float = 2e-5  # recipe: --weight-decay 0.00002
    norm_weight_decay: float = 0.0
    backbone_lr_scale: float = 0.01
    backbone_warmup_epochs: int = 5
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0

    # training
    epochs: int = 300
    freeze_backbone_epochs: int = 16  # head-only training phase

    # EMA
    ema_decay: float = 0.9999

    # paths
    run_dir: Path = field(default_factory=lambda: Path("./.runs/efficientnet_v2"))


class EfficientNetV2Trainer(BaseTrainer):
    """Trainer for EfficientNet-V2 with staged fine-tuning and AdamW."""

    def __init__(
        self,
        model: EfficientNetV2CIFAR10,
        config: EfficientNetV2Config,
        device: torch.device,
    ) -> None:
        super().__init__(model, config, device)

        # MixUp / CutMix augmentation
        self.mixup_cutmix = build_mixup_cutmix(
            num_classes=10,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with separate param groups for backbone and head.

        During the frozen phase (first N epochs), only head parameters have
        requires_grad=True, so the optimizer only sees those.
        After unfreezing, we rebuild the optimizer with all param groups.
        """
        cfg = self.config
        param_groups = self.model._get_param_groups(
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            norm_weight_decay=cfg.norm_weight_decay,
            backbone_lr_scale=cfg.backbone_lr_scale,
        )
        return torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _augment_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp/CutMix to the batch."""
        return self.mixup_cutmix(images, labels)

    def _compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(images)
        return self.criterion(logits, labels)

    def _adjust_backbone_lr(self, epoch: int) -> None:
        """Set backbone LRs: warmup ramp then follow scheduler decay ratio.

        The backbone param groups (indices 2+) are added to the optimizer after
        the scheduler was created, so the scheduler doesn't manage them. We
        manually set their LR each epoch by applying the same cosine decay
        ratio that the scheduler applies to the head groups.
        """
        cfg = self.config
        backbone_unfreeze_epoch = cfg.freeze_backbone_epochs + 1

        # Not yet unfrozen — nothing to do
        if epoch < backbone_unfreeze_epoch:
            return

        backbone_target_lr = cfg.lr * cfg.backbone_lr_scale
        warmup_end = backbone_unfreeze_epoch + cfg.backbone_warmup_epochs

        # Head group 0 has the correct LR set by the scheduler from the
        # previous epoch. Compute the cosine decay ratio relative to its
        # initial LR.
        head_lr = self.optimizer.param_groups[0]["lr"]
        # CosineLR is the second scheduler in SequentialLR
        head_init_lr = self.scheduler._schedulers[1].base_lrs[0]

        for group in self.optimizer.param_groups[2:]:
            if epoch < warmup_end:
                # Linear warmup from 0 → backbone_target_lr
                progress = (epoch - backbone_unfreeze_epoch) / cfg.backbone_warmup_epochs
                group["lr"] = backbone_target_lr * progress
            else:
                # Follow the same cosine decay ratio as the head
                group["lr"] = backbone_target_lr * (head_lr / head_init_lr)

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """Train for one epoch with gradient clipping and staged unfreezing.

        At ``freeze_backbone_epochs``, unfreeze the backbone and rebuild the
        optimizer so it includes all parameters with appropriate LR groups.
        """
        cfg = self.config

        # --- Staged unfreezing ------------------------------------------------
        if epoch == cfg.freeze_backbone_epochs + 1:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}: Unfreezing backbone for full fine-tuning")
            print(f"{'=' * 60}")
            self.model._unfreeze_backbone()
            # Add backbone param groups to the existing optimizer.
            # This preserves AdamW momentum/adaptive-LR state from the
            # head-only phase, preventing catastrophic forgetting.
            for group in self.model._get_backbone_param_groups(
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                norm_weight_decay=cfg.norm_weight_decay,
                backbone_lr_scale=cfg.backbone_lr_scale,
            ):
                self.optimizer.add_param_group(group)
            # EMA already tracks *all* model parameters (including frozen
            # backbone). Scheduler also continues from where it left off.
            # No rebuild needed for either.

        # --- Backbone LR adjustment -------------------------------------------
        self._adjust_backbone_lr(epoch)

        # --- Training loop (with gradient clipping) ---------------------------
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        from tqdm.auto import tqdm
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{cfg.epochs}",
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            images, labels = self._augment_batch(images, labels)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                loss = self._compute_loss(images, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale before clipping)
            if cfg.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.clip_grad_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update(self.model)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            avg_loss = running_loss / total_samples

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return running_loss / total_samples

    def _get_ema_model(self) -> nn.Module:
        """Return a deep-copied model with EMA weights applied."""
        ema_model = copy.deepcopy(self.model)
        self.ema.apply_to(ema_model)
        return ema_model.to(self.device)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train EfficientNet-V2 on CIFAR10 with transfer learning."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from "
             "(e.g., .runs/efficientnet_v2/checkpoints/last.pt).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="s",
        choices=["s", "m", "l"],
        help="EfficientNet-V2 variant (default: s).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input image size (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128).",
    )
    args = parser.parse_args()

    cfg = EfficientNetV2Config(
        variant=args.variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Variant: EfficientNet-V2-{args.variant.upper()}")
    print(f"Image size: {cfg.image_size}×{cfg.image_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs} (frozen for first {cfg.freeze_backbone_epochs})")

    # Build data loaders with ImageNet preprocessing
    # val_loader uses non-augmented transforms for validation during training
    # test_loader is reserved for final evaluation only
    train_loader, val_loader, test_loader = build_cifar10_imagenet_loaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    # Build model
    model = EfficientNetV2CIFAR10(num_classes=10, variant=args.variant).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = EfficientNetV2Trainer(model, cfg, device)
    best_acc = trainer.train(train_loader, val_loader, resume_from=args.resume)

    # Final evaluation on held-out test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()