"""Training script for ConvNeXt Tiny on CIFAR10 with transfer learning.

Uses torchvision's ImageNet-pretrained ConvNeXt Tiny, upscales CIFAR10 images
to 128×128 with bicubic interpolation, and fine-tunes with AdamW.

Recipe (based on TorchVision's official ConvNeXt training recipe):
    - ImageNet preprocessing (bicubic upscale to 128×128)
    - TrivialAugmentWide + RandomErasing
    - AdamW (lr=1e-3, weight_decay=0.05, no WD on norm layers)
    - Cosine LR schedule with 5-epoch linear warmup
    - Label smoothing (0.1)
    - MixUp (α=0.2) + CutMix (α=1.0)
    - EMA (decay=0.9999)
    - Gradient clipping (max_norm=1.0)
    - Staged fine-tuning: 10 epochs head-only, then full fine-tune

Usage:
    python -m cifar10.scripts.train_convnext
    python -m cifar10.scripts.train_convnext --resume .runs/convnext/checkpoints/last.pt
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
from cifar10.models import ConvNeXtCIFAR10
from cifar10.training import BaseTrainer, evaluate
from cifar10.training.trainer import TrainerConfig
from cifar10.utils import set_seed, get_device


@dataclass
class ConvNextConfig(TrainerConfig):
    """ConvNeXt-specific configuration.

    Overrides defaults to match the TorchVision ConvNeXt training recipe.
    """
    # model
    image_size: int = 128

    # optimization (AdamW per recipe)
    lr: float = 1e-3
    weight_decay: float = 0.05
    norm_weight_decay: float = 0.0  # no WD on LayerNorm
    backbone_lr_scale: float = 0.1  # backbone LR = 0.1 × head LR
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0

    # training
    epochs: int = 300
    freeze_backbone_epochs: int = 10  # head-only training phase

    # EMA
    ema_decay: float = 0.9999

    # paths
    run_dir: Path = field(default_factory=lambda: Path("./.runs/convnext"))


class ConvNextTrainer(BaseTrainer):
    """Trainer for ConvNeXt with staged fine-tuning and AdamW."""

    def __init__(
        self,
        model: ConvNeXtCIFAR10,
        config: ConvNextConfig,
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
            # Rebuild optimizer with all param groups (backbone + head)
            self.optimizer = self._build_optimizer()
            # Rebuild scheduler for the new optimizer
            self.scheduler = self._build_scheduler()
            # Move EMA to track all parameters now
            self.ema = type(self.ema)(self.model, decay=cfg.ema_decay)

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
        description="Train ConvNeXt Tiny on CIFAR10 with transfer learning."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from "
             "(e.g., .runs/convnext/checkpoints/last.pt).",
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

    cfg = ConvNextConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Image size: {cfg.image_size}×{cfg.image_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs} (frozen for first {cfg.freeze_backbone_epochs})")

    # Build data loaders with ImageNet preprocessing
    train_loader, test_loader = build_cifar10_imagenet_loaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    # Build model
    model = ConvNeXtCIFAR10(num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = ConvNextTrainer(model, cfg, device)
    best_acc = trainer.train(train_loader, test_loader, resume_from=args.resume)

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()