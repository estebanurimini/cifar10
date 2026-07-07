"""EfficientNet-V2 (torchvision pretrained) trainer with staged fine-tuning and AdamW."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from cifar10.data import build_mixup_cutmix
from cifar10.training.trainer import BaseTrainer


class EfficientNetV2Trainer(BaseTrainer):
    """Trainer for EfficientNet-V2 with staged fine-tuning and AdamW."""

    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.mixup_cutmix = build_mixup_cutmix(
            num_classes=10,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        param_groups = self.model._get_param_groups(
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            norm_weight_decay=cfg.norm_weight_decay,
            backbone_lr_scale=cfg.backbone_lr_scale,
        )
        return torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _augment_batch(self, images, labels):
        return self.mixup_cutmix(images, labels)

    def _compute_loss(self, images, labels):
        logits = self.model(images)
        return self.criterion(logits, labels)

    def _adjust_backbone_lr(self, epoch: int) -> None:
        cfg = self.config
        backbone_unfreeze_epoch = cfg.freeze_backbone_epochs + 1
        if epoch < backbone_unfreeze_epoch:
            return
        backbone_target_lr = cfg.lr * cfg.backbone_lr_scale
        warmup_end = backbone_unfreeze_epoch + cfg.backbone_warmup_epochs
        head_lr = self.optimizer.param_groups[0]["lr"]
        head_init_lr = self.scheduler._schedulers[1].base_lrs[0]
        for group in self.optimizer.param_groups[2:]:
            if epoch < warmup_end:
                progress = (epoch - backbone_unfreeze_epoch) / cfg.backbone_warmup_epochs
                group["lr"] = backbone_target_lr * progress
            else:
                group["lr"] = backbone_target_lr * (head_lr / head_init_lr)

    def _on_epoch_start(self, epoch: int) -> None:
        cfg = self.config
        if epoch == cfg.freeze_backbone_epochs + 1:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}: Unfreezing backbone for full fine-tuning")
            print(f"{'=' * 60}")
            self.model._unfreeze_backbone()
            for group in self.model._get_backbone_param_groups(
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                norm_weight_decay=cfg.norm_weight_decay,
                backbone_lr_scale=cfg.backbone_lr_scale,
            ):
                self.optimizer.add_param_group(group)
        self._adjust_backbone_lr(epoch)

    def _get_ema_model(self) -> nn.Module:
        ema_model = copy.deepcopy(self.model)
        self.ema.apply_to(ema_model)
        return ema_model.to(self.device)