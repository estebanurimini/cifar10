"""TimmResNet trainer with staged fine-tuning and AdamW."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from cifar10.data import build_mixup_cutmix
from cifar10.training.trainer import BaseTrainer


class TimmResNetTrainer(BaseTrainer):
    """Trainer for TimmResNet with staged fine-tuning and AdamW."""

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

    def _on_epoch_start(self, epoch: int) -> None:
        cfg = self.config
        if epoch == cfg.freeze_backbone_epochs + 1:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}: Unfreezing backbone for full fine-tuning")
            print(f"{'=' * 60}")
            self.model._unfreeze_backbone()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            self.ema = type(self.ema)(self.model, decay=cfg.ema_decay)

    def _get_ema_model(self) -> nn.Module:
        ema_model = copy.deepcopy(self.model)
        self.ema.apply_to(ema_model)
        return ema_model.to(self.device)