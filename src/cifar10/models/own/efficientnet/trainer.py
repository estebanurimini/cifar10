"""EfficientNet-V2 (from scratch) trainer with AdamW."""

from __future__ import annotations

import torch
import torch.nn as nn

from cifar10.data import build_mixup_cutmix
from cifar10.training.trainer import BaseTrainer


class EfficientNetTrainer(BaseTrainer):
    """Trainer for OwnEfficientNetV2 (from scratch) with AdamW."""

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
        )
        return torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _augment_batch(self, images, labels):
        return self.mixup_cutmix(images, labels)

    def _compute_loss(self, images, labels):
        logits = self.model(images)
        return self.criterion(logits, labels)