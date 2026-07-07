"""ResNet trainer with SGD, warmup + cosine, and MixUp/CutMix."""

from __future__ import annotations

import torch
import torch.nn as nn

from cifar10.data import build_mixup_cutmix
from cifar10.training.trainer import BaseTrainer


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