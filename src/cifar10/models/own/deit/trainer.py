"""Distillation trainer with hard distillation loss from a teacher model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from cifar10.data import build_mixup_cutmix
from cifar10.training.trainer import BaseTrainer


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