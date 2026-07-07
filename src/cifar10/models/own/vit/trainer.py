"""ViT trainer with MixUp/CutMix augmentation."""

from __future__ import annotations

from cifar10.data import build_mixup_cutmix
from cifar10.training.trainer import StandardTrainer


class ViTTrainer(StandardTrainer):
    """ViT trainer with MixUp/CutMix augmentation."""

    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.mixup_cutmix = build_mixup_cutmix(
            num_classes=10,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
        )

    def _augment_batch(self, images, labels):
        return self.mixup_cutmix(images, labels)