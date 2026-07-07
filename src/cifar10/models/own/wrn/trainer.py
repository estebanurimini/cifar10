"""WRN trainer — applies batch-level MixUp/CutMix augmentations."""

from __future__ import annotations

import torch

from cifar10.data.augmentations import build_batch_mixup_cutmix
from cifar10.training.trainer import StandardTrainer


class WRNTrainer(StandardTrainer):
    """WideResNet trainer with batch-level MixUp/CutMix augmentations.

    Applies MixUp (30%), CutMix (30%), or nothing (40%) on each batch
    *after* per-sample spatial augmentations (AutoAugment, Cutout, etc.).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_augment = build_batch_mixup_cutmix(
            num_classes=10,
            mixup_alpha=self.config.mixup_alpha,
            cutmix_alpha=self.config.cutmix_alpha,
            mixup_prob=0.3,
            cutmix_prob=0.3,
        )

    def _augment_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp, CutMix, or identity to a collated batch."""
        return self.batch_augment(images, labels)  # type: ignore[return-value]


__all__ = ["WRNTrainer"]
