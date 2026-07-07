"""WRN trainer — applies batch-level MixUp/CutMix augmentations when configured."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from cifar10.data.augmentations import build_batch_mixup_cutmix
from cifar10.training.trainer import StandardTrainer


class WRNTrainer(StandardTrainer):
    """WideResNet trainer with optional batch-level MixUp/CutMix augmentations.

    Applies MixUp/CutMix only when ``config.mixup_prob > 0`` or
    ``config.cutmix_prob > 0``.  When both are zero (the default for WRN),
    no batch-level augmentation is applied, matching the original WRN recipe.

    Uses a plain ``CosineAnnealingLR`` scheduler (no warmup), which is the
    standard schedule for WideResNet on CIFAR-10.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_augment = None
        if self.config.mixup_prob > 0 or self.config.cutmix_prob > 0:
            self._batch_augment = build_batch_mixup_cutmix(
                num_classes=10,
                mixup_alpha=self.config.mixup_alpha,
                cutmix_alpha=self.config.cutmix_alpha,
                mixup_prob=self.config.mixup_prob,
                cutmix_prob=self.config.cutmix_prob,
            )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Use plain cosine annealing (no warmup), standard for WRN."""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.min_lr,
        )

    def _augment_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp, CutMix, or identity to a collated batch.

        If ``_batch_augment`` is ``None`` (both probabilities are zero),
        the batch is returned unchanged.
        """
        if self._batch_augment is not None:
            return self._batch_augment(images, labels)  # type: ignore[return-value]
        return images, labels


__all__ = ["WRNTrainer"]