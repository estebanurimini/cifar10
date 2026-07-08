"""WRN trainer — uses profile-based augmentation and warmup + cosine scheduler."""

from cifar10.training.trainer import StandardTrainer


class WRNTrainer(StandardTrainer):
    """WideResNet trainer.

    Batch-level augmentation (MixUp/CutMix) is driven by the selected
    augmentation profile (``config.augment``).  Curriculum transitions
    are handled by the base class.
    """


__all__ = ["WRNTrainer"]
