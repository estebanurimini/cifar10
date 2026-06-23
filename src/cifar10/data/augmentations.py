"""MixUp / CutMix augmentation builder."""

from torchvision.transforms import v2


def build_mixup_cutmix(
    num_classes: int = 10,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
) -> v2.RandomChoice:
    """Build a transform that randomly applies MixUp or CutMix.

    Args:
        num_classes: Number of classes.
        mixup_alpha: Alpha parameter for MixUp.
        cutmix_alpha: Alpha parameter for CutMix.

    Returns:
        A ``v2.RandomChoice`` transform that applies either MixUp or CutMix.
    """
    return v2.RandomChoice([
        v2.MixUp(num_classes=num_classes, alpha=mixup_alpha),
        v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha),
    ])