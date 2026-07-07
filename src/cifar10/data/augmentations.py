"""MixUp / CutMix / Cutout augmentation utilities."""

import torch
from torchvision.transforms import v2


def build_mixup_cutmix(
    num_classes: int = 10,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
) -> v2.RandomChoice:
    """Build a transform that randomly applies MixUp or CutMix (equal probability).

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


def build_batch_mixup_cutmix(
    num_classes: int = 10,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.3,
    cutmix_prob: float = 0.3,
) -> v2.RandomChoice:
    """Build a batch-level transform with weighted MixUp / CutMix / identity.

    Applies MixUp with ``mixup_prob``, CutMix with ``cutmix_prob``, or does
    nothing with the remaining probability.

    Args:
        num_classes: Number of classes.
        mixup_alpha: Alpha parameter for MixUp.
        cutmix_alpha: Alpha parameter for CutMix.
        mixup_prob: Probability of applying MixUp.
        cutmix_prob: Probability of applying CutMix.

    Returns:
        A ``v2.RandomChoice`` transform with weighted probabilities.
    """
    identity_prob = 1.0 - mixup_prob - cutmix_prob
    return v2.RandomChoice(
        [
            v2.MixUp(num_classes=num_classes, alpha=mixup_alpha),
            v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha),
            v2.Identity(),
        ],
        p=[mixup_prob, cutmix_prob, identity_prob],
    )


class Cutout:
    """Randomly mask out square regions. DeVries & Taylor 2017.

    Operates on ``torch.Tensor`` images (C, H, W) in ``[0, 1]`` range.
    The masked region is set to zero.
    """

    def __init__(self, hole_size: int = 16) -> None:
        self.hole_size = hole_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        if self.hole_size >= h or self.hole_size >= w:
            return img

        y = torch.randint(0, h - self.hole_size + 1, (1,)).item()
        x = torch.randint(0, w - self.hole_size + 1, (1,)).item()
        img = img.clone()
        img[:, y : y + self.hole_size, x : x + self.hole_size] = 0
        return img