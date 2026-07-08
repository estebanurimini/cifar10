"""Augmentation profiles — named presets for per-image and batch-level augmentation.

Each profile defines:
- ``per_image``: which per-image transform to use (preset name)
- ``cutout``: optional Cutout hole size override
- ``mixup_alpha``, ``cutmix_alpha``: MixUp/CutMix distribution parameters
- ``mixup_prob``, ``cutmix_prob``: MixUp/CutMix application probabilities

Usage::

    >>> profile = AUG_PROFILES["mid"]
    >>> profile.per_image
    'autoaugment'
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugProfile:
    """A named augmentation profile.

    Attributes:
        per_image: Per-image augmentation preset name (e.g. ``"crop_flip"``,
            ``"randaugment"``, ``"autoaugment"``, ``"autoaugment_cutout"``).
        cutout: Cutout hole size, or ``None`` to disable Cutout.
        mixup_alpha: Alpha parameter for MixUp.
        cutmix_alpha: Alpha parameter for CutMix.
        mixup_prob: Probability of applying MixUp to a batch.
        cutmix_prob: Probability of applying CutMix to a batch.
    """
    per_image: str = "randaugment"
    cutout: int | None = None
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.0
    cutmix_prob: float = 0.0


# ---------------------------------------------------------------------------
# Named profile registry
# ---------------------------------------------------------------------------

AUG_PROFILES: dict[str, AugProfile] = {
    # Lite: crop + flip only, no Cutout, no MixUp/CutMix
    "lite": AugProfile(
        per_image="crop_flip",
    ),
    # Mid: AutoAugment + crop + flip + MixUp/CutMix (always applied, 50% each)
    "mid": AugProfile(
        per_image="autoaugment",
        mixup_prob=0.5,
        cutmix_prob=0.5,
    ),
    # Strong: AutoAugment + Cutout(16) + MixUp(30%) + CutMix(30%)
    "strong": AugProfile(
        per_image="autoaugment_cutout",
        cutout=16,
        mixup_prob=0.3,
        cutmix_prob=0.3,
    ),
    # ResNet-style: RandAugment + MixUp/CutMix (50% each, always applied)
    "resnet": AugProfile(
        per_image="randaugment",
        mixup_prob=0.5,
        cutmix_prob=0.5,
    ),
}


def resolve_augment(
    augment: str,
    cutout_size: int | None = None,
) -> list[AugProfile]:
    """Resolve an ``augment`` config value into a list of profiles.

    A single profile key (no comma) returns a one-element list for
    fixed augmentation.  Comma-separated keys define a curriculum
    sequence of profiles.

    Args:
        augment: Profile key (e.g. ``"mid"``) or comma-separated
            curriculum (e.g. ``"lite,mid,strong"``).
        cutout_size: Optional override for Cutout hole size.  If set,
            overrides any ``cutout`` value in the resolved profiles.

    Returns:
        List of ``AugProfile`` objects, one per stage.

    Raises:
        ValueError: If any key is unknown.
    """
    keys = [k.strip() for k in augment.split(",")]
    profiles = []
    for key in keys:
        if key not in AUG_PROFILES:
            raise ValueError(
                f"Unknown augmentation profile: {key!r}. "
                f"Available: {', '.join(sorted(AUG_PROFILES))}"
            )
        profile = AUG_PROFILES[key]
        if cutout_size is not None:
            profile = AugProfile(
                per_image=profile.per_image,
                cutout=cutout_size,
                mixup_alpha=profile.mixup_alpha,
                cutmix_alpha=profile.cutmix_alpha,
                mixup_prob=profile.mixup_prob,
                cutmix_prob=profile.cutmix_prob,
            )
        profiles.append(profile)
    return profiles


def resolve_switch_epochs(
    num_stages: int,
    total_epochs: int,
    switch_epochs_str: str = "",
) -> list[int]:
    """Determine the epoch at which each stage transition occurs.

    If ``switch_epochs_str`` is non-empty, parse it as a comma-separated
    list of epoch numbers.  Otherwise split the total epochs proportionally.

    Args:
        num_stages: Number of stages (profiles).
        total_epochs: Total training epochs.
        switch_epochs_str: Optional comma-separated switch epochs.

    Returns:
        List of ``num_stages - 1`` epoch numbers (1-indexed) at which to
        transition to the next stage.
    """
    if num_stages <= 1:
        return []

    if switch_epochs_str:
        parts = [int(s.strip()) for s in switch_epochs_str.split(",")]
        if len(parts) != num_stages - 1:
            raise ValueError(
                f"Expected {num_stages - 1} switch epoch(s) for "
                f"{num_stages} stages, got {len(parts)}: {switch_epochs_str}"
            )
        return parts

    # Proportional split
    switch_epochs = []
    for stage_idx in range(1, num_stages):
        epoch = int(total_epochs * stage_idx / num_stages) + 1
        switch_epochs.append(epoch)
    return switch_epochs