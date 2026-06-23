"""Scheduler factory — builds warmup + cosine schedulers."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


def build_scheduler(
    optimizer: Optimizer,
    epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-5,
    start_factor: float = 0.01,
) -> SequentialLR:
    """Build a warmup + cosine annealing scheduler.

    The learning rate linearly increases from ``start_factor * base_lr`` to
    ``base_lr`` over ``warmup_epochs``, then decays via cosine annealing down
    to ``min_lr`` for the remaining epochs.

    Args:
        optimizer: The optimizer to schedule.
        epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate after cosine decay.
        start_factor: Initial learning rate factor (relative to base_lr).

    Returns:
        A ``SequentialLR`` scheduler.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )