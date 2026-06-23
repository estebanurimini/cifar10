"""Checkpoint save/load helpers."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Save a training checkpoint to disk.

    Args:
        path: Destination file path.
        state: Dict containing model state_dict, optimizer, scheduler, epoch, etc.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load a training checkpoint from disk.

    Args:
        path: Source file path.
        device: Device to map tensors to.

    Returns:
        The checkpoint dict.
    """
    return torch.load(path, map_location=device, weights_only=False)