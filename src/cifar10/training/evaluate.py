"""Shared evaluation loop for any model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    """Evaluate a model on a dataloader, returning (avg_loss, accuracy_pct).

    Args:
        model: The model to evaluate.
        loader: DataLoader providing (images, labels).
        device: Device to run on.
        criterion: Optional loss criterion. If None, uses
            ``nn.CrossEntropyLoss()``.

    Returns:
        Tuple of ``(average_loss, accuracy_percentage)``.
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        loss_sum += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, 100.0 * correct / total