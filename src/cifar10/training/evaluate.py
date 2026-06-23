"""Shared evaluation loop for any model."""

import time

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


@torch.no_grad()
def detailed_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    num_classes: int = 10,
) -> dict[str, float | list[float]]:
    """Evaluate a model with detailed per-class metrics.

    Args:
        model: The model to evaluate.
        loader: DataLoader providing (images, labels).
        device: Device to run on.
        criterion: Optional loss criterion. If None, uses
            ``nn.CrossEntropyLoss()``.
        num_classes: Number of output classes.

    Returns:
        A dict with keys:
            - ``loss``: average loss
            - ``acc``: overall accuracy percentage
            - ``acc_per_class``: list of per-class accuracy (length num_classes)
            - ``throughput``: images processed per second
            - ``num_samples``: total number of samples evaluated
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        loss_sum += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Per-class counts
        for label, pred in zip(labels.cpu(), preds.cpu()):
            class_total[label.item()] += 1
            if label == pred:
                class_correct[label.item()] += 1

    elapsed = time.perf_counter() - start_time
    throughput = total / elapsed if elapsed > 0 else 0.0

    acc_per_class = [
        100.0 * class_correct[c] / class_total[c]
        if class_total[c] > 0 else 0.0
        for c in range(num_classes)
    ]

    return {
        "loss": loss_sum / total,
        "acc": 100.0 * correct / total,
        "acc_per_class": acc_per_class,
        "throughput": throughput,
        "num_samples": total,
    }


def format_evaluation_results(
    metrics: dict[str, float | list[float]],
    class_names: list[str] | None = None,
) -> str:
    """Format evaluation results as a human-readable string.

    Args:
        metrics: The dict returned by :func:`detailed_evaluate`.
        class_names: Optional list of class names (length must match
            ``acc_per_class``). If not provided, uses numeric labels.

    Returns:
        A formatted string suitable for printing.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"  Loss:        {metrics['loss']:.4f}")
    lines.append(f"  Accuracy:    {metrics['acc']:.2f}%")
    lines.append(f"  Throughput:  {metrics['throughput']:.0f} img/s")
    lines.append(f"  Samples:     {metrics['num_samples']}")
    lines.append("-" * 60)
    lines.append("  Per-Class Accuracy:")
    acc_per_class = metrics["acc_per_class"]
    for i, acc in enumerate(acc_per_class):
        label = class_names[i] if class_names else f"Class {i}"
        lines.append(f"    {label:>10}: {acc:6.2f}%")
    lines.append("=" * 60)
    return "\n".join(lines)
