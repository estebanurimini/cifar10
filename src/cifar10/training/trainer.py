"""Abstract base trainer using the Template Method pattern."""

from __future__ import annotations

import copy
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cifar10.config import BaseConfig
from cifar10.training.evaluate import evaluate
from cifar10.training.scheduler import build_scheduler
from cifar10.utils.checkpoint import save_checkpoint
from cifar10.utils.ema import EMA


@dataclass
class TrainerConfig(BaseConfig):
    """Configuration that applies to all trainers.

    Overrides the base run_dir default so each trainer uses a sub-folder.
    Subclasses (e.g. ``ViTTrainerConfig``) add model-specific fields.
    """
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    run_dir: Path = field(default_factory=lambda: Path("./.runs/unnamed"))


class BaseTrainer(ABC):
    """Template method for a full training run.

    Subclasses override :meth:`_compute_loss` to implement different training
    objectives (standard CE, distillation, etc.).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.use_amp = device.type == "cuda"

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.ema = EMA(model, decay=config.ema_decay)

        # Paths
        self.checkpoint_dir = config.run_dir / "checkpoints"
        self.log_dir = config.run_dir / "logs"
        self.best_ckpt = self.checkpoint_dir / "best.pt"
        self.last_ckpt = self.checkpoint_dir / "last.pt"
        self.csv_log = self.log_dir / "metrics.csv"

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.SequentialLR:
        return build_scheduler(
            self.optimizer,
            epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            min_lr=self.config.min_lr,
        )

    @abstractmethod
    def _compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Subclasses define how the loss is computed for each batch."""
        ...

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> float:
        """Run the full training loop.

        Args:
            train_loader: Training data.
            val_loader: Validation / test data.

        Returns:
            Best validation accuracy achieved (percentage).
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        best_acc = 0.0
        csv_exists = self.csv_log.exists()

        with open(self.csv_log, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if not csv_exists:
                writer.writerow([
                    "epoch", "train_loss", "val_loss",
                    "val_acc", "best_acc", "lr",
                ])

            for epoch in range(1, self.config.epochs + 1):
                train_loss = self._train_epoch(train_loader, epoch)
                self.scheduler.step()

                val_loss, val_acc = evaluate(
                    self._get_ema_model(),
                    val_loader,
                    self.device,
                )
                best_acc = max(best_acc, val_acc)

                current_lr = self.optimizer.param_groups[0]["lr"]
                writer.writerow([
                    epoch, train_loss, val_loss,
                    val_acc, best_acc, current_lr,
                ])
                csv_file.flush()

                if val_acc >= best_acc:
                    save_checkpoint(self.best_ckpt, self._state_dict(epoch))

                save_checkpoint(self.last_ckpt, self._state_dict(epoch))

                tqdm.write(
                    f"[{epoch:03d}/{self.config.epochs}] "
                    f"train={train_loss:.4f} "
                    f"val={val_loss:.4f} "
                    f"acc={val_acc:.2f}% "
                    f"best={best_acc:.2f}%"
                )

        print(f"Best accuracy: {best_acc:.2f}%")
        return best_acc

    # ------------------------------------------------------------------
    # Hook / helper methods (can be overridden by subclasses)
    # ------------------------------------------------------------------

    def _augment_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentations like MixUp/CutMix to a batch.

        Override this in subclasses that need extra augmentation.
        """
        return images, labels

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data.
            epoch: Current epoch number (1-indexed).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{self.config.epochs}",
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            images, labels = self._augment_batch(images, labels)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                loss = self._compute_loss(images, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update(self.model)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            avg_loss = running_loss / total_samples

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return running_loss / total_samples

    def _get_ema_model(self) -> nn.Module:
        """Return a deep-copied model with EMA weights applied."""
        ema_model = copy.deepcopy(self.model)
        self.ema.apply_to(ema_model)
        return ema_model.to(self.device)

    def _state_dict(self, epoch: int | None = None) -> dict[str, Any]:
        state = {
            "model": self.model.state_dict(),
            "ema": self.ema.shadow,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if epoch is not None:
            state["epoch"] = epoch
        return state



class StandardTrainer(BaseTrainer):
    """Standard supervised training (cross-entropy loss).

    Used by ViT and WideResNet.
    """

    def _compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(images)
        return self.criterion(logits, labels)