"""Abstract base trainer using the Template Method pattern."""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
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
from cifar10.utils.checkpoint import save_checkpoint, load_checkpoint
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
    use_randaugment: bool = True
    dropout: float = 0.0
    stochastic_depth: float = 0.0
    momentum: float = 0.9
    run_dir: Path = field(default_factory=lambda: Path("./.runs"))

    # optimizer type
    optimizer: str = "adamw"  # "adamw" or "sgd"

    # gradient clipping (0.0 = disabled)
    clip_grad_norm: float = 0.0

    # staged fine-tuning
    freeze_backbone_epochs: int = 0
    backbone_lr_scale: float = 1.0
    norm_weight_decay: float = 0.0  # no WD on norm layers by default

    # ------------------------------------------------------------------
    # Serialization helpers for resolve_run_dir()
    # ------------------------------------------------------------------

    def to_run_params(self) -> dict[str, Any]:
        """Convert config to a dict suitable for :func:`resolve_run_dir`.

        Subclasses can override :meth:`_arch_params` and/or
        :meth:`_teacher_params` to add model-specific keys.
        """
        return {
            "source": self.source,
            "architecture": self.architecture,
            "pretrained": self.pretrained,
            "pretrained_source": self.pretrained_source,
            "input_size": self.input_size,
            "data_norm": self.data_norm,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "label_smoothing": self.label_smoothing,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr,
            "use_randaugment": self.use_randaugment,
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
            "ema_decay": self.ema_decay,
            "dropout": self.dropout,
            "stochastic_depth": self.stochastic_depth,
            "freeze_backbone_epochs": self.freeze_backbone_epochs,
            "backbone_lr_scale": self.backbone_lr_scale,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "arch_params": self._arch_params(),
            "teacher": self._teacher_params(),
        }

    def _arch_params(self) -> dict[str, Any]:
        """Override in subclasses to add model-specific architecture params."""
        return {}

    def _teacher_params(self) -> dict[str, Any] | None:
        """Override in subclasses (e.g. DeiT) to record teacher info."""
        return None


def _config_to_params_dict(config: TrainerConfig) -> dict[str, Any]:
    """Convert a TrainerConfig (or subclass) to a serializable dict.

    Converts Path objects to strings and adds metadata fields.
    """
    d = dataclasses.asdict(config)

    # Convert Path objects to strings
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)

    # Add metadata
    now = datetime.now()
    d["created_at"] = now.isoformat(timespec="seconds")
    d["notes"] = ""

    return d


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

        # Write params.json (on first start, not on resume)
        self._write_params_json()

    def _write_params_json(self) -> None:
        """Write ``params.json`` to the run directory if it doesn't exist.

        If the file already exists (e.g., from a previous run or migration),
        it is left untouched to preserve the original configuration.
        """
        params_path = self.config.run_dir / "params.json"
        if params_path.exists():
            return

        params = _config_to_params_dict(self.config)
        params_path.write_text(json.dumps(params, indent=4, default=str))
        print(f"[params] Written: {params_path}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer based on ``self.config.optimizer``.

        Subclasses that need custom param groups (e.g. separate backbone/head
        LRs, no weight decay on norm layers) should override this method.
        """
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
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
        resume_from: Path | None = None,
    ) -> float:
        """Run the full training loop.

        Args:
            train_loader: Training data.
            val_loader: Validation / test data.
            resume_from: Optional path to a checkpoint to resume from.

        Returns:
            Best validation accuracy achieved (percentage).
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Handle resume
        start_epoch = 1
        best_acc = 0.0
        if resume_from is not None and resume_from.exists():
            start_epoch, best_acc = self._resume(resume_from)

        csv_exists = self.csv_log.exists()

        with open(self.csv_log, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if not csv_exists:
                writer.writerow([
                    "epoch", "train_loss", "val_loss",
                    "val_acc", "best_acc", "lr",
                ])

            for epoch in range(start_epoch, self.config.epochs + 1):
                self._on_epoch_start(epoch)
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
                    save_checkpoint(self.best_ckpt, self._state_dict(epoch, best_acc))

                save_checkpoint(self.last_ckpt, self._state_dict(epoch, best_acc))

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

    def _on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch (before training).

        Override in subclasses that need staged unfreezing, backbone LR
        adjustment, or other per-epoch setup.
        """
        pass

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

        Supports gradient clipping (if ``config.clip_grad_norm > 0``).

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

            # Gradient clipping (unscale before clipping)
            if self.config.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm,
                )

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

    def _state_dict(
        self,
        epoch: int | None = None,
        best_acc: float | None = None,
    ) -> dict[str, Any]:
        state = {
            "model": self.model.state_dict(),
            "ema": self.ema.shadow,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": dataclasses.asdict(self.config),
            "model_name": self._get_model_name(),
        }
        if epoch is not None:
            state["epoch"] = epoch
        if best_acc is not None:
            state["best_acc"] = best_acc
        return state

    def _get_model_name(self) -> str:
        """Return a short identifier for the model architecture.

        Override in subclasses if the default class name isn't suitable.
        """
        return type(self.model).__name__

    def _resume(self, resume_from: Path | dict[str, Any]) -> int:
        """Restore training state from a checkpoint.

        Args:
            resume_from: Either a path to a checkpoint file, or a pre-loaded
                checkpoint dict.

        Returns:
            The next epoch to start training from (1-indexed).
        """
        ckpt = (
            resume_from
            if isinstance(resume_from, dict)
            else load_checkpoint(resume_from, self.device)
        )

        # Load model weights
        self.model.load_state_dict(ckpt["model"])

        # Recreate optimizer & scheduler (since build_* create new param refs)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Restore optimizer & scheduler state (load_state_dict overwrites data)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        # Restore EMA
        self.ema.shadow = ckpt["ema"]

        # Restore AMP scaler if available (backward compat)
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

        # Restore best_acc from the checkpoint metadata
        best_acc = ckpt.get("best_acc", 0.0)

        # Next epoch: last completed epoch + 1
        start_epoch = ckpt.get("epoch", 0) + 1

        print(f"Resumed from epoch {ckpt.get('epoch', 0)}. "
              f"Starting at epoch {start_epoch}.")

        return start_epoch, best_acc


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