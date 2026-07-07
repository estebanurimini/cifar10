"""Unified model registry for CIFAR10 training.

Maps model identifiers (e.g. ``"own-resnet"``) to their config, model,
trainer, and data loader builder.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from cifar10.data import build_cifar10_loaders, build_cifar10_imagenet_loaders
from cifar10.models.own.vit import OwnViT, ViTConfig, ViTTrainer
from cifar10.models.own.deit import OwnDeiT, DeiTConfig, DistillationTrainer
from cifar10.models.own.resnet import OwnResNet, ResNetConfig, ResNetTrainer
from cifar10.models.own.wrn import OwnWRN, WRNConfig, WRNTrainer
from cifar10.models.own.vgg import OwnVGG, VGGConfig, VGGTrainer
from cifar10.models.own.efficientnet import OwnEfficientNetV2, EfficientNetConfig, EfficientNetTrainer
from cifar10.models.tv.convnext import TVConvNeXt, ConvNextConfig, ConvNextTrainer
from cifar10.models.tv.efficientnet_v2 import TVEfficientNetV2, EfficientNetV2Config, EfficientNetV2Trainer
from cifar10.models.timm.resnet import TimmResNet
from cifar10.models.timm.resnet.config import TimmResNetConfig
from cifar10.models.timm.resnet.trainer import TimmResNetTrainer
from cifar10.training.trainer import BaseTrainer, TrainerConfig
from cifar10.utils.checkpoint import load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Type alias for a model registry entry
# ---------------------------------------------------------------------------

ModelEntry = tuple[
    type[TrainerConfig],          # config class
    type[nn.Module],              # model class
    type[BaseTrainer],            # trainer class
    Callable,                     # data loader builder
    dict[str, Any],              # extra kwargs for data loader builder
    str,                         # human-readable description
]

# ---------------------------------------------------------------------------
# Data loader builder wrappers
# ---------------------------------------------------------------------------


def _build_cifar10_loaders(cfg: TrainerConfig, device: torch.device):
    return build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        with_validation_split=True,
        augmentation=cfg.augmentation,
    )


def _build_cifar10_loaders_no_val_split(cfg: TrainerConfig, device: torch.device):
    return build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        with_validation_split=False,
        augmentation=cfg.augmentation,
    )


def _build_imagenet_loaders(cfg: TrainerConfig, device: torch.device):
    return build_cifar10_imagenet_loaders(
        data_dir=cfg.data_dir,
        image_size=cfg.input_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )


# ---------------------------------------------------------------------------
# The unified model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelEntry] = {
    "own-resnet": (
        ResNetConfig,
        OwnResNet,
        ResNetTrainer,
        _build_cifar10_loaders,
        {},
        "ResNet (own impl.) — trained from scratch on CIFAR-10",
    ),
    "own-wrn": (
        WRNConfig,
        OwnWRN,
        WRNTrainer,
        _build_cifar10_loaders,
        {},
        "Wide ResNet (own impl.) — trained from scratch on CIFAR-10",
    ),
    "own-vgg": (
        VGGConfig,
        OwnVGG,
        VGGTrainer,
        _build_cifar10_loaders,
        {},
        "VGG (own impl.) — trained from scratch on CIFAR-10",
    ),
    "own-vit": (
        ViTConfig,
        OwnViT,
        ViTTrainer,
        _build_cifar10_loaders,
        {},
        "Vision Transformer (own impl.) — trained from scratch on CIFAR-10",
    ),
    "own-deit": (
        DeiTConfig,
        OwnDeiT,
        DistillationTrainer,
        _build_cifar10_loaders,
        {},
        "DeiT (own impl.) — distillation with WRN-16-4 teacher",
    ),
    "own-efficientnetv2": (
        EfficientNetConfig,
        OwnEfficientNetV2,
        EfficientNetTrainer,
        _build_cifar10_loaders_no_val_split,
        {},
        "EfficientNetV2 (own impl.) — trained from scratch on CIFAR-10",
    ),
    "tv-convnext": (
        ConvNextConfig,
        TVConvNeXt,
        ConvNextTrainer,
        _build_imagenet_loaders,
        {},
        "ConvNeXt (torchvision) — pretrained on ImageNet-1K",
    ),
    "tv-efficientnetv2": (
        EfficientNetV2Config,
        TVEfficientNetV2,
        EfficientNetV2Trainer,
        _build_imagenet_loaders,
        {},
        "EfficientNetV2 (torchvision) — pretrained on ImageNet-1K",
    ),
    "timm-resnet": (
        TimmResNetConfig,
        TimmResNet,
        TimmResNetTrainer,
        _build_imagenet_loaders,
        {},
        "ResNet (timm) — pretrained on ImageNet-1K, supports resnet18/34/50/101/152",
    ),
}

# ---------------------------------------------------------------------------
# Config inspection helpers
# ---------------------------------------------------------------------------

# Fields that can be overridden via CLI flags in train.py
CLI_OVERRIDABLE_FIELDS = {"variant", "model_name", "image_size", "batch_size", "lr", "epochs"}


def format_config_defaults(cfg: TrainerConfig) -> list[str]:
    """Format notable config defaults for display in ``--list-models``.

    Shows architecture-specific fields plus any CLI-overridable fields that
    differ from the base ``TrainerConfig`` defaults.
    """
    base = TrainerConfig()
    parts: list[str] = []

    # Architecture-specific fields (from _arch_params)
    arch_params = getattr(cfg, "_arch_params", lambda: {})()
    for key in sorted(arch_params):
        val = getattr(cfg, key, None)
        if val is not None:
            parts.append(f"{key}={val}")

    # Add input_size always (it's important for users to know)
    if "input_size" not in arch_params:
        parts.append(f"input_size={cfg.input_size}")

    return parts


def show_config(name: str) -> None:
    """Print all config fields with defaults for a given model."""
    if name not in MODEL_REGISTRY:
        print(f"Unknown model: {name}")
        print(f"Use --list-models to see available models.")
        return

    cfg_cls, model_cls, trainer_cls, _, _, description = MODEL_REGISTRY[name]
    cfg = cfg_cls()

    print(f"Model: {name}")
    print(f"Description: {description}")
    print(f"Config class: {cfg_cls.__name__}")
    print(f"Model class: {model_cls.__name__}")
    print(f"Trainer class: {trainer_cls.__name__}")
    print()
    print("Config fields:")
    for field in dataclasses.fields(cfg):
        val = getattr(cfg, field.name)
        # Skip private / internal fields
        if field.name.startswith("_"):
            continue
        marker = " (CLI-overridable)" if field.name in CLI_OVERRIDABLE_FIELDS else ""
        print(f"  {field.name}: {val!r}{marker}")
    print()
    print("CLI-overridable flags: --variant, --image-size, --batch-size, --lr, --epochs")
    print("These can be passed to 'python -m cifar10.scripts.train --model <name>'.")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_teacher(config: DeiTConfig) -> OwnWRN:
    """Build a WRN teacher model for DeiT distillation."""
    return OwnWRN(
        depth=config.wrn_depth,
        widen_factor=config.wrn_width,
        dropout_rate=config.wrn_dropout,
        num_classes=10,
    )


def _copy_teacher_to_run_dir(config: DeiTConfig) -> Path:
    """Copy teacher checkpoint into the run directory's ``teacher/`` subfolder."""
    teacher_dir = config.run_dir / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    local_path = teacher_dir / "teacher_best.pt"
    if not local_path.exists():
        if not config.teacher_ckpt.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found at {config.teacher_ckpt}. "
                f"Run 'python -m cifar10.scripts.train --model own-wrn' first."
            )
        print(f"Copying teacher weights from {config.teacher_ckpt} to {local_path}")
        ckpt = load_checkpoint(config.teacher_ckpt, torch.device("cpu"))
        save_checkpoint(local_path, ckpt)
    return local_path


def build_model(model_cls: type[nn.Module], config: TrainerConfig) -> nn.Module:
    """Build a model from its class and config, handling different signatures."""
    model_name = f"{config.source}-{config.architecture}"

    if model_name == "own-resnet":
        return model_cls(variant=config.variant, num_classes=10)  # type: ignore[arg-type]
    elif model_name == "own-wrn":
        return model_cls(
            depth=config.depth,
            widen_factor=config.widen_factor,
            dropout_rate=config.wrn_dropout,
            num_classes=10,
        )
    elif model_name == "own-vgg":
        return model_cls(variant=config.variant, num_classes=10, dropout=config.vgg_dropout)
    elif model_name == "own-vit":
        return model_cls(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=10,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
    elif model_name == "own-deit":
        return model_cls(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=10,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            drop_path_rate=config.drop_path_rate,
        )
    elif model_name == "own-efficientnetv2":
        return model_cls(
            num_classes=10,
            dropout_rate=0.2,
            stochastic_depth_prob=config.stochastic_depth_prob,
        )
    elif model_name == "tv-convnext":
        return model_cls(num_classes=10)
    elif model_name == "tv-efficientnetv2":
        return model_cls(num_classes=10, variant=config.variant)
    elif model_name == "timm-resnet":
        return model_cls(
            num_classes=10,
            model_name=config.model_name,
            pretrained=config.pretrained,
            freeze_backbone=True,
        )
    else:
        return model_cls(num_classes=10)


def print_model_info(model: nn.Module) -> None:
    """Print parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")