"""Unified training script for all CIFAR10 models.

Select a model with ``--model`` using the ``{source}-{architecture}`` naming
convention (e.g. ``own-vit``, ``tv-convnext``, ``own-wrn``).

Usage:
    python -m cifar10.scripts.train --model own-vit
    python -m cifar10.scripts.train --model own-resnet --variant resnet56
    python -m cifar10.scripts.train --model tv-convnext --image-size 128
    python -m cifar10.scripts.train --model own-deit --resume .runs/own_deit/checkpoints/last.pt
    python -m cifar10.scripts.train --list-models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from cifar10.training import (
    BaseTrainer,
    evaluate,
    resolve_run_dir,
    registry,
)
from cifar10.training.registry import (
    MODEL_REGISTRY,
    build_model,
    _build_teacher,
    _copy_teacher_to_run_dir,
    format_config_defaults,
    print_model_info,
    show_config,
)
from cifar10.utils import set_seed, get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for all CIFAR10 models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model identifier (e.g. own-vit, tv-convnext).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit.",
    )
    parser.add_argument(
        "--show-config",
        type=str,
        default=None,
        metavar="MODEL",
        help="Show full config with defaults for a model (e.g. own-wrn) and exit.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant (e.g. resnet56, vgg16_bn). For timm-resnet use --model-name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="TIMM model name (e.g. resnet50, resnet101). Used only with --model timm-resnet.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size (for models that support it).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs.",
    )
    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        print("Available models:")
        for name in sorted(MODEL_REGISTRY.keys()):
            cfg_cls, model_cls, trainer_cls, _, _, description = MODEL_REGISTRY[name]
            cfg = cfg_cls()
            defaults = format_config_defaults(cfg)
            defaults_str = ", ".join(defaults)
            print(f"  {name:25s}  {description}")
            print(f"  {'':25s}  Defaults: {defaults_str}")
            print(f"  {'':25s}  Overridable: --variant, --image-size, --batch-size, --lr, --epochs")
            print()
        return

    # Show config for a specific model and exit
    if args.show_config is not None:
        show_config(args.show_config)
        return

    if args.model is None:
        parser.error("--model is required (use --list-models to see available models)")

    # Look up model entry
    cfg_cls, model_cls, trainer_cls, loader_fn, _, _ = MODEL_REGISTRY[args.model]

    # Build config with optional overrides
    overrides = {}
    if args.variant is not None:
        overrides["variant"] = args.variant
    if args.model_name is not None:
        overrides["model_name"] = args.model_name
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
        overrides["input_size"] = args.image_size
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    cfg = cfg_cls(**overrides)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    # Resolve run directory (versioning)
    run_base_name = f"{cfg.source}_{cfg.architecture}"
    cfg.run_dir = resolve_run_dir(run_base_name, cfg.to_run_params())

    # Build data loaders
    train_loader, val_loader, test_loader = loader_fn(cfg, device)

    # Build model
    model = build_model(model_cls, cfg).to(device)
    print_model_info(model)

    # Special handling for DeiT (teacher model)
    teacher = None
    if args.model == "own-deit":
        # Copy teacher checkpoint into run directory
        local_teacher_ckpt = _copy_teacher_to_run_dir(cfg)
        cfg.teacher_ckpt = local_teacher_ckpt

        # Load teacher
        teacher = _build_teacher(cfg).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        if cfg.teacher_ckpt.exists():
            print(f"Loading teacher weights from {cfg.teacher_ckpt}...")
            ckpt = load_checkpoint(cfg.teacher_ckpt, device)
            if "ema" in ckpt:
                teacher.load_state_dict(ckpt["ema"])
            else:
                teacher.load_state_dict(ckpt["model"])
        else:
            print(
                f"Warning: Teacher checkpoint not found at {cfg.teacher_ckpt}. "
                "Teacher will have random weights. Run train_wrn.py first!"
            )

    # Build trainer
    if args.model == "own-deit":
        trainer = trainer_cls(model, cfg, device, teacher)  # type: ignore[arg-type]
    else:
        trainer = trainer_cls(model, cfg, device)  # type: ignore[arg-type]

    best_acc = trainer.train(train_loader, val_loader, resume_from=args.resume)

    # Final evaluation on held-out test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()