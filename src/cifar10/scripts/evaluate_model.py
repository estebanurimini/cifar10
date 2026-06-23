"""Evaluate a trained model on the CIFAR-10 test set.

Loads a checkpoint, reconstructs the model architecture (using either embedded
config from the checkpoint or default configs from the training modules), and
prints detailed evaluation metrics including per-class accuracy and inference
throughput.

Usage:
    # Evaluate with embedded config (new checkpoints)
    python -m cifar10.scripts.evaluate_model \\
        --model vit \\
        --checkpoint .runs/vit/checkpoints/best.pt

    # Evaluate a legacy checkpoint (no embedded config)
    python -m cifar10.scripts.evaluate_model \\
        --model wrn \\
        --checkpoint .runs/wrn/checkpoints/best.pt

    # With custom batch size
    python -m cifar10.scripts.evaluate_model \\
        --model deit \\
        --checkpoint .runs/deit/checkpoints/best.pt \\
        --batch-size 256
"""

import argparse
import dataclasses
from pathlib import Path

import torch

from cifar10.data.cifar10 import CIFAR10_NORM
from cifar10.training import evaluate, detailed_evaluate, format_evaluation_results

# CIFAR-10 class names (in order of the dataset's class indices)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ---------------------------------------------------------------------------
# Default configs for each model type — used when checkpoint has no embedded
# config (legacy checkpoints). These must be imported lazily to avoid circular
# imports and to keep the script lightweight if only one model is needed.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple] = {}

def _register_defaults():
    """Lazy-import and register default config & model builders."""
    from cifar10.scripts.train_vit import ViTConfig
    from cifar10.scripts.train_wrn import WRNConfig
    from cifar10.scripts.train_deit import DeiTConfig
    from cifar10.models import ViT, DeiT, WideResNet

    MODEL_REGISTRY["vit"] = (ViTConfig, _build_vit)
    MODEL_REGISTRY["wrn"] = (WRNConfig, _build_wrn)
    MODEL_REGISTRY["deit"] = (DeiTConfig, _build_deit)

    # Store model classes for checkpoint-based rebuild
    global _MODEL_CLASSES
    _MODEL_CLASSES = {
        "vit": ViT,
        "wrn": WideResNet,
        "deit": DeiT,
    }


def _build_vit(config, device):
    from cifar10.models import ViT
    return ViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=10,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
    ).to(device)


def _build_wrn(config, device):
    from cifar10.models import WideResNet
    return WideResNet(
        depth=config.depth,
        widen_factor=config.widen_factor,
        dropout_rate=config.wrn_dropout,
        num_classes=10,
    ).to(device)


def _build_deit(config, device):
    from cifar10.models import DeiT
    return DeiT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=10,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        drop_path_rate=config.drop_path_rate,
    ).to(device)


# ---------------------------------------------------------------------------
# Config reconstruction from checkpoint
# ---------------------------------------------------------------------------

def reconstruct_config(config_dict: dict, model_type: str):
    """Rebuild a config dataclass from a dict (e.g., from checkpoint metadata).

    Args:
        config_dict: The ``"config"`` dict stored in the checkpoint.
        model_type: One of ``"vit"``, ``"wrn"``, ``"deit"``.

    Returns:
        A config dataclass instance with fields populated from the dict.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available: {list(MODEL_REGISTRY)}")

    config_cls = MODEL_REGISTRY[model_type][0]

    # Filter the config dict to only include fields that the dataclass expects
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

    # Convert Path fields back from string
    for field in dataclasses.fields(config_cls):
        if field.type is Path and field.name in filtered:
            filtered[field.name] = Path(filtered[field.name])
        elif field.type is Path and field.name in filtered:
            # Handle Optional[Path] or Union types
            pass

    return config_cls(**filtered)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CIFAR-10 model.",
    )
    parser.add_argument(
        "--model",
        choices=["vit", "wrn", "deit"],
        required=True,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file (e.g., .runs/vit/checkpoints/best.pt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./.data"),
        help="CIFAR-10 dataset root directory (default: ./.data).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=True,
        help="Print per-class accuracy (default: True).",
    )
    parser.add_argument(
        "--no-detailed",
        action="store_false",
        dest="detailed",
        help="Skip per-class accuracy breakdown.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy-register defaults
    _register_defaults()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    # Reconstruct config
    if "config" in ckpt:
        print("Using embedded config from checkpoint.")
        config = reconstruct_config(ckpt["config"], args.model)
    else:
        print("No embedded config found — using default config.")
        config_cls = MODEL_REGISTRY[args.model][0]
        config = config_cls()

    # Build model
    model = MODEL_REGISTRY[args.model][1](config, device)
    model.eval()

    # Load weights (try EMA first, fall back to raw model)
    if "ema" in ckpt:
        print("Loading EMA weights.")
        model.load_state_dict(ckpt["ema"])
    else:
        print("Loading raw model weights.")
        model.load_state_dict(ckpt["model"])

    # Build test loader directly (always uses eval transforms)
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_NORM["mean"], CIFAR10_NORM["std"]),
    ])
    test_dataset = datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=True,
        transform=eval_tfms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Evaluate
    if args.detailed:
        metrics = detailed_evaluate(
            model, test_loader, device, num_classes=10,
        )
        print(format_evaluation_results(metrics, class_names=CIFAR10_CLASSES))
    else:
        loss, acc = evaluate(model, test_loader, device)
        print(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()