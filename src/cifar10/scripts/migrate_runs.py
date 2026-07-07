"""Migrate existing run directories to the new naming convention.

For each existing run, this script:
1. Reads the checkpoint config (best.pt or last.pt)
2. Writes ``params.json`` from that config
3. Renames the folder if it doesn't follow the ``{source}_{architecture}`` convention
4. Cleans interleaved CSVs (keeps last contiguous run)

Usage:
    python -m cifar10.scripts.migrate_runs          # Preview mode
    python -m cifar10.scripts.migrate_runs --apply   # Apply changes
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

# Mapping: old folder name → (new folder name, source, architecture, params_override)
RENAME_MAP: dict[str, dict] = {
    "vgg": {
        "new_name": "own_vgg",
        "source": "own",
        "architecture": "vgg",
        "defaults": {
            "variant": "vgg16_bn",
            "vgg_dropout": 0.5,
            "lr": 0.05,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "epochs": 200,
            "warmup_epochs": 0,
            "min_lr": 0.0,
            "optimizer": "sgd",
        },
    },
    "wrn": {
        "new_name": "own_wrn",
        "source": "own",
        "architecture": "wrn",
        "defaults": {
            "depth": 16,
            "widen_factor": 4,
            "wrn_dropout": 0.0,
            "lr": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "epochs": 200,
            "warmup_epochs": 0,
            "min_lr": 0.0,
            "optimizer": "sgd",
        },
    },
    "vit": {
        "new_name": "own_vit",
        "source": "own",
        "architecture": "vit",
        "defaults": {
            "image_size": 32,
            "patch_size": 4,
            "embed_dim": 192,
            "depth": 6,
            "num_heads": 3,
            "mlp_ratio": 4,
            "dropout": 0.1,
            "optimizer": "adamw",
        },
    },
    "deit": {
        "new_name": "own_deit",
        "source": "own",
        "architecture": "deit",
        "defaults": {
            "image_size": 32,
            "patch_size": 4,
            "embed_dim": 192,
            "depth": 6,
            "num_heads": 3,
            "mlp_ratio": 4,
            "dropout": 0.1,
            "drop_path_rate": 0.1,
            "teacher_reliance": 0.7,
            "wrn_depth": 16,
            "wrn_width": 4,
            "optimizer": "adamw",
        },
    },
    "convnext": {
        "new_name": "tv_convnext",
        "source": "tv",
        "architecture": "convnext",
        "defaults": {
            "image_size": 128,
            "pretrained": True,
            "pretrained_source": "imagenet1k",
            "input_size": 128,
            "data_norm": "imagenet",
            "lr": 1e-3,
            "weight_decay": 0.05,
            "freeze_backbone_epochs": 10,
            "backbone_lr_scale": 0.1,
            "optimizer": "adamw",
        },
    },
    "efficientnet_v2": {
        "new_name": "tv_efficientnetv2",
        "source": "tv",
        "architecture": "efficientnetv2",
        "defaults": {
            "variant": "s",
            "image_size": 128,
            "pretrained": True,
            "pretrained_source": "imagenet1k",
            "input_size": 128,
            "data_norm": "imagenet",
            "lr": 0.0625,
            "weight_decay": 2e-5,
            "freeze_backbone_epochs": 16,
            "backbone_lr_scale": 0.01,
            "optimizer": "adamw",
        },
    },
    "efficientnet_scratch": {
        "new_name": "own_efficientnetv2",
        "source": "own",
        "architecture": "efficientnetv2",
        "defaults": {
            "stochastic_depth_prob": 0.1,
            "lr": 3e-3,
            "weight_decay": 5e-2,
            "optimizer": "adamw",
        },
    },
    "resnet": {
        "new_name": "own_resnet",
        "source": "own",
        "architecture": "resnet",
        "defaults": {
            "variant": "resnet56",
            "lr": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "epochs": 200,
            "warmup_epochs": 5,
            "mixup_alpha": 0.2,
            "cutmix_alpha": 1.0,
            "optimizer": "sgd",
        },
    },
}


def _get_checkpoint_dir(run_dir: Path) -> Path | None:
    """Find checkpoint directory inside a run dir."""
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        return ckpt_dir
    return None


def _get_best_checkpoint(ckpt_dir: Path) -> Path | None:
    """Get best.pt or last.pt from checkpoint directory."""
    for name in ["best.pt", "last.pt"]:
        path = ckpt_dir / name
        if path.exists():
            return path
    return None


def _extract_config_from_checkpoint(ckpt_path: Path) -> dict | None:
    """Load checkpoint and extract config dict."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("config", None)
    except Exception:
        return None


def _clean_csv(csv_path: Path) -> bool:
    """Remove interleaved runs from a CSV, keeping only the last contiguous run.

    This handles the case where a CSV has data from multiple training runs
    (e.g., due to training restarts). We keep only the last sequence that
    starts at epoch=1 and increases monotonically.

    Args:
        csv_path: Path to the metrics CSV.

    Returns:
        True if the CSV was cleaned (modified), False if it was already clean.
    """
    import csv as csv_module

    if not csv_path.exists():
        return False

    with open(csv_path, "r") as f:
        reader = csv_module.reader(f)
        rows = list(reader)

    if len(rows) < 2:  # header only or empty
        return False

    header = rows[0]
    data = rows[1:]

    # Find the last contiguous run: scan backwards for the last epoch=1,
    # then keep everything from there to the end.
    last_start = -1
    for i in range(len(data) - 1, -1, -1):
        try:
            if int(data[i][0]) == 1:  # epoch column
                last_start = i
                break
        except (ValueError, IndexError):
            continue

    if last_start <= 0:
        return False  # Already clean or only one run

    # Keep only the last contiguous run
    kept_data = data[last_start:]

    # Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(header)
        writer.writerows(kept_data)

    return True


def _write_params_json(run_dir: Path, config: dict, metadata: dict) -> None:
    """Write params.json to a run directory.

    Args:
        run_dir: The run directory.
        config: Config dict from checkpoint (may be None for legacy runs).
        metadata: Dict with source, architecture, defaults, etc.
    """
    params_path = run_dir / "params.json"
    if params_path.exists():
        print(f"  [skip] params.json already exists: {params_path}")
        return

    # Start with checkpoint config if available, else use defaults
    params = {}
    if config:
        params.update(config)
    else:
        params.update(metadata.get("defaults", {}))

    # Add metadata fields
    params["source"] = metadata["source"]
    params["architecture"] = metadata["architecture"]
    params["pretrained"] = params.get("pretrained", False)
    params["pretrained_source"] = params.get("pretrained_source", None)
    params["input_size"] = params.get("input_size", 32)
    params["data_norm"] = params.get("data_norm", "cifar10")
    params["use_randaugment"] = params.get("use_randaugment", True)
    params["label_smoothing"] = params.get("label_smoothing", 0.1)
    params["seed"] = params.get("seed", 42)
    params["arch_params"] = params.get("arch_params", {})
    params["teacher"] = params.get("teacher", None)
    params["created_at"] = params.get("created_at", "2026-01-01T00:00:00")
    params["notes"] = params.get("notes", "Migrated from legacy naming")

    # Ensure run_dir key is a string
    if "run_dir" in params:
        params["run_dir"] = str(params["run_dir"])
    if "data_dir" in params:
        params["data_dir"] = str(params["data_dir"])
    for k in ["teacher_ckpt"]:
        if k in params and isinstance(params[k], (str, Path)):
            params[k] = str(params[k])

    params_path.write_text(json.dumps(params, indent=4, default=str))
    print(f"  [params] Written: {params_path}")


def migrate_run(old_dir: Path, apply: bool = False) -> dict:
    """Migrate a single run directory.

    Args:
        old_dir: Path to the old run directory.
        apply: If True, apply changes. If False, preview only.

    Returns:
        Dict with migration info: old_name, new_name, renamed, params_written, csv_cleaned.
    """
    result = {
        "old_name": old_dir.name,
        "new_name": None,
        "renamed": False,
        "params_written": False,
        "csv_cleaned": False,
    }

    folder_name = old_dir.name

    # Determine mapping
    mapping = RENAME_MAP.get(folder_name)
    if not mapping:
        # Check if it's already in the new format
        if folder_name.startswith(("own_", "tv_", "timm_")):
            result["new_name"] = folder_name
        else:
            return result  # Unknown, skip

    # Find checkpoint
    ckpt_dir = _get_checkpoint_dir(old_dir)
    config = None
    if ckpt_dir:
        ckpt_path = _get_best_checkpoint(ckpt_dir)
        if ckpt_path:
            config = _extract_config_from_checkpoint(ckpt_path)

    # Determine new name
    new_name = mapping["new_name"]
    result["new_name"] = new_name
    target_dir = old_dir.parent / new_name

    # Clean CSV if needed
    csv_path = old_dir / "logs" / "metrics.csv"
    if csv_path.exists():
        cleaned = _clean_csv(csv_path)
        result["csv_cleaned"] = cleaned
        if apply and cleaned:
            print(f"  [csv] Cleaned interleaved runs: {csv_path}")

    # Rename directory
    if old_dir.name != new_name:
        result["renamed"] = True
        if apply:
            if target_dir.exists():
                print(f"  [warn] Target {target_dir} already exists. "
                      f"Merging content into {target_dir}...")
                # Copy checkpoints and logs into existing target
                if ckpt_dir and ckpt_dir.exists():
                    for p in ckpt_dir.iterdir():
                        dest = target_dir / "checkpoints" / p.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dest)
                log_dir = old_dir / "logs"
                if log_dir.exists():
                    for p in log_dir.iterdir():
                        dest = target_dir / "logs" / p.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dest)
            else:
                old_dir.rename(target_dir)
                print(f"  [rename] {old_dir} → {target_dir}")
        else:
            print(f"  [preview] Would rename: {old_dir} → {target_dir}")
    else:
        target_dir = old_dir

    # Write params.json
    if mapping:
        current_dir = target_dir if (apply or old_dir.name == new_name) else old_dir
        if (current_dir / "params.json").exists():
            result["params_written"] = True
        else:
            if apply:
                _write_params_json(current_dir, config, mapping)
                result["params_written"] = True
            else:
                print(f"  [preview] Would write params.json to {current_dir}")
                result["params_written"] = False

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate run directories to new naming convention."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default: preview only).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("./.runs"),
        help="Root runs directory (default: ./.runs).",
    )
    args = parser.parse_args()

    runs_root = args.runs_dir
    if not runs_root.exists():
        print(f"No runs directory found at {runs_root}")
        return

    mode = "APPLY" if args.apply else "PREVIEW"
    print(f"=== Migration Mode: {mode} ===\n")

    results = []
    for item in sorted(runs_root.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        result = migrate_run(item, apply=args.apply)
        results.append(result)

    print(f"\n=== Summary ===")
    renamed = sum(1 for r in results if r["renamed"])
    params_written = sum(1 for r in results if r["params_written"])
    csv_cleaned = sum(1 for r in results if r["csv_cleaned"])
    skipped = sum(1 for r in results if r["new_name"] is None)

    print(f"  Total directories scanned: {len(results)}")
    print(f"  Renamed: {renamed}")
    print(f"  params.json written: {params_written}")
    print(f"  CSVs cleaned: {csv_cleaned}")
    print(f"  Skipped (unknown format): {skipped}")

    if not args.apply:
        print(f"\n  [hint] Run with --apply to apply changes.")


if __name__ == "__main__":
    main()