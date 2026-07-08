"""Automatic versioned run directory resolution.

Scans ``.runs/`` for existing run directories with matching parameters.
If found, reuses the directory (resume). Otherwise creates a versioned folder.
"""

from __future__ import annotations

import json
from pathlib import Path


TRAINING_PARAMS = {
    # The set of fields that determine whether two runs are "the same".
    # Metadata fields like "created_at", "notes", "experiment_name", "run_name"
    # are excluded from comparison.
    "source", "architecture", "pretrained", "pretrained_source",
    "input_size", "data_norm",
    "optimizer", "lr", "weight_decay", "momentum", "label_smoothing",
    "epochs", "warmup_epochs", "min_lr",
    "use_randaugment", "use_trivialaugmentwide",
    "augmentation", "mixup_prob", "cutmix_prob",
    "augment", "augment_switch_epochs", "cutout_size",
    "mixup_alpha", "cutmix_alpha",
    "ema_decay", "dropout", "stochastic_depth",
    "freeze_backbone_epochs", "backbone_lr_scale",
    "batch_size", "seed",
    "arch_params",
    "teacher",
}


def params_match(a: dict, b: dict) -> bool:
    """Compare two param dicts for equality (ignoring metadata fields).

    Only fields in ``TRAINING_PARAMS`` are compared. Missing fields are
    treated as equal (e.g., comparing a new config dict with an older one
    that lacks a newly-added field).
    """
    for key in TRAINING_PARAMS:
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            return False
    return True


def resolve_run_dir(base_name: str, config: dict) -> Path:
    """Scan .runs/ for ``{base_name}`` directories.

    If an existing dir has matching params → reuse it (resume).
    Otherwise → create a new versioned folder.

    Args:
        base_name: The experiment base name (e.g. ``"own_wrn"``).
        config: The full resolved config dict (as would be written to
            ``params.json``). Must contain at least the training-relevant keys.

    Returns:
        The path to the (existing or newly-created) run directory.
    """
    runs_root = Path("./.runs")
    runs_root.mkdir(parents=True, exist_ok=True)

    existing_dirs = sorted(runs_root.glob(f"{base_name}*"))

    for dirpath in existing_dirs:
        if not dirpath.is_dir():
            continue
        params_path = dirpath / "params.json"
        if params_path.exists():
            try:
                existing_params = json.loads(params_path.read_text())
                if params_match(existing_params, config):
                    print(
                        f"[resume] Found matching run: {dirpath} — "
                        f"resuming from existing directory"
                    )
                    return dirpath
            except (json.JSONDecodeError, OSError):
                continue
        # If dir has no params.json, also skip (might be old/manual dir)
        else:
            # But if it only contains checkpoints, treat as matching
            # (backward compat for migrated runs)
            checkpoint_dir = dirpath / "checkpoints"
            if checkpoint_dir.exists():
                print(
                    f"[resume] Found existing run dir {dirpath} "
                    f"(no params.json, has checkpoints) — reusing"
                )
                return dirpath

    # No match → create versioned folder
    version = 1
    candidate = runs_root / base_name
    while candidate.exists():
        version += 1
        candidate = runs_root / f"{base_name}.{version}"

    if version == 1:
        print(f"[new] Creating new run directory: {candidate}")
    else:
        print(
            f"[new] Run {runs_root / base_name} already exists with "
            f"different params — creating {candidate}"
        )

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate