"""Run discovery, filtering, and metrics loading.

Provides a programmatic API to scan ``.runs/`` for experiment data,
filter by parameters, and load metrics for analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Set of training-relevant fields used for params matching
TRAINING_PARAMS = {
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


@dataclass
class RunInfo:
    """Metadata about a single experiment run.

    Attributes:
        name: Folder name, e.g. ``"own_wrn"`` or ``"own_wrn.1"``.
        path: Full ``Path`` to the run directory.
        params: The ``params.json`` contents as a dict.
        metrics_path: ``Path`` to the metrics CSV file (may not exist).
        has_checkpoint: Whether a checkpoint exists (``checkpoints/`` dir).
        source: Model source, e.g. ``"own"``, ``"tv"``, ``"timm"``.
        architecture: Model architecture, e.g. ``"wrn"``, ``"vgg"``.
    """
    name: str
    path: Path
    params: dict
    metrics_path: Path
    has_checkpoint: bool
    source: str
    architecture: str

    @property
    def base_name(self) -> str:
        """Return the base experiment name (without version suffix).

        E.g. ``"own_wrn.1"`` → ``"own_wrn"``.
        """
        return self.name.split(".")[0]


def discover_runs(root: Path = Path("./.runs")) -> list[RunInfo]:
    """Scan ``.runs/`` for all experiment folders.

    A folder qualifies as a run if it contains a ``params.json``
    (and optionally a ``metrics.csv`` and/or ``checkpoints/``).

    Args:
        root: Root directory to scan (default ``./.runs``).

    Returns:
        A list of ``RunInfo`` objects sorted by name.
    """
    if not root.exists():
        return []

    runs: list[RunInfo] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue

        params_path = item / "params.json"
        if not params_path.exists():
            # Skip directories without params.json (not a valid run)
            continue

        try:
            params = json.loads(params_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        metrics_path = item / "logs" / "metrics.csv"
        checkpoint_dir = item / "checkpoints"
        has_checkpoint = checkpoint_dir.exists() and any(
            checkpoint_dir.iterdir()
        )

        source = params.get("source", "unknown")
        architecture = params.get("architecture", item.name)

        runs.append(RunInfo(
            name=item.name,
            path=item,
            params=params,
            metrics_path=metrics_path,
            has_checkpoint=has_checkpoint,
            source=source,
            architecture=architecture,
        ))

    return runs


def params_match(a: dict, b: dict) -> bool:
    """Compare two param dicts for equality (ignoring metadata fields).

    Only fields in ``TRAINING_PARAMS`` are compared. Missing fields are
    treated as equal.
    """
    for key in TRAINING_PARAMS:
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            return False
    return True


def filter_runs(
    runs: list[RunInfo],
    **filters: Any,
) -> list[RunInfo]:
    """Filter runs by any params field.

    Examples:
        >>> filter_runs(runs, source="own")
        >>> filter_runs(runs, architecture="vgg")
        >>> filter_runs(runs, pretrained=True)
        >>> filter_runs(runs, input_size=128)

    Args:
        runs: List of ``RunInfo`` objects to filter.
        **filters: Key-value pairs to match against ``params``.

    Returns:
        Filtered list of ``RunInfo`` objects.
    """
    result = []
    for run in runs:
        match = True
        for key, value in filters.items():
            if key not in run.params or run.params[key] != value:
                match = False
                break
        if match:
            result.append(run)
    return result


def get_runs(
    names: list[str],
    root: Path = Path("./.runs"),
) -> list[RunInfo]:
    """Get runs by their folder names.

    Supports exact matches (``"own_wrn"``) and base names with version
    (``"own_wrn.1"``). If a base name is given with no version, returns
    the latest version.

    Args:
        names: List of run folder names (e.g. ``["own_wrn", "tv_convnext"]``).
        root: Root runs directory.

    Returns:
        List of ``RunInfo`` objects for the requested runs.
    """
    all_runs = discover_runs(root)

    result = []
    for name in names:
        candidates = [r for r in all_runs if r.name == name]
        if candidates:
            result.append(candidates[0])
            continue

        # Try to find the latest version
        base = name.split(".")[0]
        matching = sorted(
            [r for r in all_runs if r.base_name == base],
            key=lambda r: r.name,
        )
        if matching:
            result.append(matching[-1])

    return result


def load_metrics(run: RunInfo) -> "pd.DataFrame | None":
    """Load metrics CSV as a pandas DataFrame.

    Args:
        run: A ``RunInfo`` object.

    Returns:
        A pandas DataFrame with the metrics, or ``None`` if the CSV
        doesn't exist or pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    if not run.metrics_path.exists():
        return None

    return pd.read_csv(run.metrics_path)


def print_run_table(runs: list[RunInfo]) -> None:
    """Print a summary table of runs for quick inspection.

    Columns: name, source, architecture, params (abbreviated), best accuracy
    (loaded from metrics CSV if available), epochs.

    Args:
        runs: List of ``RunInfo`` objects.
    """
    if not runs:
        print("No runs found.")
        return

    # Try to load best accuracy from metrics CSVs
    rows = []
    for run in runs:
        best_acc = "N/A"
        epochs = "N/A"
        params_str = _abbreviate_params(run.params)

        df = load_metrics(run)
        if df is not None and not df.empty:
            best_acc = f"{df['best_acc'].max():.2f}%"
            epochs = str(len(df))

        rows.append({
            "name": run.name,
            "source": run.source,
            "architecture": run.architecture,
            "params": params_str,
            "best_acc": best_acc,
            "epochs": epochs,
            "checkpoint": "✓" if run.has_checkpoint else "✗",
        })

    # Print as a simple table
    header = f"{'Name':<25} {'Src':<6} {'Arch':<15} {'Params':<30} {'Best Acc':<10} {'Epochs':<8} {'CKPT':<6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<25} {row['source']:<6} {row['architecture']:<15} "
            f"{row['params']:<30} {row['best_acc']:<10} {row['epochs']:<8} {row['checkpoint']:<6}"
        )


def _abbreviate_params(params: dict) -> str:
    """Create a short parameter summary string."""
    parts = []
    if "lr" in params:
        parts.append(f"lr={params['lr']}")
    if "optimizer" in params:
        parts.append(f"opt={params['optimizer']}")
    if "batch_size" in params:
        parts.append(f"bs={params['batch_size']}")
    if "epochs" in params:
        parts.append(f"ep={params['epochs']}")
    return ", ".join(parts)