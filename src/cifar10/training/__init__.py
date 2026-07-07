"""Training package — generic infrastructure and model-specific configuration.

Sub-packages:
    registry.py — Unified model registry mapping identifiers to components.
    resolver.py — Automatic versioned run directory resolution.
"""

from .trainer import BaseTrainer, StandardTrainer, TrainerConfig
from .evaluate import evaluate, detailed_evaluate, format_evaluation_results
from .scheduler import build_scheduler
from .resolver import resolve_run_dir, params_match

__all__ = [
    "BaseTrainer",
    "StandardTrainer",
    "TrainerConfig",
    "evaluate",
    "detailed_evaluate",
    "format_evaluation_results",
    "build_scheduler",
    "resolve_run_dir",
    "params_match",
]