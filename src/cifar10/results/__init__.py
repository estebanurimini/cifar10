"""Results discovery and filtering module.

Provides utilities to discover, filter, and load experiment run data
from the ``.runs/`` directory.

See ``loader.py`` for the main API.
"""

from .loader import (
    RunInfo,
    discover_runs,
    filter_runs,
    load_metrics,
    params_match,
    print_run_table,
    get_runs,
)

__all__ = [
    "RunInfo",
    "discover_runs",
    "filter_runs",
    "load_metrics",
    "params_match",
    "print_run_table",
    "get_runs",
]