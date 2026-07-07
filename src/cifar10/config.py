"""Shared configuration base class for all CIFAR10 training runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaseConfig:
    """Shared defaults for all CIFAR10 training runs.

    All training scripts inherit from this to avoid duplicating common fields.
    Default paths use hidden folders at the project root (./.data, ./.runs/).
    """

    # reproducibility
    seed: int = 42

    # training
    epochs: int = 300
    batch_size: int = 128
    num_workers: int = 0

    # optimization
    weight_decay: float = 5e-2
    label_smoothing: float = 0.1

    # EMA
    ema_decay: float = 0.999

    # model metadata — filled in by subclasses
    source: str = "own"          # "own", "tv", "timm"
    architecture: str = ""       # e.g. "vgg", "wrn", "convnext"
    pretrained: bool = False
    pretrained_source: str | None = None  # e.g. "imagenet1k"
    input_size: int = 32
    data_norm: str = "cifar10"

    # paths — subclasses override run_dir to add model name
    data_dir: Path = field(default_factory=lambda: Path("./.data"))
    run_dir: Path = field(default_factory=lambda: Path("./.runs"))