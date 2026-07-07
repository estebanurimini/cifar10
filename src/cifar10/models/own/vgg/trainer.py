"""VGG trainer — uses StandardTrainer (no custom training logic).

VGG uses the same standard cross-entropy training as most models,
so we simply re-export StandardTrainer under the VGG name.
"""

from cifar10.training.trainer import StandardTrainer as VGGTrainer

__all__ = ["VGGTrainer"]