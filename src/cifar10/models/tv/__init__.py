"""TorchVision wrapper models with pretrained weights.

Each model is organized as a sub-package with model.py, config.py, and trainer.py.
"""

from cifar10.models.tv.convnext import TVConvNeXt, ConvNextConfig, ConvNextTrainer
from cifar10.models.tv.efficientnet_v2 import TVEfficientNetV2, EfficientNetV2Config, EfficientNetV2Trainer

__all__ = [
    "TVConvNeXt", "ConvNextConfig", "ConvNextTrainer",
    "TVEfficientNetV2", "EfficientNetV2Config", "EfficientNetV2Trainer",
]