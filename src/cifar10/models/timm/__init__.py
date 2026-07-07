"""TIMM (PyTorch Image Models) wrapper models with pretrained weights.
"""

from cifar10.models.timm.resnet import TimmResNet
from cifar10.models.timm.resnet.config import TimmResNetConfig
from cifar10.models.timm.resnet.trainer import TimmResNetTrainer

__all__ = [
    "TimmResNet",
    "TimmResNetConfig",
    "TimmResNetTrainer",
]
