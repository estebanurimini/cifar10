"""Custom model implementations trained from scratch (no external weights).

Each model is organized as a sub-package with model.py, config.py, and trainer.py.
"""

from cifar10.models.own.vit import OwnViT, ViTConfig, ViTTrainer
from cifar10.models.own.deit import OwnDeiT, ConvStemPatchEmbedding, DeiTConfig, DistillationTrainer
from cifar10.models.own.resnet import OwnResNet, BasicBlock as ResNetBasicBlock, ResNetConfig, ResNetTrainer
from cifar10.models.own.wrn import OwnWRN, BasicBlock as WRNBasicBlock, NetworkBlock, WRNConfig, WRNTrainer
from cifar10.models.own.vgg import OwnVGG, VGGConfig, VGGTrainer
from cifar10.models.own.efficientnet import (
    OwnEfficientNetV2, MBConv, SqueezeExcitation, FusedMBConv,
    EfficientNetConfig, EfficientNetTrainer,
)
from cifar10.models.own.blocks import (
    PatchEmbedding, MLP, Attention, TransformerBlock, drop_path,
)

__all__ = [
    # Vit
    "OwnViT", "ViTConfig", "ViTTrainer",
    # DeiT
    "OwnDeiT", "ConvStemPatchEmbedding", "DeiTConfig", "DistillationTrainer",
    # ResNet
    "OwnResNet", "ResNetBasicBlock", "ResNetConfig", "ResNetTrainer",
    # WRN
    "OwnWRN", "WRNBasicBlock", "NetworkBlock", "WRNConfig", "WRNTrainer",
    # VGG
    "OwnVGG", "VGGConfig", "VGGTrainer",
    # EfficientNet
    "OwnEfficientNetV2", "MBConv", "SqueezeExcitation", "FusedMBConv",
    "EfficientNetConfig", "EfficientNetTrainer",
    # Shared blocks
    "PatchEmbedding", "MLP", "Attention", "TransformerBlock", "drop_path",
]