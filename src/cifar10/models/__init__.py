"""
Model source separation.

``models/`` is organized by source:

- ``models.own`` — Custom implementations trained from scratch (no external weights)
- ``models.tv`` — TorchVision wrappers with pretrained weights
- ``models.timm`` — TIMM wrappers with pretrained weights
"""

# ---------------------------------------------------------------------------
# Own models (from scratch, no pretrained weights)
# ---------------------------------------------------------------------------
from cifar10.models.own import (
    OwnVGG,
    OwnWRN, WRNBasicBlock, NetworkBlock,
    OwnResNet, ResNetBasicBlock,
    OwnViT, ViTConfig, ViTTrainer,
    OwnDeiT, ConvStemPatchEmbedding, DeiTConfig, DistillationTrainer,
    OwnEfficientNetV2, MBConv, SqueezeExcitation, FusedMBConv, EfficientNetConfig, EfficientNetTrainer,
    PatchEmbedding, MLP, Attention, TransformerBlock, drop_path,
    WRNConfig, WRNTrainer,
    VGGConfig, VGGTrainer,
    ResNetConfig, ResNetTrainer,
)

# ---------------------------------------------------------------------------
# TV models (torchvision pretrained)
# ---------------------------------------------------------------------------
from cifar10.models.tv import (
    TVConvNeXt, ConvNextConfig, ConvNextTrainer,
    TVEfficientNetV2, EfficientNetV2Config, EfficientNetV2Trainer,
)

# ---------------------------------------------------------------------------
# TIMM models (timm pretrained, optional dependency)
# ---------------------------------------------------------------------------
try:
    from cifar10.models.timm.resnet import TimmResNet
except ImportError:
    class TimmResNet:  # type: ignore
        """Placeholder: install ``timm`` to use this model."""
        def __init__(self, *args, **kwargs):
            raise ImportError("timm is not installed. Run: pip install timm")

__all__ = [
    # Own models
    "OwnVGG", "OwnWRN", "OwnResNet", "OwnViT", "OwnDeiT", "OwnEfficientNetV2",
    # Own blocks
    "MBConv", "SqueezeExcitation", "FusedMBConv",
    "ConvStemPatchEmbedding", "WRNBasicBlock", "NetworkBlock",
    "ResNetBasicBlock", "PatchEmbedding", "MLP", "Attention",
    "TransformerBlock", "drop_path",
    # Own configs & trainers
    "ViTConfig", "ViTTrainer",
    "DeiTConfig", "DistillationTrainer",
    "ResNetConfig", "ResNetTrainer",
    "WRNConfig", "WRNTrainer",
    "VGGConfig", "VGGTrainer",
    "EfficientNetConfig", "EfficientNetTrainer",
    # TV models
    "TVConvNeXt", "TVEfficientNetV2",
    # TV configs & trainers
    "ConvNextConfig", "ConvNextTrainer",
    "EfficientNetV2Config", "EfficientNetV2Trainer",
    # TIMM models
    "TimmResNet",
]