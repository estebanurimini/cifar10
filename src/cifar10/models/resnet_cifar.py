"""ResNet models for CIFAR10.

Implements the CIFAR10-specific ResNet variants from the original paper
(Deep Residual Learning, He et al. 2016). These differ from ImageNet
ResNet in that they use a 3×3 initial convolution (no 7×7 conv, no
max-pool) to preserve the 32×32 input resolution.

Variants:
    - ResNet20:  3 stages × 3 blocks each  (depth = 6×3 + 2 = 20)
    - ResNet56:  3 stages × 9 blocks each  (depth = 6×9 + 2 = 56)

Architecture only — no training code. Import and use with the training framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal


# =========================================================
# BasicBlock (post-activation, used in CIFAR ResNet)
# =========================================================

class BasicBlock(nn.Module):
    """Post-activation BasicBlock for CIFAR ResNet.

    Structure: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU

    This is the *post-activation* variant (from the original ResNet paper),
    as opposed to the pre-activation variant used in WideResNet.

    Args:
        in_planes: Number of input channels.
        planes: Number of output channels.
        stride: Stride for the first convolution (and shortcut).
    """

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# =========================================================
# ResNet CIFAR
# =========================================================

RESNET_CIFAR_VARIANTS = {
    "resnet20": {"num_blocks": [3, 3, 3]},
    "resnet56": {"num_blocks": [9, 9, 9]},
}


class ResNetCIFAR(nn.Module):
    """ResNet for CIFAR10.

    Adapted for small images (32×32):
        - 3×3 conv stride 1 instead of 7×7 conv stride 2.
        - No max-pool after the initial convolution.
        - Three stages with channels [16, 32, 64] (smaller channel counts).
        - Global average pooling → FC.

    Args:
        variant: One of ``"resnet20"`` or ``"resnet56"``.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        variant: Literal["resnet20", "resnet56"] = "resnet20",
        num_classes: int = 10,
    ):
        super().__init__()
        if variant not in RESNET_CIFAR_VARIANTS:
            raise ValueError(
                f"Unknown ResNet variant '{variant}'. "
                f"Available: {list(RESNET_CIFAR_VARIANTS)}"
            )

        self.variant = variant
        num_blocks = RESNET_CIFAR_VARIANTS[variant]["num_blocks"]

        # Initial convolution (no max-pool, preserving 32×32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Three stages
        self.layer1 = self._make_layer(16, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks[2], stride=2)

        # Classifier
        self.linear = nn.Linear(64, num_classes)

        # Weight initialisation
        self._init_weights()

    def _make_layer(
        self,
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_planes, planes, s))
            in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x