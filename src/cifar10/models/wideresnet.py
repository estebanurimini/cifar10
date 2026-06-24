"""WideResNet model for CIFAR10.

Architecture only — no training code. Import and use with the training framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Pre-activation Basic Block for WideResNet."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3,
            stride=1, padding=1, bias=False,
        )

        if stride == 1 and in_planes == out_planes:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1,
                stride=stride, bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + shortcut


class NetworkBlock(nn.Module):
    """A sequence of BasicBlocks."""

    def __init__(
        self,
        num_layers: int,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        layers = [
            BasicBlock(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1,
                dropout_rate,
            )
            for i in range(num_layers)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNet(nn.Module):
    """Wide Residual Network for CIFAR10.

    Args:
        depth: Total depth (must satisfy ``(depth - 4) % 6 == 0``).
        widen_factor: Width multiplier.
        dropout_rate: Dropout rate between convolutional layers.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        dropout_rate: float = 0.0,
        num_classes: int = 10,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must satisfy (depth - 4) % 6 == 0"
        n = (depth - 4) // 6
        k = widen_factor
        channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], 1, dropout_rate)
        self.block2 = NetworkBlock(n, channels[1], channels[2], 2, dropout_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], 2, dropout_rate)
        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)

        self._init_weights()

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
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x