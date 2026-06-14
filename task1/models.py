"""Models for CIFAR-10 task 1.

The main network intentionally contains the components required by the assignment:
Conv2d, pooling, activations, fully connected layer, BatchNorm, Dropout, and
residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    """Create an activation module from a compact name."""
    name = name.lower().replace("-", "_")
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(
        f"Unknown activation {name!r}. Choose from: relu, leaky_relu, elu, gelu, silu."
    )


class ConvBlock(nn.Module):
    """Conv2d -> optional BatchNorm -> activation -> optional Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """A small residual block suitable for 32x32 CIFAR images."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_bn: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            use_bn=use_bn,
            activation=activation,
            dropout=0.0,
        )
        second: list[nn.Module] = [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        ]
        if use_bn:
            second.append(nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(*second)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = make_activation(activation)

        if in_channels != out_channels:
            projection: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not use_bn)
            ]
            if use_bn:
                projection.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*projection)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + self.shortcut(x)
        return self.activation(out)


class CIFARConvNet(nn.Module):
    """Compact CNN for CIFAR-10 classification.

    Input shape: [N, 3, 32, 32]
    Output shape: [N, 10]
    """

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 32,
        activation: str = "relu",
        dropout: float = 0.10,
        use_bn: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            ConvBlock(3, c1, use_bn=use_bn, activation=activation),
            ResidualBlock(
                c1,
                c1,
                use_bn=use_bn,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
            ),
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(
                c1,
                c2,
                use_bn=use_bn,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
            ),
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(
                c2,
                c3,
                use_bn=use_bn,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
            ),
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(
                c3,
                c3,
                use_bn=use_bn,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(c3, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    *,
    num_classes: int = 10,
    base_channels: int = 32,
    activation: str = "relu",
    dropout: float = 0.10,
    use_bn: bool = True,
    use_residual: bool = True,
) -> CIFARConvNet:
    return CIFARConvNet(
        num_classes=num_classes,
        base_channels=base_channels,
        activation=activation,
        dropout=dropout,
        use_bn=use_bn,
        use_residual=use_residual,
    )
