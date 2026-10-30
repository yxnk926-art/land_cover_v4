# -*- coding: utf-8 -*-
"""Dependency-free MobileNetV3-Large feature extractor for dense prediction.

The block schedule follows Howard et al., ICCV 2019.  For segmentation the
last spatial reduction can be replaced by dilation so that the backbone keeps
an output stride of 16 (or 8).  The module returns the 1/4, 1/8 and final
feature maps needed by DeepLabV3+ and LR-ASPP.

Only PyTorch is required; torchvision is intentionally not imported because
the training servers used by this project do not have a stable torchvision
version.  This module initializes from scratch; the leaves load timm's
ImageNet weights through ``common.pretrained`` (full tail, BN eps 1e-5).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


def make_divisible(value, divisor=8, min_value=None):
    """Round a channel count as in the MobileNet implementations."""
    min_value = divisor if min_value is None else min_value
    rounded = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, dilation=1, activation="hswish",
                 norm_eps=1e-3, norm_momentum=0.01):
        padding = dilation * (kernel_size - 1) // 2
        if activation == "hswish":
            act = nn.Hardswish(inplace=True)
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation is None:
            act = nn.Identity()
        else:
            raise ValueError(f"unsupported activation: {activation!r}")
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(
                out_channels, eps=norm_eps, momentum=norm_momentum
            ),
            act,
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, squeeze_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return x * self.gate(scale)


@dataclass(frozen=True)
class InvertedResidualConfig:
    kernel: int
    expanded_channels: int
    out_channels: int
    use_se: bool
    activation: str
    stride: int


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, cfg, dilation=1, norm_eps=1e-3):
        super().__init__()
        if cfg.stride not in (1, 2):
            raise ValueError("MobileNetV3 stride must be 1 or 2")
        layers = []
        if cfg.expanded_channels != in_channels:
            layers.append(ConvBNAct(in_channels, cfg.expanded_channels, 1,
                                    activation=cfg.activation, norm_eps=norm_eps))
        self.depthwise_index = len(layers)
        layers.append(ConvBNAct(
            cfg.expanded_channels,
            cfg.expanded_channels,
            cfg.kernel,
            stride=cfg.stride,
            groups=cfg.expanded_channels,
            dilation=dilation,
            activation=cfg.activation,
            norm_eps=norm_eps,
        ))
        if cfg.use_se:
            squeeze = make_divisible(cfg.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(cfg.expanded_channels, squeeze))
        layers.append(ConvBNAct(cfg.expanded_channels, cfg.out_channels, 1,
                                activation=None, norm_eps=norm_eps))
        self.block = nn.Sequential(*layers)
        self.use_residual = cfg.stride == 1 and in_channels == cfg.out_channels

    def forward(self, x):
        out, _ = self.forward_with_depthwise(x)
        return out

    def forward_with_depthwise(self, x):
        """Return the block result and its activated depthwise feature."""
        out = x
        depthwise = None
        for index, layer in enumerate(self.block):
            out = layer(out)
            if index == self.depthwise_index:
                depthwise = out
        result = x + out if self.use_residual else out
        return result, depthwise


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Large backbone with segmentation output-stride control."""

    _LARGE = (
        # k, expansion, output, SE, non-linearity, stride
        (3, 16, 16, False, "relu", 1),
        (3, 64, 24, False, "relu", 2),
        (3, 72, 24, False, "relu", 1),
        (5, 72, 40, True, "relu", 2),
        (5, 120, 40, True, "relu", 1),
        (5, 120, 40, True, "relu", 1),
        (3, 240, 80, False, "hswish", 2),
        (3, 200, 80, False, "hswish", 1),
        (3, 184, 80, False, "hswish", 1),
        (3, 184, 80, False, "hswish", 1),
        (3, 480, 112, True, "hswish", 1),
        (3, 672, 112, True, "hswish", 1),
        (5, 672, 160, True, "hswish", 2),
        (5, 960, 160, True, "hswish", 1),
        (5, 960, 160, True, "hswish", 1),
    )

    def __init__(self, in_channels=3, width_mult=1.0, output_stride=16,
                 reduced_tail=False, norm_eps=1e-3):
        super().__init__()
        if output_stride not in (8, 16, 32):
            raise ValueError("output_stride must be one of 8, 16, 32")
        if in_channels < 1 or width_mult <= 0:
            raise ValueError("in_channels and width_mult must be positive")

        # BatchNorm eps: 1e-3 is the torchvision/TF value used by the v3
        # scratch baselines; timm's ImageNet weights were trained with 1e-5.
        stem_channels = make_divisible(16 * width_mult, 8)
        self.stem = ConvBNAct(in_channels, stem_channels, 3, stride=2,
                              activation="hswish", norm_eps=norm_eps)

        current_stride = 2
        dilation = 1
        input_channels = stem_channels
        blocks = []
        low_channels = low8_channels = low8_expanded_channels = None

        for block_index, (
                kernel, expansion, output, use_se, activation, nominal_stride
        ) in enumerate(self._LARGE):
            # Howard et al. halve the final MobileNetV3 stage for their
            # Cityscapes candidates (Table 7, ``RF2``).  Keep this optional so
            # DeepLabV3+ can still use the unmodified classification schedule.
            tail_divisor = 2 if reduced_tail and block_index >= 12 else 1
            expansion = expansion / tail_divisor
            output = output / tail_divisor
            expanded_channels = make_divisible(expansion * width_mult, 8)
            output_channels = make_divisible(output * width_mult, 8)
            stride = nominal_stride
            if current_stride >= output_stride and stride == 2:
                stride = 1
                block_dilation = dilation
                dilation *= 2
            else:
                current_stride *= stride
                block_dilation = dilation
            cfg = InvertedResidualConfig(
                kernel, expanded_channels, output_channels, use_se, activation, stride
            )
            blocks.append(InvertedResidual(
                input_channels, cfg, dilation=block_dilation, norm_eps=norm_eps
            ))
            input_channels = output_channels
            # Capture the ends of the canonical /4 and /8 stages by block
            # identity.  Looking only at current_stride would overwrite low8
            # with every later block when output_stride=8.
            if block_index == 2:
                low_channels = output_channels
            if block_index == 5:
                low8_channels = output_channels
                low8_expanded_channels = expanded_channels

        final_divisor = 2 if reduced_tail else 1
        high_channels = make_divisible(960 / final_divisor * width_mult, 8)
        self.blocks = nn.ModuleList(blocks)
        self.final_conv = ConvBNAct(input_channels, high_channels, 1,
                                    activation="hswish", norm_eps=norm_eps)
        self.low_channels = low_channels
        self.low8_channels = low8_channels
        self.low8_expanded_channels = low8_expanded_channels
        self.high_channels = high_channels
        self.output_stride = output_stride
        self.reduced_tail = bool(reduced_tail)
        self.width_mult = width_mult
        self._initialize_weights()

    def pretrained_arch(self):
        """Architecture name of matching ImageNet weights (full tail only)."""
        if self.reduced_tail:
            raise ValueError(
                "ImageNet MobileNetV3 权重对应完整尾部；加载预训练时请设 "
                "MOBILENET_REDUCED_TAIL=False"
            )
        return f"mobilenetv3_large_{round(self.width_mult * 100)}"

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        low = low8 = low8_expanded = None
        for block_index, block in enumerate(self.blocks):
            if block_index == 5:
                x, low8_expanded = block.forward_with_depthwise(x)
            else:
                x = block(x)
            if block_index == 2:
                low = x
            if block_index == 5:
                low8 = x
        high = self.final_conv(x)
        if low is None or low8 is None or low8_expanded is None:
            raise RuntimeError("failed to collect MobileNetV3 low-level features")
        return {
            "low": low,
            "low8": low8,
            "low8_expanded": low8_expanded,
            "high": high,
        }


if __name__ == "__main__":
    net = MobileNetV3Backbone(width_mult=0.75, output_stride=16).eval()
    with torch.no_grad():
        features = net(torch.randn(1, 3, 129, 161))
    print({name: tuple(value.shape) for name, value in features.items()})
