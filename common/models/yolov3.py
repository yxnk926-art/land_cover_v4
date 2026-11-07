# -*- coding: utf-8 -*-
"""YOLOv3 (Redmon & Farhadi, arXiv:1804.02767, 2018) as a semantic segmenter.

The network is built layer by layer from the official ``cfg/yolov3.cfg`` of
pjreddie/darknet; the comments give the darknet layer indices.

* Darknet-53 (layers 0-74): a 3x3 stem and five stages, each a stride-2 3x3
  convolution followed by 1, 2, 8, 8, 4 residual blocks (1x1 halving, 3x3,
  identity shortcut).  Every convolution has BatchNorm and LeakyReLU(0.1).
  Parameter names follow timm's ``darknet53`` so tests can compare the two.
* Neck (layers 75-79, 83-91, 95-103): five alternating 1x1/3x3 convolutions on
  stride 32, then 1x1 reduction, nearest x2 upsampling and concatenation with
  the stride-16 stage (route 61), five convolutions, and again with the
  stride-8 stage (route 36), five convolutions.
* Output (layers 104-105): a 3x3 convolution and a 1x1 linear convolution on
  the stride-8 grid.

Adaptation to segmentation.  For every grid cell a YOLO output layer predicts,
per anchor, box offsets, an objectness score and class scores; YOLOv1 draws
the class part as a "class probability map".  Here the stride-8 output layer
predicts only the ``num_classes`` scores of its cell, and the map is upsampled
bilinearly to the input size, as DDRNet and PIDNet do with their 1/8 logits.
The stride-32 and stride-16 output branches (layers 80-82, 92-94) serve only
box prediction for larger objects and are left out; the stride-8 branch still
receives both coarser levels through routes 83-86 and 95-98.  Taking the 1/8
level for dense prediction follows YOLOP (Wu et al., MIR 2022).  Class scores
are trained with the repository's softmax cross-entropy (set cross-entropy on
the unified protocol) instead of YOLOv3's independent logistic classifiers,
because land-cover classes are mutually exclusive.

Darknet details kept: LeakyReLU slope 0.1, BatchNorm eps 1e-5 (darknet's GPU
normalisation) and momentum 0.01 (its rolling averages), nearest upsampling,
and darknet's initialisation of new convolutions.  The ImageNet weights expect
RGB in [0, 1] without mean/std normalisation, so ``forward`` undoes the data
pipeline's normalisation before the first convolution.  Inputs whose sides
are not multiples of 32 are padded on the right and bottom with the mean
colour (zero after normalisation) and the logits are cropped back.

``forward`` returns a one-element list of full-resolution logits.
"""
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DARKNET53_WIDTHS = (64, 128, 256, 512, 1024)
DARKNET53_DEPTHS = (1, 2, 8, 8, 4)
LEAKY_SLOPE = 0.1
BN_MOMENTUM = 0.01
STRIDE = 32


def _darknet_init(root):
    """Darknet's initialisation: kernels U(-s, s) with s = sqrt(2 / fan_in), zero biases."""
    for module in root.modules():
        if isinstance(module, nn.Conv2d):
            fan_in = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
            bound = math.sqrt(2.0 / fan_in)
            nn.init.uniform_(module.weight, -bound, bound)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class ConvBNLeaky(nn.Module):
    """Darknet ``[convolutional]`` with ``batch_normalize=1`` and ``activation=leaky``."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.act = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DarkBlock(nn.Module):
    """1x1 halving and 3x3 convolutions with ``[shortcut] from=-3``."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNLeaky(channels, channels // 2, 1)
        self.conv2 = ConvBNLeaky(channels // 2, channels, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class DarkStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.conv_down = ConvBNLeaky(in_channels, out_channels, 3, stride=2)
        self.blocks = nn.Sequential(*[DarkBlock(out_channels) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.conv_down(x))


class Darknet53(nn.Module):
    """Layers 0-74; returns the stage outputs at strides 8, 16 and 32 (layers 36, 61, 74)."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(OrderedDict(conv1=ConvBNLeaky(in_channels, 32, 3)))
        stages, previous = [], 32
        for width, depth in zip(DARKNET53_WIDTHS, DARKNET53_DEPTHS):
            stages.append(DarkStage(previous, width, depth))
            previous = width
        self.stages = nn.Sequential(*stages)
        _darknet_init(self)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features[2:]


def darknet53_layout(in_channels=3):
    """(module path, in, out, kernel) of every Darknet-53 convolution in file order.

    This is the order of ``darknet53.cfg`` and of its ``.weights`` files; every
    convolution is followed by batch normalisation.
    """
    layout = [("stem.conv1", in_channels, 32, 3)]
    previous = 32
    for i, (width, depth) in enumerate(zip(DARKNET53_WIDTHS, DARKNET53_DEPTHS)):
        layout.append((f"stages.{i}.conv_down", previous, width, 3))
        for j in range(depth):
            layout.append((f"stages.{i}.blocks.{j}.conv1", width, width // 2, 1))
            layout.append((f"stages.{i}.blocks.{j}.conv2", width // 2, width, 3))
        previous = width
    return layout


def _five_convs(in_channels, channels):
    """Alternating 1x1/3x3 convolutions that open each YOLOv3 output branch."""
    return nn.Sequential(
        ConvBNLeaky(in_channels, channels, 1),
        ConvBNLeaky(channels, channels * 2, 3),
        ConvBNLeaky(channels * 2, channels, 1),
        ConvBNLeaky(channels, channels * 2, 3),
        ConvBNLeaky(channels * 2, channels, 1),
    )


def _upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class YOLOv3Neck(nn.Module):
    """Layers 75-79, 83-91 and 95-103: the top-down path down to the stride-8 branch."""

    def __init__(self):
        super().__init__()
        self.block5 = _five_convs(1024, 512)       # layers 75-79
        self.reduce5 = ConvBNLeaky(512, 256, 1)     # layer 84 (route 83 -> 79)
        self.block4 = _five_convs(256 + 512, 256)   # layers 87-91 (route 86: 85 + 61)
        self.reduce4 = ConvBNLeaky(256, 128, 1)     # layer 96 (route 95 -> 91)
        self.block3 = _five_convs(128 + 256, 128)   # layers 99-103 (route 98: 97 + 36)

    def forward(self, c3, c4, c5):
        p5 = self.block5(c5)
        p4 = self.block4(torch.cat((_upsample(self.reduce5(p5)), c4), dim=1))
        return self.block3(torch.cat((_upsample(self.reduce4(p4)), c3), dim=1))


class YOLOv3Head(nn.Module):
    """Layers 104-105 on the stride-8 grid; layer 105 outputs class scores only."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv = ConvBNLeaky(128, 256, 3)                   # layer 104
        self.classifier = nn.Conv2d(256, num_classes, 1)       # layer 105, linear

    def forward(self, x):
        return self.classifier(self.conv(x))


class YOLOv3Seg(nn.Module):
    def __init__(self, num_classes=6, in_channels=3, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        if not len(mean) == len(std) == in_channels:
            raise ValueError(
                f"YOLOv3 需要每个输入通道一个 MEAN/STD：IN_CHANNELS={in_channels}，"
                f"MEAN 有 {len(mean)} 项，STD 有 {len(std)} 项")
        self.backbone = Darknet53(in_channels)
        self.neck = YOLOv3Neck()
        self.head = YOLOv3Head(num_classes)
        _darknet_init(self.neck)
        _darknet_init(self.head)
        # x * std + mean recovers the [0, 1] RGB that the darknet weights were trained on.
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
                             persistent=False)
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
                             persistent=False)

    def pretrained_spec(self):
        return {"module": self.backbone, "arch": "darknet53", "allowed_missing": ()}

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("YOLOv3 expects [B,C,H,W]")
        height, width = x.shape[-2:]
        x = F.pad(x, (0, -width % STRIDE, 0, -height % STRIDE))
        x = x * self.input_std + self.input_mean
        logits = self.head(self.neck(*self.backbone(x)))
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return [logits[..., :height, :width]]


if __name__ == "__main__":
    model = YOLOv3Seg().eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 129, 161))
    print([tuple(output.shape) for output in outputs])
