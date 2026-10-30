# -*- coding: utf-8 -*-
"""ResNet feature extractor with torchvision-compatible parameter names.

The layout follows He et al., CVPR 2016 with the stride placed on the 3x3
convolution ("ResNet v1.5"), exactly as torchvision's ``resnet50``.  Keeping
torchvision's names lets the ImageNet checkpoint ``resnet50-0676ba61.pth``
load without key remapping (see ``common/pretrained.py``).

``output_stride`` 16 or 8 replaces the last strides by dilation, following
torchvision's ``replace_stride_with_dilation`` rule used by DeepLab.

``forward`` returns five feature maps for dense decoders:
stem (1/2, 64), layer1 (1/4, 256), layer2 (1/8, 512), layer3 (1/16, 1024)
and layer4 (1/32 or dilated, 2048).
"""
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 dilation=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    DEPTHS = {
        "resnet50": (3, 4, 6, 3),
        "resnet101": (3, 4, 23, 3),
    }
    # Which of layer2/3/4 trade their stride for dilation.
    DILATE = {
        32: (False, False, False),
        16: (False, False, True),
        8: (False, True, True),
    }

    def __init__(self, name="resnet50", in_channels=3, output_stride=32):
        super().__init__()
        if name not in self.DEPTHS:
            raise ValueError(f"未知 ResNet 编码器 {name!r}（可选: {', '.join(self.DEPTHS)}）")
        if output_stride not in self.DILATE:
            raise ValueError("ResNet output_stride must be 8, 16 or 32")
        depths = self.DEPTHS[name]
        dilate = self.DILATE[output_stride]
        self.name = name
        self.output_stride = output_stride
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, depths[0])
        self.layer2 = self._make_layer(128, depths[1], stride=2, dilate=dilate[0])
        self.layer3 = self._make_layer(256, depths[2], stride=2, dilate=dilate[1])
        self.layer4 = self._make_layer(512, depths[3], stride=2, dilate=dilate[2])
        self.channels = (64, 256, 512, 1024, 2048)
        self._initialize_weights()

    def _make_layer(self, channels, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        out_channels = channels * Bottleneck.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(self.inplanes, channels, stride, downsample,
                             previous_dilation)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, channels,
                                     dilation=self.dilation))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        stem = self.relu(self.bn1(self.conv1(x)))
        c1 = self.layer1(self.maxpool(stem))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [stem, c1, c2, c3, c4]
