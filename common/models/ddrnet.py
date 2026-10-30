# -*- coding: utf-8 -*-
"""DDRNet-23-slim (Hong et al., "Deep Dual-resolution Networks", 2021).

Follows the authors' released ``DualResNet`` (github.com/ydhongHIT/DDRNet):
a low-resolution context branch (layer1..layer5 + DAPPM) and a 1/8
high-resolution branch (layer3_..layer5_) exchanging information through
bilateral fusion (down3/down4 and compression3/compression4).  Parameter
names of the shared trunk match the authors' ImageNet checkpoint
``DDRNet23s_imagenet.pth`` so it loads without remapping; that checkpoint's
classifier (down5, last_layer, linear) is unused.

DDRNet is the dual-branch model that PIDNet extends with a boundary (D)
branch; the residual blocks, DAPPM and segmentation head are reused from
``pidnet.py`` so the pair differs only in architecture, not implementation.
Head modules are named like the repository's PIDNet (``final_head``,
``aux_head``) rather than the release's ``final_layer``/``seghead_extra``;
heads are always trained from scratch.

forward(x) returns [main, aux] in training mode with ``augment=True`` and
[main] otherwise, all resized to the input size.
"""
import torch.nn as nn
import torch.nn.functional as F

from .pidnet import BN_MOMENTUM, BasicBlock, Bottleneck, DAPPM, PIDNet, SegHead


class DDRNet(nn.Module):
    VARIANTS = {
        "23_slim": dict(layers=(2, 2, 2, 2), planes=32, spp_planes=128, head_planes=64),
        "23": dict(layers=(2, 2, 2, 2), planes=64, spp_planes=128, head_planes=128),
    }

    def __init__(self, num_classes=6, in_channels=3, variant="23_slim",
                 augment=True):
        super().__init__()
        variant = variant.lower().replace("ddrnet-", "").replace("ddrnet_", "")
        if variant not in self.VARIANTS:
            raise ValueError("DDRNET_VARIANT must be '23_slim' or '23'")
        spec = self.VARIANTS[variant]
        layers, planes = spec["layers"], spec["planes"]
        high = planes * 2
        make = PIDNet._make_layer
        self.variant = variant
        self.augment = augment
        # Not in-place: stage outputs are reused by both branches.
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, planes, 3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.layer1 = make(BasicBlock, planes, planes, layers[0])
        self.layer2 = make(BasicBlock, planes, planes * 2, layers[1], stride=2)
        self.layer3 = make(BasicBlock, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = make(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, high, 1, bias=False),
            nn.BatchNorm2d(high, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, high, 1, bias=False),
            nn.BatchNorm2d(high, momentum=BN_MOMENTUM),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(high, planes * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(high, planes * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 8, momentum=BN_MOMENTUM),
        )
        self.layer3_ = make(BasicBlock, planes * 2, high, 2)
        self.layer4_ = make(BasicBlock, high, high, 2)
        self.layer5_ = make(Bottleneck, high, high, 1)
        self.layer5 = make(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = DAPPM(planes * 16, spec["spp_planes"], planes * 4)
        self.final_head = SegHead(planes * 4, spec["head_planes"], num_classes)
        self.aux_head = (SegHead(high, spec["head_planes"], num_classes)
                         if augment else None)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def pretrained_spec(self):
        return {"module": self, "arch": f"ddrnet{self.variant}",
                "allowed_missing": ("spp.", "final_head.", "aux_head.")}

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.conv1(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(self.relu(layer1))
        layer3 = self.layer3(self.relu(layer2))
        high_size = layer2.shape[-2:]  # 1/8

        x_high = self.layer3_(self.relu(layer2))
        x_low = layer3 + self.down3(self.relu(x_high))
        x_high = x_high + self._resize(self.compression3(self.relu(layer3)), high_size)
        auxiliary = x_high

        layer4 = self.layer4(self.relu(x_low))
        x_high = self.layer4_(self.relu(x_high))
        x_low = layer4 + self.down4(self.relu(x_high))
        x_high = x_high + self._resize(self.compression4(self.relu(layer4)), high_size)

        x_high = self.layer5_(self.relu(x_high))
        context = self._resize(self.spp(self.layer5(self.relu(x_low))), high_size)
        main = self._resize(self.final_head(context + x_high), input_size)
        if self.aux_head is None or not self.training:
            return [main]
        return [main, self._resize(self.aux_head(auxiliary), input_size)]
