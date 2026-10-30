# -*- coding: utf-8 -*-
"""DeepLabV3+ adapted to this repository's lightweight training workflow.

The decoder follows Chen et al. (ECCV 2018): ASPP emits 256 channels, the
1/4 low-level feature is projected to 48 channels, and two separable 3x3
convolutions refine their concatenation.  MobileNetV3-Large is used as the
engineering backbone so the implementation is practical on the existing
OpenEarthMap servers; this is documented as an adaptation rather than an exact
reproduction of the paper's modified Xception experiments.

``backbone="resnet50"`` swaps in an ImageNet-initialisable ResNet-50 (1/4
layer1 as the low-level feature, dilated layer4 as the ASPP input) while the
ASPP and decoder stay identical, so the two variants differ only in the
encoder.  The leaf uses ResNet-50.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenetv3 import ConvBNAct, MobileNetV3Backbone
from .resnet import ResNetBackbone


class SeparableConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 activation="relu"):
        padding = dilation * (kernel_size - 1) // 2
        act = nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                      dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act,
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if activation == "relu" else nn.Identity(),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18),
                 dropout=0.1):
        super().__init__()
        if len(rates) != 3 or any(rate < 1 for rate in rates):
            raise ValueError("ASPP_RATES must contain three positive integers")
        self.branches = nn.ModuleList([
            ConvBNAct(in_channels, out_channels, 1, activation="relu"),
            *[
                SeparableConvBNAct(in_channels, out_channels, 3, dilation=rate)
                for rate in rates
            ],
        ])
        # Global pooling deliberately omits BatchNorm, which is undefined for
        # a [1,C,1,1] tensor in training mode.
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            ConvBNAct(out_channels * 5, out_channels, 1, activation="relu"),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(
            pooled, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=9, in_channels=3, width_mult=1.0,
                 output_stride=16, aspp_channels=256, decoder_channels=256,
                 low_channels=48, aspp_rates=None, dropout=0.1,
                 backbone="mobilenetv3", mobilenet_bn_eps=1e-3):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("DeepLabV3+ output_stride must be 8 or 16")
        rates = tuple(aspp_rates or ((6, 12, 18) if output_stride == 16
                                     else (12, 24, 36)))
        self.backbone_name = backbone
        if backbone == "mobilenetv3":
            self.backbone = MobileNetV3Backbone(
                in_channels=in_channels,
                width_mult=width_mult,
                output_stride=output_stride,
                norm_eps=mobilenet_bn_eps,
            )
            backbone_low, backbone_high = (self.backbone.low_channels,
                                           self.backbone.high_channels)
        elif backbone in ResNetBackbone.DEPTHS:
            self.backbone = ResNetBackbone(backbone, in_channels, output_stride)
            backbone_low, backbone_high = (self.backbone.channels[1],
                                           self.backbone.channels[4])
        else:
            raise ValueError(
                f"DEEPLAB_BACKBONE 必须是 mobilenetv3 或 "
                f"{'/'.join(ResNetBackbone.DEPTHS)}，收到 {backbone!r}"
            )
        self.aspp = ASPP(backbone_high, aspp_channels, rates, dropout)
        self.low_project = ConvBNAct(
            backbone_low, low_channels, 1, activation="relu"
        )
        self.decoder = nn.Sequential(
            SeparableConvBNAct(aspp_channels + low_channels, decoder_channels),
            SeparableConvBNAct(decoder_channels, decoder_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )
        self._initialize_head()

    def _initialize_head(self):
        backbone_ids = {id(module) for module in self.backbone.modules()}
        for module in self.modules():
            if id(module) in backbone_ids:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def pretrained_spec(self):
        if self.backbone_name == "mobilenetv3":
            arch = self.backbone.pretrained_arch()
        else:
            arch = self.backbone_name
        return {"module": self.backbone, "arch": arch, "allowed_missing": ()}

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        if self.backbone_name == "mobilenetv3":
            low_feature, high_feature = features["low"], features["high"]
        else:
            low_feature, high_feature = features[1], features[4]
        high = self.aspp(high_feature)
        low = self.low_project(low_feature)
        high = F.interpolate(
            high, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.decoder(torch.cat([high, low], dim=1))
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        return [logits]


if __name__ == "__main__":
    model = DeepLabV3Plus(width_mult=0.75).eval()
    with torch.no_grad():
        result = model(torch.randn(1, 3, 129, 161))
    print([tuple(value.shape) for value in result])
