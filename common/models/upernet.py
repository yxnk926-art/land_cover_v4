# -*- coding: utf-8 -*-
"""Swin-T + UperNet (Xiao et al., ECCV 2018) segmentation model.

The head follows the mmsegmentation ``UPerHead`` used in the Swin paper's
ADE20K experiments: a pyramid pooling module (PPM, scales 1/2/3/6) on the last
stage, 1x1 lateral convolutions, a top-down FPN with 3x3 smoothing, fusion of
all levels at 1/4 and a 512-channel 3x3 bottleneck.  An FCN auxiliary head on
stage 3 (weight 0.4 via ``DEEP_SUP_WEIGHTS``) is used during training only.

One deliberate deviation: the 1x1 global-pooling branch of the PPM has no
BatchNorm.  With a per-GPU batch of 8 its statistics would come from 8 values
per channel, the pattern suspected of destabilising PIDNet validation; the
repository's DeepLabV3+ ASPP pooling branch omits BatchNorm for the same
reason.  The 2x2/3x3/6x6 branches keep Conv-BN-ReLU.

forward(x) returns [main, aux] in training mode and [main] otherwise, all at
the input size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin import SwinTransformer


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def _resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        branches = []
        for scale in pool_scales:
            if scale == 1:
                branches.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, channels, 1),
                    nn.ReLU(inplace=True),
                ))
            else:
                branches.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvBNReLU(in_channels, channels, 1),
                ))
        self.branches = nn.ModuleList(branches)
        self.bottleneck = ConvBNReLU(in_channels + len(pool_scales) * channels,
                                     channels, 3)

    def forward(self, x):
        size = x.shape[-2:]
        pooled = [_resize(branch(x), size) for branch in self.branches]
        return self.bottleneck(torch.cat([x, *pooled], dim=1))


class UperHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes,
                 pool_scales=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        self.ppm = PyramidPooling(in_channels[-1], channels, pool_scales)
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(c, channels, 1) for c in in_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(channels, channels, 3) for _ in in_channels[:-1]
        ])
        self.fpn_bottleneck = ConvBNReLU(len(in_channels) * channels, channels, 3)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features[:-1])]
        laterals.append(self.ppm(features[-1]))
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + _resize(
                laterals[index], laterals[index - 1].shape[-2:]
            )
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals[:-1])]
        outputs.append(laterals[-1])
        size = outputs[0].shape[-2:]
        fused = torch.cat([outputs[0], *[_resize(o, size) for o in outputs[1:]]], dim=1)
        return self.classifier(self.dropout(self.fpn_bottleneck(fused)))


class FCNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, dropout=0.1):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, channels, 3)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        return self.classifier(self.dropout(self.conv(x)))


class SwinUperNet(nn.Module):
    def __init__(self, num_classes=6, in_channels=3, variant="tiny",
                 window_size=7, drop_path_rate=0.3, channels=512,
                 pool_scales=(1, 2, 3, 6), dropout=0.1, auxiliary=True,
                 aux_channels=256):
        super().__init__()
        self.backbone = SwinTransformer(variant, in_channels, window_size,
                                        drop_path_rate)
        self.decode_head = UperHead(self.backbone.channels, channels, num_classes,
                                    tuple(pool_scales), dropout)
        self.aux_head = (FCNHead(self.backbone.channels[2], aux_channels,
                                 num_classes, dropout) if auxiliary else None)
        self._initialize_heads()

    def _initialize_heads(self):
        heads = [self.decode_head] + ([self.aux_head] if self.aux_head else [])
        for head in heads:
            for module in head.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def pretrained_spec(self):
        return {"module": self.backbone, "arch": f"swin_{self.backbone.variant}",
                "allowed_missing": ("out_norms.",)}

    def no_weight_decay_keywords(self):
        return self.backbone.no_weight_decay_keywords()

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        main = _resize(self.decode_head(features), size)
        if self.aux_head is None or not self.training:
            return [main]
        return [main, _resize(self.aux_head(features[2]), size)]
