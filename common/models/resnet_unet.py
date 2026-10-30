# -*- coding: utf-8 -*-
"""U-Net and UNet++ decoders on a shared ImageNet ResNet encoder.

The pair is built as a controlled comparison: both models use the same
``ResNetBackbone``, the same per-level decoder widths, the same final
full-resolution block and the same classifier.  They differ only in the skip
topology:

* U-Net (Ronneberger et al., 2015): node X^{i} fuses encoder level i with the
  upsampled node below it.
* UNet++ (Zhou et al., 2018/2019): nested nodes X^{i,j} fuse every earlier
  node on level i with the upsampled X^{i+1,j-1}; X^{0,4} feeds the head.

Encoder levels i=0..4 are the ResNet stem (1/2) and layer1..layer4
(1/4..1/32).  ``decoder_channels`` is listed deepest first as in
segmentation_models.pytorch: widths for levels 3, 2, 1, 0 and the final
full-resolution block.  Upsampling is bilinear to the skip's exact size, so
any input of at least 32x32 pixels is accepted.

forward(x) returns [logits] at the input size without softmax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNetBackbone


class ConvBlock(nn.Sequential):
    """Two 3x3 Conv-BN-ReLU layers, as in the repository's U-Net blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def _up_to(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class _ResNetSkipDecoder(nn.Module):
    """Shared encoder, widths, final block and head of the two skip designs."""

    def __init__(self, num_classes, in_channels, encoder, decoder_channels):
        super().__init__()
        decoder_channels = tuple(int(c) for c in decoder_channels)
        if len(decoder_channels) != 5 or min(decoder_channels) < 1:
            raise ValueError("UNET_DECODER_CHANNELS 必须是 5 个正整数（由深到浅）")
        self.backbone = ResNetBackbone(encoder, in_channels, output_stride=32)
        self.encoder_name = encoder
        # level 0 (1/2) .. level 3 (1/16)
        self.level_channels = tuple(reversed(decoder_channels[:4]))
        self.final = ConvBlock(self.level_channels[0], decoder_channels[4])
        self.classifier = nn.Conv2d(decoder_channels[4], num_classes, 3, padding=1)

    def _initialize_decoder(self):
        backbone_ids = {id(module) for module in self.backbone.modules()}
        for module in self.modules():
            if id(module) in backbone_ids:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def pretrained_spec(self):
        return {"module": self.backbone, "arch": self.encoder_name,
                "allowed_missing": ()}

    def _head(self, x, input_size):
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return [self.classifier(self.final(x))]


class ResNetUNet(_ResNetSkipDecoder):
    def __init__(self, num_classes=6, in_channels=3, encoder="resnet50",
                 decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__(num_classes, in_channels, encoder, decoder_channels)
        enc = self.backbone.channels
        width = self.level_channels
        self.blocks = nn.ModuleList([
            ConvBlock(enc[i] + (enc[4] if i == 3 else width[i + 1]), width[i])
            for i in range(4)
        ])
        self._initialize_decoder()

    def forward(self, x):
        features = self.backbone(x)
        y = features[4]
        for i in (3, 2, 1, 0):
            y = self.blocks[i](torch.cat([features[i], _up_to(y, features[i])], dim=1))
        return self._head(y, x.shape[-2:])


class ResNetUNetPlusPlus(_ResNetSkipDecoder):
    def __init__(self, num_classes=6, in_channels=3, encoder="resnet50",
                 decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__(num_classes, in_channels, encoder, decoder_channels)
        enc = self.backbone.channels
        width = self.level_channels
        self.nodes = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                below = enc[i + 1] if j == 1 else width[i + 1]
                in_ch = enc[i] + (j - 1) * width[i] + below
                self.nodes[f"x{i}_{j}"] = ConvBlock(in_ch, width[i])
        self._initialize_decoder()

    def forward(self, x):
        features = self.backbone(x)
        nodes = {(i, 0): feature for i, feature in enumerate(features)}
        for j in range(1, 5):
            for i in range(0, 5 - j):
                inputs = [nodes[(i, k)] for k in range(j)]
                inputs.append(_up_to(nodes[(i + 1, j - 1)], features[i]))
                nodes[(i, j)] = self.nodes[f"x{i}_{j}"](torch.cat(inputs, dim=1))
        return self._head(nodes[(0, 4)], x.shape[-2:])
