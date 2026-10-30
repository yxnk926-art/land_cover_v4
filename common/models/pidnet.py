# -*- coding: utf-8 -*-
"""PIDNet adapted for OpenEarthMap and the repository's structured loss.

The P (detail), I (context) and D (boundary) branches follow Xu et al., CVPR
2023 and the authors' released implementation.  Training returns a structured
mapping so the one-channel boundary output cannot accidentally be treated as
a nine-class semantic head.  Every returned tensor is resized to the input
size; ``main`` is the only output used for validation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1


class SafeBatchNorm2d(nn.BatchNorm2d):
    """Use running statistics for a single pooled value in training mode.

    PIDNet's official PAPPM/DAPPM places BatchNorm after global pooling.  A
    one-image debug batch therefore contains only one value per channel and
    ordinary BatchNorm raises.  Default multi-image training is unchanged;
    only that degenerate debug case falls back to the stored statistics.
    """

    def forward(self, x):
        values_per_channel = x.numel() // x.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps
            )
        return super().forward(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 2, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + residual
        return out if self.no_relu else self.relu(out)


class SegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        return self.classifier(self.relu(self.bn2(x)))


class DAPPM(nn.Module):
    """Deep aggregation pyramid pooling, used by PIDNet-L."""

    def __init__(self, in_channels, branch_channels, out_channels):
        super().__init__()
        self.scale0 = self._scale(in_channels, branch_channels, None)
        self.scale1 = self._scale(in_channels, branch_channels, (5, 2, 2))
        self.scale2 = self._scale(in_channels, branch_channels, (9, 4, 4))
        self.scale3 = self._scale(in_channels, branch_channels, (17, 8, 8))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SafeBatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
        )
        self.processes = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(branch_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels, branch_channels, 3, padding=1,
                          bias=False),
            ) for _ in range(4)
        ])
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_channels * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels * 5, out_channels, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    @staticmethod
    def _scale(in_channels, branch_channels, pool):
        modules = []
        if pool is not None:
            kernel, stride, padding = pool
            modules.append(nn.AvgPool2d(kernel, stride, padding))
        modules.extend([
            SafeBatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
        ])
        return nn.Sequential(*modules)

    def forward(self, x):
        size = x.shape[-2:]
        outputs = [self.scale0(x)]
        scales = [self.scale1, self.scale2, self.scale3, self.scale4]
        for scale, process in zip(scales, self.processes):
            value = F.interpolate(
                scale(x), size=size, mode="bilinear", align_corners=False
            )
            outputs.append(process(value + outputs[-1]))
        return self.compression(torch.cat(outputs, dim=1)) + self.shortcut(x)


class PAPPM(nn.Module):
    """Parallel aggregation pyramid pooling, used by PIDNet-S/M."""

    def __init__(self, in_channels, branch_channels, out_channels):
        super().__init__()
        self.scale0 = DAPPM._scale(in_channels, branch_channels, None)
        self.scale1 = DAPPM._scale(in_channels, branch_channels, (5, 2, 2))
        self.scale2 = DAPPM._scale(in_channels, branch_channels, (9, 4, 4))
        self.scale3 = DAPPM._scale(in_channels, branch_channels, (17, 8, 8))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SafeBatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
        )
        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(branch_channels * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels * 4, branch_channels * 4, 3, padding=1,
                      groups=4, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_channels * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels * 5, out_channels, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        size = x.shape[-2:]
        base = self.scale0(x)
        values = []
        for scale in (self.scale1, self.scale2, self.scale3, self.scale4):
            value = F.interpolate(
                scale(x), size=size, mode="bilinear", align_corners=False
            )
            values.append(value + base)
        processed = self.scale_process(torch.cat(values, dim=1))
        return self.compression(torch.cat([base, processed], dim=1)) + self.shortcut(x)


class PagFM(nn.Module):
    """Pixel-attention-guided fusion: gate=1 selects the I branch."""

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.f_p = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.f_i = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )

    def forward(self, p, i):
        size = p.shape[-2:]
        i_query = F.interpolate(
            self.f_i(i), size=size, mode="bilinear", align_corners=False
        )
        p_key = self.f_p(p)
        gate = torch.sigmoid((p_key * i_query).sum(dim=1, keepdim=True))
        i = F.interpolate(i, size=size, mode="bilinear", align_corners=False)
        return (1.0 - gate) * p + gate * i


class Bag(nn.Module):
    """Boundary-attention-guided fusion: gate=1 selects P/detail."""

    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, p, i, d):
        gate = torch.sigmoid(d)
        return self.project(gate * p + (1.0 - gate) * i)


class LightBag(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.p_project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.i_project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, p, i, d):
        gate = torch.sigmoid(d)
        p_add = self.p_project((1.0 - gate) * i + p)
        i_add = self.i_project(i + gate * p)
        return p_add + i_add


class PIDNet(nn.Module):
    VARIANTS = {
        "s": dict(m=2, n=3, planes=32, ppm_planes=96, head_planes=128),
        "m": dict(m=2, n=3, planes=64, ppm_planes=96, head_planes=128),
        "l": dict(m=3, n=4, planes=64, ppm_planes=112, head_planes=256),
    }

    def __init__(self, num_classes=9, in_channels=3, variant="s",
                 augment=True):
        super().__init__()
        variant = variant.lower().replace("pidnet-", "").replace("pidnet_", "")
        if variant not in self.VARIANTS:
            raise ValueError("PIDNET_VARIANT must be 's', 'm' or 'l'")
        spec = self.VARIANTS[variant]
        m, n = spec["m"], spec["n"]
        planes = spec["planes"]
        ppm_planes = spec["ppm_planes"]
        head_planes = spec["head_planes"]
        self.variant = variant
        self.augment = augment
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, planes, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        # I branch.
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2,
                                       stride=2)

        # P branch.
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        # D branch.  S/M use the reduced detail width from the official model.
        if m == 2:
            self.layer3_d = self._make_single_layer(
                BasicBlock, planes * 2, planes
            )
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4)
        else:
            self.layer3_d = self._make_single_layer(
                BasicBlock, planes * 2, planes * 2
            )
            self.layer4_d = self._make_single_layer(
                BasicBlock, planes * 2, planes * 2
            )
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4)
        self.layer5_d = self._make_layer(
            Bottleneck, planes * 2, planes * 2, 1
        )

        self.final_head = SegHead(planes * 4, head_planes, num_classes)
        if augment:
            self.aux_head = SegHead(planes * 2, head_planes, num_classes)
            self.boundary_head = SegHead(planes * 2, planes, 1)
        self._initialize_weights()

    @staticmethod
    def _make_layer(block, in_channels, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion,
                               momentum=BN_MOMENTUM),
            )
        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for index in range(1, blocks):
            layers.append(block(
                in_channels, channels, no_relu=index == blocks - 1
            ))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_single_layer(block, in_channels, channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion,
                               momentum=BN_MOMENTUM),
            )
        return block(in_channels, channels, stride, downsample, no_relu=True)

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
        # The authors' ImageNet model trains the I and P branches with a
        # classifier on layer5; the D branch, PAPPM/DAPPM, the fusion module
        # and the segmentation heads start from scratch, as in the official
        # (non-strict) loading.
        return {"module": self, "arch": f"pidnet_{self.variant}",
                "allowed_missing": ("layer3_d.", "layer4_d.", "layer5_d.",
                                    "diff3.", "diff4.", "spp.", "dfm.",
                                    "final_head.", "aux_head.", "boundary_head.")}

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.layer1(self.conv1(x))
        x = self.relu(self.layer2(self.relu(x)))

        p = self.layer3_p(x)
        d = self.layer3_d(x)
        detail_size = p.shape[-2:]

        x = self.relu(self.layer3(x))
        p = self.pag3(p, self.compression3(x))
        d = d + self._resize(self.diff3(x), detail_size)
        aux_feature = p

        x = self.relu(self.layer4(x))
        p = self.layer4_p(self.relu(p))
        d = self.layer4_d(self.relu(d))
        p = self.pag4(p, self.compression4(x))
        d = d + self._resize(self.diff4(x), detail_size)
        boundary_feature = d

        p = self.layer5_p(self.relu(p))
        d = self.layer5_d(self.relu(d))
        context = self.spp(self.layer5(x))
        context = self._resize(context, p.shape[-2:])
        main = self.final_head(self.dfm(p, context, d))
        main = self._resize(main, input_size)

        # Auxiliary semantic/boundary heads are training-only in the paper.
        if not self.augment or not self.training:
            return [main]
        auxiliary = self._resize(self.aux_head(aux_feature), input_size)
        boundary = self._resize(
            self.boundary_head(boundary_feature), input_size
        )
        return {"main": main, "aux": [auxiliary], "boundary": boundary}


if __name__ == "__main__":
    model = PIDNet(variant="s").eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 129, 161))
    print([tuple(value.shape) for value in outputs])
