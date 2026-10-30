# -*- coding: utf-8 -*-
"""MobileNetV3-Large with the Lite R-ASPP segmentation head.

Stage C 修复（本轮）
====================================================================
原实现在 ``_context_gate`` 里按 **特征图尺寸** 在两条池化路径之间静默切换：

    if min(H, W) >= 49:  F.avg_pool2d(high, kernel_size=49, stride=(16,20))
    else:                F.adaptive_avg_pool2d(high, 1)

而训练与验证的输入尺寸不同（训练 crop=768 → 1/16 特征图 48x48；
验证整图 1024 → 64x64），于是：

    输入 512   → 32x32 → min=32 < 49 → 走 adaptive（全局）
    输入 768   → 48x48 → min=48 < 49 → 走 adaptive（全局）   ← 训练走这条
    输入 1024  → 64x64 → min=64 >= 49 → 走 fixed 49x49       ← 验证走这条

即**训练与推理走了不同的代码路径**。更糟的是 fixed 分支里
``stride=(16,20)`` 在 64x64 上只输出 1x1，平均的是**左上角 49x49 窗口**
（占 58.6% 面积，且偏向左上角），与全局平均不是同一个量。

修复策略
--------------------------------------------------------------------
1. 默认改为**始终使用全局自适应池化**（``context_pool="adaptive"``）。
   这与 torchvision 官方 ``LRASPPHead`` 的做法一致（该实现用的就是
   ``AdaptiveAvgPool2d(1)``），尺寸无关，训练/推理行为完全一致。
2. 论文的固定 49x49 核保留为显式选项 ``context_pool="fixed"``；
   但当特征图小于 49 时**直接抛错**，而不是静默降级——静默降级正是这个
   bug 的成因。
3. ``output_stride`` 与 ``context_pool`` 都由模型工厂显式传入，避免配置里
   声明了参数、实际模型却继续使用硬编码值。
4. 不新增任何可学习参数、不改动任何既有参数名 → **旧 checkpoint 仍可加载**。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenetv3 import ConvBNAct, MobileNetV3Backbone


class MobileNetV3LRASPP(nn.Module):
    """Efficient full-resolution semantic segmentation for OpenEarthMap.

    ``forward`` returns ``[logits]`` to match the repository-wide model
    contract.  The skip classifier uses the expanded 1/8 feature selected by
    Google's later DeepLab integration, while the high-level branch follows
    the context gate in the paper's Figure 10.

    ``context_pool``
        ``"adaptive"``（默认）: 全局自适应平均池化，训练/推理行为一致，
        与 torchvision 的 LRASPPHead 相同。
        ``"fixed"``: 论文的 49x49 固定核 + stride (16,20)；要求特征图
        ``min(H, W) >= 49``，否则抛 ``RuntimeError``。
    """

    def __init__(self, num_classes=9, in_channels=3, width_mult=1.0,
                 output_stride=16, head_channels=128, dropout=0.0,
                 context_pool="adaptive", reduced_tail=True, norm_eps=1e-3):
        super().__init__()
        if num_classes < 1 or head_channels < 1:
            raise ValueError("num_classes and head_channels must be positive")
        if context_pool not in ("adaptive", "fixed"):
            raise ValueError(
                f"context_pool 必须是 'adaptive' 或 'fixed'，收到 {context_pool!r}"
            )
        # The paper's LR-ASPP halves the last stage (reduced tail); ImageNet
        # weights exist only for the full tail, as in torchvision's LR-ASPP.
        self.backbone = MobileNetV3Backbone(
            in_channels=in_channels,
            width_mult=width_mult,
            output_stride=output_stride,
            reduced_tail=reduced_tail,
            norm_eps=norm_eps,
        )
        self.high_proj = ConvBNAct(
            self.backbone.high_channels, head_channels, kernel_size=1,
            activation="relu"
        )
        # No BN after 1x1 global pooling: it would fail for train batch size 1.
        self.context_conv = nn.Conv2d(
            self.backbone.high_channels, head_channels, 1
        )
        self.context_activation = nn.Sigmoid()
        self.output_stride = output_stride
        self.context_pool = context_pool
        self.context_pool_kernel = 49
        self.context_pool_stride = (16, 20)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.high_classifier = nn.Conv2d(head_channels, num_classes, 1)
        self.low_classifier = nn.Conv2d(
            self.backbone.low8_expanded_channels, num_classes, 1
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
        return {"module": self.backbone, "arch": self.backbone.pretrained_arch(),
                "allowed_missing": ()}

    def _context_gate(self, high):
        """全局上下文门控。训练与推理必须走同一条路径。"""
        height, width = high.shape[-2:]
        if self.context_pool == "fixed":
            if min(height, width) < self.context_pool_kernel:
                raise RuntimeError(
                    f"context_pool='fixed' 需要特征图 min(H, W) >= "
                    f"{self.context_pool_kernel}，当前为 {height}x{width}"
                    f"（按 output_stride={self.output_stride}，对应输入约 "
                    f"{height * self.output_stride}x{width * self.output_stride}）。"
                    f"请改用 context_pool='adaptive'（默认，尺寸无关），"
                    f"或把 CROP_SIZE 提到至少 "
                    f"{self.context_pool_kernel * self.output_stride}。"
                )
            context = F.avg_pool2d(
                high,
                kernel_size=self.context_pool_kernel,
                stride=self.context_pool_stride,
            )
        else:
            context = F.adaptive_avg_pool2d(high, 1)
        context = self.context_activation(self.context_conv(context))
        return F.interpolate(
            context, size=(height, width), mode="bilinear",
            align_corners=False
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        high = self.high_proj(features["high"])
        high = self.dropout(high * self._context_gate(features["high"]))
        high = self.high_classifier(high)
        high = F.interpolate(
            high, size=features["low8_expanded"].shape[-2:], mode="bilinear",
            align_corners=False
        )
        logits = high + self.low_classifier(features["low8_expanded"])
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        return [logits]


if __name__ == "__main__":
    # 自检 1：默认（adaptive）在训练与验证尺寸下必须走同一条路径
    model = MobileNetV3LRASPP(width_mult=0.75).eval()
    with torch.no_grad():
        for size in (512, 768, 1024):
            out = model(torch.randn(1, 3, size, size))
            print(f"[adaptive] {size:>5} -> {[tuple(v.shape) for v in out]}")

    # 自检 2：显式 fixed 在小输入下必须直接报错，而不是静默降级
    strict = MobileNetV3LRASPP(width_mult=0.75, context_pool="fixed").eval()
    with torch.no_grad():
        try:
            strict(torch.randn(1, 3, 768, 768))
            print("[fixed] 768 未报错 —— 与预期不符")
        except RuntimeError as exc:
            print(f"[fixed] 768 正确拒绝: {str(exc)[:60]}...")
        out = strict(torch.randn(1, 3, 1024, 1024))
        print(f"[fixed] 1024 -> {[tuple(v.shape) for v in out]}")
