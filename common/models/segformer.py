# -*- coding: utf-8 -*-
"""Self-contained SegFormer (MiT encoder + all-MLP decoder).

This implementation mirrors the official B0-B5 stage definitions without
depending on mmcv, mmsegmentation or timm.  It keeps the repository contract:
``forward`` returns a list whose first element is full-resolution logits.
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast_off(device_type):
    """Run numerically sensitive attention operations outside AMP autocast."""
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except (AttributeError, TypeError):
        if device_type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return contextlib.nullcontext()


SEGFORMER_VARIANTS = {
    "b0": dict(dims=(32, 64, 160, 256), depths=(2, 2, 2, 2), decoder_dim=256),
    "b1": dict(dims=(64, 128, 320, 512), depths=(2, 2, 2, 2), decoder_dim=256),
    "b2": dict(dims=(64, 128, 320, 512), depths=(3, 4, 6, 3), decoder_dim=768),
    "b3": dict(dims=(64, 128, 320, 512), depths=(3, 4, 18, 3), decoder_dim=768),
    "b4": dict(dims=(64, 128, 320, 512), depths=(3, 8, 27, 3), decoder_dim=768),
    "b5": dict(dims=(64, 128, 320, 512), depths=(3, 6, 40, 3), decoder_dim=768),
}


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size, stride,
            padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        # q/k matmul is the numerically sensitive part of MiT under CUDA AMP.
        # Keep the complete attention core in FP32 so the unscaled score matrix
        # cannot overflow FP16 before ``self.scale`` is applied.  The enclosing
        # Transformer block and the rest of SegFormer still use AMP normally.
        with _autocast_off(x.device.type):
            x = x.float()
            q = self.q(x).reshape(
                batch, tokens, self.num_heads, channels // self.num_heads
            ).permute(0, 2, 1, 3)
            if self.sr_ratio > 1:
                if min(height, width) < self.sr_ratio:
                    raise ValueError(
                        f"feature map {height}x{width} is smaller than sr_ratio "
                        f"{self.sr_ratio}; use an input of at least 32x32"
                    )
                reduced = x.transpose(1, 2).reshape(
                    batch, channels, height, width
                )
                reduced = self.sr(reduced).reshape(
                    batch, channels, -1
                ).transpose(1, 2)
                reduced = self.norm(reduced)
            else:
                reduced = x
            kv = self.kv(reduced).reshape(
                batch, -1, 2, self.num_heads, channels // self.num_heads
            ).permute(2, 0, 3, 1, 4)
            key, value = kv[0], kv[1]
            # Scale q before the matrix multiplication rather than scaling an
            # already-materialized score matrix.
            attention = (q * self.scale) @ key.transpose(-2, -1)
            attention = self.attn_drop(attention.softmax(dim=-1))
            out = (attention @ value).transpose(1, 2).reshape(
                batch, tokens, channels
            )
            return self.proj_drop(self.proj(out))


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.depthwise = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim
        )
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, height, width):
        batch, _, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = self.depthwise(x).flatten(2).transpose(1, 2)
        x = self.dropout(self.activation(x))
        return self.dropout(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio=4.0,
                 dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = EfficientSelfAttention(
            dim, num_heads, sr_ratio, qkv_bias=True,
            attn_drop=attn_dropout, proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        x = x + self.drop_path(self.ffn(self.norm2(x), height, width))
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, in_channels, dims, depths, drop_path_rate=0.1,
                 dropout=0.0, attn_dropout=0.0):
        super().__init__()
        heads = (1, 2, 5, 8)
        sr_ratios = (8, 4, 2, 1)
        patch_kernels = (7, 3, 3, 3)
        patch_strides = (4, 2, 2, 2)
        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        for stage in range(4):
            input_dim = in_channels if stage == 0 else dims[stage - 1]
            self.patch_embeds.append(OverlapPatchEmbed(
                input_dim, dims[stage], patch_kernels[stage], patch_strides[stage]
            ))
            blocks = []
            for index in range(depths[stage]):
                blocks.append(TransformerBlock(
                    dims[stage], heads[stage], sr_ratios[stage],
                    mlp_ratio=4.0, dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path_rate=rates[offset + index],
                ))
            offset += depths[stage]
            self.stages.append(nn.ModuleList(blocks))
            self.norms.append(nn.LayerNorm(dims[stage], eps=1e-6))
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1]
                fan_out *= module.out_channels / module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        outputs = []
        for patch_embed, blocks, norm in zip(
                self.patch_embeds, self.stages, self.norms):
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            batch = x.shape[0]
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs


class LinearEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        batch, _, height, width = x.shape
        x = self.proj(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(batch, -1, height, width)


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, decoder_dim, num_classes, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([
            LinearEmbedding(channels, decoder_dim) for channels in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, features):
        target_size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear",
                    align_corners=False
                )
            projected.append(feature)
        # The official decoder concatenates from F4 to F1.
        fused = self.fuse(torch.cat(list(reversed(projected)), dim=1))
        return self.classifier(self.dropout(fused))


class SegFormer(nn.Module):
    def __init__(self, num_classes=9, in_channels=3, variant="b0",
                 decoder_dim=None, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        variant = variant.lower()
        if variant not in SEGFORMER_VARIANTS:
            raise ValueError(
                f"unknown SegFormer variant {variant!r}; expected one of "
                f"{sorted(SEGFORMER_VARIANTS)}"
            )
        spec = SEGFORMER_VARIANTS[variant]
        dims, depths = spec["dims"], spec["depths"]
        decoder_dim = spec["decoder_dim"] if decoder_dim is None else decoder_dim
        self.variant = variant
        self.backbone = MixVisionTransformer(
            in_channels, dims, depths, drop_path_rate=drop_path_rate
        )
        self.decode_head = SegFormerHead(
            dims, decoder_dim, num_classes, dropout=dropout
        )
        self._initialize_head()

    def _initialize_head(self):
        for module in self.decode_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def pretrained_spec(self):
        return {"module": self.backbone, "arch": f"mit_{self.variant}",
                "allowed_missing": ()}

    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 32:
            raise ValueError("SegFormer expects [B,C,H,W] with H and W >= 32")
        input_size = x.shape[-2:]
        logits = self.decode_head(self.backbone(x))
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        return [logits]


if __name__ == "__main__":
    model = SegFormer(variant="b0").eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 129, 161))
    print([tuple(output.shape) for output in outputs])
