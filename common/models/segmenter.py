# -*- coding: utf-8 -*-
"""Segmenter (Strudel et al., ICCV 2021): plain ViT encoder + mask transformer.

Self-contained (no timm/mmcv at runtime), following rstrudel/segmenter.  Encoder
parameter names follow timm, so ``timm/deit_small_patch16_224.fb_in1k`` loads
once its position grid is resampled; tests/test_v4dev_backbones.py checks
numerical equality with timm on shared weights.

Unlike the hierarchical MiT and Swin encoders, the ViT keeps a single 1/16 token
grid and every block attends over all tokens.  The decoder passes K learned
class embeddings through two more blocks together with the patch tokens; the
mask of a class is the scalar product of its L2-normalised embedding with the
L2-normalised patch tokens.  The masks are ordinary per-pixel logits trained
with cross-entropy, so there is no bipartite matching.

Differences from the official code:

* the stored position grid is fixed at 32x32 (512 px) whatever the crop size;
  other grids are resampled in ``forward``, as the official code does for
  inference sizes;
* the scalar product runs outside AMP autocast, see ``MaskTransformer.forward``.

``forward`` returns a one-element list of full-resolution logits.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer import _autocast_off
from .swin import DropPath


SEGMENTER_VARIANTS = {
    "s16": dict(arch="deit_small_patch16", embed_dim=384, depth=12, num_heads=6,
                patch_size=16),
}
POS_GRID = 32  # 512 px at patch size 16


def resize_pos_embed(pos_embed, grid):
    """Resample the patch part of ``pos_embed``; the class token keeps its own."""
    old = math.isqrt(pos_embed.shape[1] - 1)
    if (old, old) == tuple(grid):
        return pos_embed
    cls_pos, patch_pos = pos_embed[:, :1], pos_embed[:, 1:]
    patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=tuple(grid), mode="bilinear",
                              align_corners=False)
    return torch.cat((cls_pos, patch_pos.flatten(2).transpose(1, 2)), dim=1)


def _initialize_weights(root):
    for module in root.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, channels)))


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, drop_path=0.0, norm_eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, 4 * dim, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """Returns the patch tokens (class token removed) and their (height, width) grid."""

    def __init__(self, in_channels=3, embed_dim=384, depth=12, num_heads=6, patch_size=16,
                 pos_grid=POS_GRID, drop_path_rate=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + pos_grid ** 2, embed_dim))
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, drop_path=rate, norm_eps=1e-6) for rate in rates
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        _initialize_weights(self)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        grid = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + resize_pos_embed(self.pos_embed, grid)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 1:], grid


class MaskTransformer(nn.Module):
    def __init__(self, num_classes, dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.proj_dec = nn.Linear(dim, dim)
        self.cls_emb = nn.Parameter(torch.zeros(1, num_classes, dim))
        self.blocks = nn.ModuleList([
            Block(dim, dim // 64, dropout=dropout) for _ in range(num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(dim)
        self.proj_patch = nn.Parameter(torch.randn(dim, dim) * dim ** -0.5)
        self.proj_classes = nn.Parameter(torch.randn(dim, dim) * dim ** -0.5)
        self.mask_norm = nn.LayerNorm(num_classes)
        _initialize_weights(self)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, tokens, grid):
        x = self.proj_dec(tokens)
        x = torch.cat((x, self.cls_emb.expand(x.shape[0], -1, -1)), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.decoder_norm(x)
        patches, classes = x[:, :-self.num_classes], x[:, -self.num_classes:]
        # Cosine similarities differ little between classes, and mask_norm then
        # scales those differences up, so half precision would coarsen the logits.
        with _autocast_off(x.device.type):
            patches = F.normalize(patches.float() @ self.proj_patch, dim=-1)
            classes = F.normalize(classes.float() @ self.proj_classes, dim=-1)
            masks = self.mask_norm(patches @ classes.transpose(1, 2))
        return masks.transpose(1, 2).reshape(x.shape[0], self.num_classes, *grid)


class Segmenter(nn.Module):
    def __init__(self, num_classes=6, in_channels=3, variant="s16", decoder_layers=2,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        if variant not in SEGMENTER_VARIANTS:
            raise ValueError(f"SEGMENTER_VARIANT 必须是 {', '.join(SEGMENTER_VARIANTS)} 之一")
        spec = SEGMENTER_VARIANTS[variant]
        self.arch = spec["arch"]
        self.backbone = VisionTransformer(
            in_channels, spec["embed_dim"], spec["depth"], spec["num_heads"],
            spec["patch_size"], drop_path_rate=drop_path_rate,
        )
        self.decode_head = MaskTransformer(num_classes, spec["embed_dim"], decoder_layers, dropout)

    def pretrained_spec(self):
        return {"module": self.backbone, "arch": self.arch, "allowed_missing": ()}

    @staticmethod
    def no_weight_decay_keywords():
        return ("cls_token", "pos_embed", "cls_emb")

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Segmenter expects [B,C,H,W]")
        height, width = x.shape[-2:]
        patch = self.backbone.patch_size
        x = F.pad(x, (0, -width % patch, 0, -height % patch))
        tokens, grid = self.backbone(x)
        masks = self.decode_head(tokens, grid)
        masks = F.interpolate(masks, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return [masks[..., :height, :width]]


if __name__ == "__main__":
    model = Segmenter().eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 129, 161))
    print([tuple(output.shape) for output in outputs])
