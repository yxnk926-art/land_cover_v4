# -*- coding: utf-8 -*-
"""Swin Transformer backbone (Liu et al., ICCV 2021) for dense prediction.

Self-contained (no timm/mmcv at runtime).  Parameter names follow timm >= 0.9
so the ImageNet checkpoint ``timm/swin_tiny_patch4_window7_224.ms_in1k`` loads
without remapping; tests/test_backbones.py checks numerical equality
with timm on shared weights.

Differences from the classification model, all standard for segmentation:

* any input size is accepted: the image is padded to a multiple of the patch
  size, odd feature maps are padded before patch merging, and tokens are
  padded to window multiples inside each block with the shifted-window mask
  built for the padded size (timm's dynamic-mask path);
* odd blocks always use shifted windows, also when a feature map is smaller
  than one window (as in the official segmentation code);
* every stage output gets its own LayerNorm (``out_norms``), newly
  initialised; the classifier's final ``norm`` and ``head`` are not used.

``forward`` returns four NCHW maps at strides 4, 8, 16 and 32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


SWIN_VARIANTS = {
    "tiny": dict(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)),
    "small": dict(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24)),
    "base": dict(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)),
}


def window_partition(x, window_size):
    """[B, H, W, C] -> [B * num_windows, window_size**2, C]."""
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size,
               width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def window_reverse(windows, window_size, height, width):
    channels = windows.shape[-1]
    x = windows.view(-1, height // window_size, width // window_size,
                     window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, height, width, channels)


def relative_position_index(window_size):
    coords = torch.stack(torch.meshgrid(
        torch.arange(window_size), torch.arange(window_size), indexing="ij"
    )).flatten(1)
    relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window_size - 1
    relative[:, :, 1] += window_size - 1
    relative[:, :, 0] *= 2 * window_size - 1
    return relative.sum(-1)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        self.register_buffer(
            "relative_position_index", relative_position_index(window_size),
            persistent=False,
        )
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(windows, tokens, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(tokens, tokens, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            bias = bias.unsqueeze(1) + mask.view(1, num_windows, 1, tokens, tokens)
            bias = bias.expand(windows // num_windows, -1, -1, -1, -1)
            bias = bias.reshape(windows, self.num_heads, tokens, tokens)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.proj(x.transpose(1, 2).reshape(windows, tokens, channels))


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift, mlp_ratio=4.0,
                 drop_path=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path2 = DropPath(drop_path)
        self._mask_cache = {}

    def _shift_mask(self, height, width, device):
        key = (height, width, str(device))
        if key not in self._mask_cache:
            size, shift = self.window_size, self.shift_size
            image = torch.zeros(1, height, width, 1, device=device)
            region = 0
            for rows in (slice(0, -size), slice(-size, -shift), slice(-shift, None)):
                for cols in (slice(0, -size), slice(-size, -shift), slice(-shift, None)):
                    image[:, rows, cols, :] = region
                    region += 1
            windows = window_partition(image, size).squeeze(-1)
            mask = windows.unsqueeze(1) - windows.unsqueeze(2)
            self._mask_cache[key] = mask.masked_fill(mask != 0, -100.0)
        return self._mask_cache[key]

    def _attention(self, x):
        _, height, width, _ = x.shape
        size, shift = self.window_size, self.shift_size
        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        pad_h = (size - height % size) % size
        pad_w = (size - width % size) % size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1:3]
        mask = self._shift_mask(padded_h, padded_w, x.device) if shift else None
        windows = self.attn(window_partition(x, size), mask)
        x = window_reverse(windows, size, padded_h, padded_w)[:, :height, :width, :]
        if shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        return x

    def forward(self, x):
        x = x + self.drop_path1(self._attention(self.norm1(x)))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x):
        batch, height, width, channels = x.shape
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        height, width = x.shape[1:3]
        x = x.reshape(batch, height // 2, 2, width // 2, 2, channels)
        x = x.permute(0, 1, 3, 4, 2, 5).flatten(3)
        return self.reduction(self.norm(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class SwinStage(nn.Module):
    def __init__(self, dim, out_dim, depth, num_heads, window_size, downsample,
                 drop_paths):
        super().__init__()
        self.downsample = PatchMerging(dim, out_dim) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[
            SwinBlock(out_dim, num_heads, window_size, shift=index % 2 == 1,
                      drop_path=drop_paths[index])
            for index in range(depth)
        ])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class SwinTransformer(nn.Module):
    def __init__(self, variant="tiny", in_channels=3, window_size=7,
                 drop_path_rate=0.3, patch_size=4):
        super().__init__()
        if variant not in SWIN_VARIANTS:
            raise ValueError(f"SWIN_VARIANT 必须是 {', '.join(SWIN_VARIANTS)} 之一")
        spec = SWIN_VARIANTS[variant]
        self.variant = variant
        embed_dim, depths, heads = spec["embed_dim"], spec["depths"], spec["num_heads"]
        dims = tuple(embed_dim * 2 ** index for index in range(len(depths)))
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.layers = nn.ModuleList()
        offset = 0
        for index, depth in enumerate(depths):
            self.layers.append(SwinStage(
                dims[max(index - 1, 0)], dims[index], depth, heads[index],
                window_size, downsample=index > 0,
                drop_paths=rates[offset:offset + depth],
            ))
            offset += depth
        self.out_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])
        self.channels = dims
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def no_weight_decay_keywords():
        return ("relative_position_bias_table",)

    def forward_stages(self, x, normalize=True):
        x = self.patch_embed(x)
        outputs = []
        for layer, norm in zip(self.layers, self.out_norms):
            x = layer(x)
            stage = norm(x) if normalize else x
            outputs.append(stage.permute(0, 3, 1, 2).contiguous())
        return outputs

    def forward(self, x):
        return self.forward_stages(x, normalize=True)
