# -*- coding: utf-8 -*-
"""ImageNet initialisation of model backbones with pinned provenance.

Every checkpoint is registered with its source URL and SHA256.  A config
selects one by name (``PRETRAINED = "resnet50.tv_in1k"``); training refuses to
start when the file is missing or its hash differs, and the name plus hash
enter the resolved config and the resume fingerprint.

Files live outside the repository, in ``$LAND_COVER_PRETRAINED_DIR`` or by
default in a ``pretrained/`` directory next to the repository (shared by all
worktrees).  ``tools/fetch_pretrained.py`` downloads and verifies them.

Loading is strict in both directions: each kept checkpoint tensor must land on
a parameter/buffer of the same name and shape, and each target entry must be
covered except the prefixes a model declares as newly initialised (e.g. the
segmentation head).  A model opts in by implementing ``pretrained_spec()``
returning ``{"module", "arch", "allowed_missing"}``.

Checkpoints whose naming differs from the repository modules carry a
``convert`` function.  It runs after the classifier tensors are dropped, maps
every remaining source key exactly once and raises on any key it does not
recognise, so a changed release cannot be half-loaded silently.

Files that are not torch checkpoints carry a ``reader`` (path -> state).  The
darknet ``.weights`` format stores no names, only the values of each layer in
network order, so its reader names them from the architecture and refuses a
file whose length does not match it exactly.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import re
import struct

import numpy as np
import torch

from common.models.segmenter import POS_GRID, resize_pos_embed
from common.models.yolov3 import darknet53_layout
from common.native_datasets import sha256_file


@dataclass(frozen=True)
class PretrainedSpec:
    name: str
    arch: str
    filename: str
    url: str
    sha256: str
    drop_prefixes: tuple = ()  # classifier-only tensors
    drop_suffixes: tuple = ()  # derived buffers that the model recomputes
    source: str = ""
    convert: object = None     # state -> state in repository naming (see below)
    reader: object = None      # path -> state for non-torch files (default: torch.load)


# ---------------------------------------------------------------- key conversion
def _rename(state, rules):
    """Apply the first matching (pattern, replacement) to every key; unknown keys fail."""
    out, unknown = {}, []
    for key, value in state.items():
        for pattern, replacement in rules:
            if re.fullmatch(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                if new_key in out:
                    raise ValueError(f"转换后的键重复：{key} -> {new_key}")
                out[new_key] = value
                break
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"转换规则未覆盖 {len(unknown)} 个键，如 {sorted(unknown)[:4]}")
    return out


_HF_MIT_RULES = (
    (r"segformer\.encoder\.patch_embeddings\.(\d)\.proj\.(weight|bias)",
     r"patch_embeds.\1.proj.\2"),
    (r"segformer\.encoder\.patch_embeddings\.(\d)\.layer_norm\.(weight|bias)",
     r"patch_embeds.\1.norm.\2"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.layer_norm_([12])\.(weight|bias)",
     r"stages.\1.\2.norm\3.\4"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.attention\.self\.query\.(weight|bias)",
     r"stages.\1.\2.attention.q.\3"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.attention\.self\.sr\.(weight|bias)",
     r"stages.\1.\2.attention.sr.\3"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.attention\.self\.layer_norm\.(weight|bias)",
     r"stages.\1.\2.attention.norm.\3"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.attention\.output\.dense\.(weight|bias)",
     r"stages.\1.\2.attention.proj.\3"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.mlp\.dense1\.(weight|bias)",
     r"stages.\1.\2.ffn.fc1.\3"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.mlp\.dwconv\.dwconv\.(weight|bias)",
     r"stages.\1.\2.ffn.depthwise.\3"),
    (r"segformer\.encoder\.block\.(\d)\.(\d+)\.mlp\.dense2\.(weight|bias)",
     r"stages.\1.\2.ffn.fc2.\3"),
    (r"segformer\.encoder\.layer_norm\.(\d)\.(weight|bias)", r"norms.\1.\2"),
)
_HF_MIT_KV = re.compile(
    r"segformer\.encoder\.block\.(\d)\.(\d+)\.attention\.self\.(key|value)\.(weight|bias)"
)


def convert_hf_mit(state):
    """HF ``nvidia/mit-b*`` -> ``MixVisionTransformer``.

    HF keeps separate key/value projections; the repository fuses them into one
    ``kv`` Linear whose output is reshaped as ``(2, heads, head_dim)``, so key
    rows come first and value rows second, as in the official MiT.
    """
    pairs, rest = {}, {}
    for key, value in state.items():
        match = _HF_MIT_KV.fullmatch(key)
        if match:
            stage, block, which, kind = match.groups()
            pairs.setdefault((stage, block, kind), {})[which] = value
        else:
            rest[key] = value
    out = _rename(rest, _HF_MIT_RULES)
    for (stage, block, kind), parts in sorted(pairs.items()):
        if set(parts) != {"key", "value"}:
            raise ValueError(f"MiT block {stage}.{block} 缺少 key 或 value 的 {kind}")
        out[f"stages.{stage}.{block}.attention.kv.{kind}"] = torch.cat(
            (parts["key"], parts["value"]), dim=0
        )
    return out


_BN = r"(weight|bias|running_mean|running_var|num_batches_tracked)"


def convert_timm_mobilenetv3(state):
    """timm ``mobilenetv3_large_100`` -> ``MobileNetV3Backbone``.

    timm groups the 15 bottlenecks into stages (``blocks.<stage>.<index>``) and
    ends with a 1x1 ConvBnAct stage; the repository keeps one flat list plus
    ``final_conv``.  Inside a block the timm order is expand (conv_pw/bn1),
    depthwise (conv_dw/bn2), optional SE, projection (conv_pwl/bn3); the first
    block has no expansion (conv_dw/bn1, conv_pw/bn2).
    """
    positions = sorted({
        (int(stage), int(index))
        for stage, index in re.findall(r"^blocks\.(\d+)\.(\d+)\.", "\n".join(state), re.M)
    })
    last_stage = max(stage for stage, _ in positions)
    flat = {pos: i for i, pos in enumerate(p for p in positions if p[0] != last_stage)}
    has_se = {pos for pos in positions
              if f"blocks.{pos[0]}.{pos[1]}.se.conv_reduce.weight" in state}
    rules = [(r"conv_stem\.weight", "stem.0.weight"), (rf"bn1\.{_BN}", r"stem.1.\1"),
             (rf"blocks\.{last_stage}\.0\.conv\.weight", "final_conv.0.weight"),
             (rf"blocks\.{last_stage}\.0\.bn1\.{_BN}", r"final_conv.1.\1")]
    for (stage, index), i in flat.items():
        src = rf"blocks\.{stage}\.{index}\."
        dst = f"blocks.{i}.block."
        if i == 0:  # depthwise-separable block: no expansion
            rules += [(src + r"conv_dw\.weight", dst + "0.0.weight"),
                      (src + rf"bn1\.{_BN}", dst + r"0.1.\1"),
                      (src + r"conv_pw\.weight", dst + "1.0.weight"),
                      (src + rf"bn2\.{_BN}", dst + r"1.1.\1")]
            continue
        project = 3 if (stage, index) in has_se else 2
        rules += [(src + r"conv_pw\.weight", dst + "0.0.weight"),
                  (src + rf"bn1\.{_BN}", dst + r"0.1.\1"),
                  (src + r"conv_dw\.weight", dst + "1.0.weight"),
                  (src + rf"bn2\.{_BN}", dst + r"1.1.\1"),
                  (src + r"se\.conv_reduce\.(weight|bias)", dst + r"2.fc1.\1"),
                  (src + r"se\.conv_expand\.(weight|bias)", dst + r"2.fc2.\1"),
                  (src + r"conv_pwl\.weight", dst + f"{project}.0.weight"),
                  (src + rf"bn3\.{_BN}", dst + rf"{project}.1.\1")]
    return _rename(state, rules)


def convert_pidnet_official(state):
    """Official PIDNet ImageNet release -> ``PIDNet`` (I and P branches only).

    Renames ``layer{3,4,5}_`` (P branch) to ``layer{3,4,5}_p`` and the PagFM
    projections ``f_x``/``f_y`` (applied to the P/I inputs) to ``f_p``/``f_i``.
    The official stem convolutions carry a bias that the repository omits
    because BatchNorm follows; folding it into the BN running mean is exact in
    eval mode, and in train mode batch statistics cancel the bias anyway.
    """
    state = dict(state)
    for conv, bn in (("conv1.0", "conv1.1"), ("conv1.3", "conv1.4")):
        bias = state.pop(f"{conv}.bias", None)
        if bias is None:
            raise ValueError(f"PIDNet 权重缺少 {conv}.bias，与已知的官方发布不一致")
        state[f"{bn}.running_mean"] = state[f"{bn}.running_mean"] - bias
    rules = (
        (r"layer([345])_\.(.+)", r"layer\1_p.\2"),
        (r"pag([34])\.f_x\.(.+)", r"pag\1.f_p.\2"),
        (r"pag([34])\.f_y\.(.+)", r"pag\1.f_i.\2"),
        (r"(conv1|layer[1-5]|compression[34])\.(.+)", r"\1.\2"),
    )
    return _rename(state, rules)


_TIMM_DEIT_KEYS = (
    r"cls_token|pos_embed|patch_embed\.proj\.(?:weight|bias)|norm\.(?:weight|bias)"
    r"|blocks\.\d+\.(?:norm[12]|attn\.qkv|attn\.proj|mlp\.fc[12])\.(?:weight|bias)"
)


def convert_timm_deit(state):
    """timm ``deit_small_patch16_224`` -> Segmenter's ``VisionTransformer``.

    The names already match.  The 14x14 position grid is resampled to the
    model's 32x32 (bilinear, class token kept), as the official Segmenter does
    when it loads ImageNet weights.
    """
    out = _rename(state, ((_TIMM_DEIT_KEYS, r"\g<0>"),))
    out["pos_embed"] = resize_pos_embed(out["pos_embed"], (POS_GRID, POS_GRID))
    return out


# ---------------------------------------------------------------- darknet files
def _darknet_values(layers):
    return sum(out * (4 if batch_norm else 1) + out * cin * k * k
               for _, cin, out, k, batch_norm in layers)


def read_darknet_weights(path, layers, optional_tail=()):
    """Darknet ``.weights`` -> state dict, given the convolutions in file order.

    ``layers`` holds (prefix, in, out, kernel, batch_norm).  The header is three
    int32 (major, minor, revision) and ``seen`` (int64 from version 0.2, else
    int32), as in darknet's ``load_weights_upto``.  Each convolution then stores
    float32 values: with batch norm the BN bias, scale, running mean and running
    variance, without it the conv bias; then the (out, in, k, k) kernel.  The
    file must end after ``layers``, or after ``layers + optional_tail``.
    """
    raw = Path(path).read_bytes()
    major, minor, _ = struct.unpack_from("<3i", raw)
    offset = 20 if major * 10 + minor >= 2 and major < 1000 and minor < 1000 else 16
    values = np.frombuffer(raw, dtype="<f4", offset=offset)
    layers = list(layers)
    if values.size != _darknet_values(layers):
        if optional_tail and values.size == _darknet_values(layers + list(optional_tail)):
            layers += list(optional_tail)
        else:
            expected = [_darknet_values(layers)]
            if optional_tail:
                expected.append(_darknet_values(layers + list(optional_tail)))
            raise ValueError(f"{path} 有 {values.size} 个权重值，与网络不符（应为 "
                             f"{' 或 '.join(map(str, expected))}）")
    state, position = {}, 0

    def take(*shape):
        nonlocal position
        count = int(np.prod(shape))
        tensor = torch.from_numpy(values[position:position + count].reshape(shape).copy())
        position += count
        return tensor

    for prefix, cin, out, k, batch_norm in layers:
        if batch_norm:
            for name in ("bn.bias", "bn.weight", "bn.running_mean", "bn.running_var"):
                state[f"{prefix}.{name}"] = take(out)
            state[f"{prefix}.conv.weight"] = take(out, cin, k, k)
        else:
            state[f"{prefix}.bias"] = take(out)
            state[f"{prefix}.weight"] = take(out, cin, k, k)
    return state


# The 1x1 classifier after global average pooling in darknet53.cfg.
DARKNET53_CLASSIFIER = ("fc", 1024, 1000, 1, False)


def read_darknet53(path):
    """``darknet53.conv.74`` (the 52 convolutions) or ``darknet53.weights`` (plus ``fc``)."""
    layers = [(*layer, True) for layer in darknet53_layout()]
    return read_darknet_weights(path, layers, optional_tail=(DARKNET53_CLASSIFIER,))


REGISTRY = {spec.name: spec for spec in (
    PretrainedSpec(
        name="resnet50.tv_in1k",
        arch="resnet50",
        filename="resnet50-0676ba61.pth",
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        sha256="0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
        drop_prefixes=("fc.",),
        source="torchvision ResNet50_Weights.IMAGENET1K_V1",
    ),
    PretrainedSpec(
        name="swin_tiny_patch4_window7_224.ms_in1k",
        arch="swin_tiny",
        filename="swin_tiny_patch4_window7_224.ms_in1k.bin",
        url=("https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k"
             "/resolve/main/pytorch_model.bin"),
        sha256="4d7d61fb43a9093e1deb6b9d6216000677ae8920e6fe4d42bfdde546d93aa833",
        drop_prefixes=("head.", "norm."),
        drop_suffixes=("relative_position_index", "attn_mask"),
        source="timm (Microsoft ImageNet-1k release)",
    ),
    PretrainedSpec(
        name="ddrnet23_slim.imagenet",
        arch="ddrnet23_slim",
        filename="DDRNet23s_imagenet.pth",
        url=("https://drive.usercontent.google.com/download?"
             "id=1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4&export=download&confirm=t"),
        sha256="f6d03fa29b80e2e648f89a7639eb0c6d6c088e8f70f2cba1b2872b314a5c54b6",
        drop_prefixes=("down5.", "last_layer.", "linear."),
        source="ydhongHIT/DDRNet README, ImageNet DDRNet_23_slim (top-1 error 29.8)",
    ),
    PretrainedSpec(
        name="mit_b2.nvidia_in1k",
        arch="mit_b2",
        filename="nvidia_mit-b2.pytorch_model.bin",
        url="https://huggingface.co/nvidia/mit-b2/resolve/main/pytorch_model.bin",
        sha256="4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f",
        drop_prefixes=("classifier.",),
        source="NVIDIA SegFormer MiT-B2 ImageNet-1k, HF transformers port",
        convert=convert_hf_mit,
    ),
    PretrainedSpec(
        name="mit_b0.nvidia_in1k",
        arch="mit_b0",
        filename="nvidia_mit-b0.pytorch_model.bin",
        url="https://huggingface.co/nvidia/mit-b0/resolve/main/pytorch_model.bin",
        sha256="4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc",
        drop_prefixes=("classifier.",),
        source="NVIDIA SegFormer MiT-B0 ImageNet-1k, HF transformers port",
        convert=convert_hf_mit,
    ),
    PretrainedSpec(
        name="mobilenetv3_large_100.ra_in1k",
        arch="mobilenetv3_large_100",
        filename="mobilenetv3_large_100.ra_in1k.bin",
        url=("https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k"
             "/resolve/main/pytorch_model.bin"),
        sha256="f2ab36b2846db4c33c5472b0f7fdd46aca0836b0eeb5ccf469492f8709db0c08",
        drop_prefixes=("conv_head.", "classifier."),
        source="timm MobileNetV3-Large 1.0 ImageNet-1k (RandAugment recipe, BN eps 1e-5)",
        convert=convert_timm_mobilenetv3,
    ),
    PretrainedSpec(
        name="pidnet_s.imagenet",
        arch="pidnet_s",
        filename="PIDNet_S_ImageNet.pth.tar",
        url=("https://drive.usercontent.google.com/download?"
             "id=1Rr2vtrtTujGFohvg9AYbdfgjbWqFWXNQ&export=download&confirm=t"),
        sha256="f96e2c96b1aca1400a6f54ac41093d98b4817f6008a5d869abe6248551a5f359",
        drop_prefixes=("down5.", "last_layer.", "linear."),
        source="XuJiacong/PIDNet README, Google Drive folder PIDNet_Models/ImageNet",
        convert=convert_pidnet_official,
    ),
    PretrainedSpec(
        name="deit_small_patch16_224.fb_in1k",
        arch="deit_small_patch16",
        filename="deit_small_patch16_224.fb_in1k.bin",
        url=("https://huggingface.co/timm/deit_small_patch16_224.fb_in1k"
             "/resolve/main/pytorch_model.bin"),
        sha256="52fa3d9d746c967a02a6765fa31368eb481d52cea269798a4a33c172f1c03242",
        drop_prefixes=("head.",),
        source="timm (Facebook DeiT-S ImageNet-1k release)",
        convert=convert_timm_deit,
    ),
    PretrainedSpec(
        name="darknet53.imagenet",
        arch="darknet53",
        filename="darknet53.conv.74",
        url="https://pjreddie.com/media/files/darknet53.conv.74",
        sha256="2495c2690283e4e0bc2050cbd4660b77a8074e14e9c11150c6412fd63db496a7",
        drop_prefixes=("fc.",),
        # The 52 convolutions of an ImageNet Darknet-53 that the authors distribute for
        # training YOLOv3.  Not the same run as darknet53.weights (kernels uncorrelated);
        # the released COCO yolov3.weights descends from this file.
        source="Redmon & Farhadi, YOLOv3 training instructions (pjreddie.com/darknet/yolo)",
        reader=read_darknet53,
    ),
)}


def default_pretrained_dir():
    configured = os.environ.get("LAND_COVER_PRETRAINED_DIR")
    if configured:
        return Path(configured).expanduser()
    project_root = Path(__file__).resolve().parents[1]
    return project_root.parent / "pretrained"


def get_spec(name):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"未登记的预训练权重 {name!r}（可选: {', '.join(sorted(REGISTRY))}）"
        ) from None


def resolve_pretrained(name, directory=None):
    """Return (spec, path) after checking that the file exists and matches its hash."""
    spec = get_spec(name)
    path = Path(directory).expanduser() if directory else default_pretrained_dir()
    path = path / spec.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少预训练权重 {path}；请运行 python tools/fetch_pretrained.py --names {name}"
        )
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ValueError(
            f"预训练权重 {path} 的 SHA256 不一致：期望 {spec.sha256[:12]}，实际 {actual[:12]}"
        )
    return spec, path


def load_checkpoint_state(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13
        payload = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            payload = payload[key]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不包含 state_dict")
    state = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            state[key[len("module."):] if key.startswith("module.") else key] = value
    return state


def apply_pretrained(model, name, directory=None):
    """Load a registered checkpoint into ``model``'s declared backbone module."""
    spec, path = resolve_pretrained(name, directory)
    if not hasattr(model, "pretrained_spec"):
        raise ValueError(f"{type(model).__name__} 未声明 pretrained_spec()，不能加载 {name}")
    target = model.pretrained_spec()
    if target["arch"] != spec.arch:
        raise ValueError(
            f"{name} 是 {spec.arch} 权重，而 {type(model).__name__} 需要 {target['arch']}"
        )
    module = target["module"]
    allowed_missing = tuple(target.get("allowed_missing", ()))

    state = (spec.reader or load_checkpoint_state)(path)
    dropped = [key for key in state if key.startswith(spec.drop_prefixes)]
    derived = [key for key in state if key.endswith(spec.drop_suffixes)
               and not key.startswith(spec.drop_prefixes)]
    state = {key: value for key, value in state.items()
             if not key.startswith(spec.drop_prefixes)
             and not key.endswith(spec.drop_suffixes)}
    if spec.convert is not None:
        state = spec.convert(state)
    own = module.state_dict()
    unexpected = sorted(key for key in state if key not in own)
    mismatched = sorted(key for key in state
                        if key in own and own[key].shape != state[key].shape)
    missing = sorted(
        key for key in own
        if key not in state
        and not key.startswith(allowed_missing)
        and not key.endswith("num_batches_tracked")
    )
    if unexpected or mismatched or missing:
        details = []
        for label, keys in (("多余", unexpected), ("形状不符", mismatched), ("未覆盖", missing)):
            if keys:
                details.append(f"{label} {len(keys)} 个，如 {keys[:4]}")
        raise ValueError(f"{name} 与 {type(model).__name__} 不匹配：" + "；".join(details))

    module.load_state_dict(state, strict=False)
    initialised = sum(1 for key in own if key.startswith(allowed_missing))
    return {
        "name": name,
        "path": str(path),
        "sha256": spec.sha256,
        "loaded_tensors": len(state),
        "dropped_tensors": len(dropped),
        "derived_tensors": len(derived),
        "new_tensors": initialised,
    }
