# -*- coding: utf-8 -*-
"""数据集类、采样器与统一细类协议的 DataLoader。

``OpenEarthMapDataset`` 虽然保留这个名字，实际读取全部五个准备副本：影像路径、标签
路径、位深和类别数都由 ``common/native_datasets.py`` 的元数据给出。

数据增强约定
====================================================================
``rot90_prob`` 始终表示“发生非零 90 度旋转的概率”，默认 0.75。训练图不足 crop 时使用
居中补边。图像默认以数据集 mean 填充（归一化后为 0），标签以 ``IGNORE_INDEX`` 填充。
几何、光度和 cutout 分别使用独立的样本级随机流，因此切换光度策略不会悄悄改变
裁剪/翻转序列。

``image_bands`` 为 None 时走 ``Image.open(...).convert("RGB")``；一旦显式指定，就按原始
位深读取、不做 convert。``dtype_scale="auto"`` 对 uint8 是 ``/255``，对 uint16 是
``/65535``。遇到非 8 位、非 RGB/L 的影像而 ``image_bands`` 为 None 时直接报错。
"""
import glob
import pathlib
import random
import os

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Sampler

Image.MAX_IMAGE_PIXELS = None  # OEM 大图，关闭 PIL 解压炸弹警告

# dtype_scale 允许的字符串取值 -> 除数
DTYPE_SCALE_DIVISORS = {
    "uint8": 255.0,
    "uint16": 65535.0,
    "reflectance_1e4": 10000.0,   # Sentinel-2 L2A 常见的 1e4 反射率缩放
}


def _index_images(root, pattern="*/images/*.tif"):
    """按 glob 模式扫描影像，返回 {文件名: 完整路径}。

    pattern 里的 ``/`` 会先归一化成 ``os.sep``：直接 ``os.path.join(root, "*/images/*.tif")``
    在 Windows 上会得到 ``root\\*/images/*.tif`` 这种混合分隔符，glob 按 ``\\`` 切分后
    最后一段变成 ``images/*.tif``，永远匹配不到。
    """
    pattern = pattern.replace("/", os.sep).replace("\\", os.sep)
    mapping = {}
    for p in sorted(glob.glob(os.path.join(root, pattern))):
        mapping[os.path.basename(p)] = p
    return mapping


def _replace_path_parts(path, pairs):
    """按**路径分量**逐段替换，而不是对整串做 str.replace。

    旧实现用 ``img_path.replace("images", "labels")`` 这类整串替换，路径里任何
    位置出现 "images" 都会被换掉（例如 /data/images_raw/images/x.tif）。这里只
    替换完整等于 src 的那一段，语义更严格；对默认情况结果与旧实现完全相同。

    用 ``PurePath`` 而不是手工 ``split('/')`` + ``os.path.join``：后者在 Windows 上
    会把首段 ``'D:'``（没有尾分隔符）当成**盘符相对**引用，拼出 ``'D:foo\\bar'``
    这种无效路径，导致标签永远找不到。PurePath 的 ``parts[0]`` 是带反斜杠的锚点
    ``'D:\\\\'``，因此能正确还原绝对路径。
    """
    parts = list(pathlib.PurePath(path).parts)
    for src, dst in pairs:
        parts = [dst if seg == src else seg for seg in parts]
    return str(pathlib.PurePath(*parts)) if parts else path



def _read_split(list_file):
    with open(list_file, "r") as f:
        return [os.path.basename(ln.strip()) for ln in f if ln.strip()]


class OpenEarthMapDataset(Dataset):
    def __init__(self, root, list_file, training, crop_size=512,
                 scale_range=(0.75, 1.5), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 pad_divisor=32, ignore_index=0, image_pad_mode="mean",
                 # ---- 可搜索的增强强度 ----
                 hflip_prob=0.5, vflip_prob=0.5, rot90_prob=0.75,
                 legacy_rot90=False,
                 brightness=0.0, contrast=0.0, saturation=0.0,
                 blur_prob=0.0, blur_sigma=(0.5, 1.5),
                 cutout_prob=0.0, cutout_frac=(0.02, 0.10),
                 cutout_label_mode="ignore", cutout_n=1,
                 # ---- Phase 0：跨数据集适配（默认值 == 原行为）----
                 image_glob="*/images/*.tif",
                 label_replace=(("images", "labels"),),
                 image_bands=None,
                 dtype_scale="auto",
                 label_suffix=None,
                 # ---- 严格标签协议 ----
                 num_classes=None,
                 strict_label_schema=False):
        self.root = root
        self.training = training
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.pad_divisor = pad_divisor
        self.ignore_index = int(ignore_index)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.strict_label_schema = bool(strict_label_schema)
        self.image_pad_mode = str(image_pad_mode).lower()

        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rot90_prob = float(rot90_prob)
        self.legacy_rot90 = bool(legacy_rot90)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.blur_prob = float(blur_prob)
        self.blur_sigma = tuple(blur_sigma)
        self.cutout_prob = float(cutout_prob)
        self.cutout_frac = tuple(cutout_frac)
        self.cutout_label_mode = str(cutout_label_mode).lower()
        self.cutout_n = int(cutout_n)

        self.image_glob = str(image_glob)
        self.label_replace = tuple((str(a), str(b)) for a, b in label_replace)
        self.label_suffix = None if label_suffix is None else str(label_suffix)
        if self.label_suffix is not None and not self.label_suffix.startswith("."):
            raise ValueError(f"label_suffix 必须以 '.' 开头，收到 {label_suffix!r}")
        self.image_bands = (None if image_bands is None
                            else tuple(int(b) for b in image_bands))
        self.dtype_scale = dtype_scale
        self._warned_photo = False
        self._validate_augmentation()

        index = _index_images(root, self.image_glob)
        if not index:
            raise FileNotFoundError(
                f"在 {root} 下按模式 {self.image_glob!r} 没有找到任何影像，请检查 DATA_ROOT")

        names = _read_split(list_file)
        self.samples, missing = [], 0
        for name in names:
            if name in index:
                img_path = index[name]
                lab_path = _replace_path_parts(img_path, self.label_replace)
                if self.label_suffix is not None:  # e.g. JPEG image -> PNG label
                    lab_path = str(pathlib.PurePath(lab_path).with_suffix(self.label_suffix))
                if os.path.exists(lab_path):
                    self.samples.append((img_path, lab_path, name))
                else:
                    missing += 1
            else:
                missing += 1
        if missing:
            print(f"[dataset] {os.path.basename(list_file)}: 跳过 {missing} 个缺失条目"
                  f"（wo_xBD 版本中 xBD 影像不存在属正常现象），有效样本 {len(self.samples)}")
        if not self.samples:
            raise RuntimeError(f"{list_file} 中没有匹配到任何有效样本")

        if self.num_classes is not None and self.num_classes < 1:
            raise ValueError(f"num_classes 必须是正整数，收到 {self.num_classes}")
        if self.strict_label_schema and self.num_classes is None:
            raise ValueError("strict_label_schema=True 时必须显式提供 num_classes")
        if self.strict_label_schema:
            self._validate_all_labels()

        # 通道数与数值除数在读到真实影像后才能最终确定（dtype_scale="auto" 需要探测）。
        self.n_channels = len(self.image_bands) if self.image_bands is not None else 3
        self._scale = self._resolve_scale(self.samples[0][0])
        if self.image_bands is not None:
            with Image.open(self.samples[0][0]) as sample:
                probe = np.array(sample)
            self._raw_dtype = probe.dtype
            n_src = 1 if probe.ndim == 2 else probe.shape[-1]
            if max(self.image_bands) >= n_src:
                raise ValueError(
                    f"image_bands 取到通道 {max(self.image_bands)}，但 "
                    f"{os.path.basename(self.samples[0][0])} 只有 {n_src} 个通道")
        else:
            self._raw_dtype = None

    def _resolve_scale(self, sample_path):
        """确定把原始整数像素值映射到 [0,1] 的除数；uint8 → 255，与原行为逐位相同。"""
        spec = self.dtype_scale
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            return float(spec)
        if isinstance(spec, str) and spec in DTYPE_SCALE_DIVISORS:
            return float(DTYPE_SCALE_DIVISORS[spec])
        if spec != "auto":
            raise ValueError(f"无法解析的 dtype_scale: {spec!r}")
        if self.image_bands is None:
            # 旧路径走 convert("RGB")，输出必然是 8 位；源若是 >8 位会在那里丢精度，
            # 因此这里只接受 8 位源，遇到更宽的位深直接报错而不是静默截断。
            with Image.open(sample_path) as sample:
                mode = sample.mode
            if mode not in ("RGB", "RGBA", "L", "P", "1"):
                raise ValueError(
                    f"源影像 mode={mode!r} 不是 8 位常规模式。请显式设置 image_bands "
                    f"以按原始位深读取（convert('RGB') 会把 >8 位数据压掉），"
                    f"或先用 tools/resample_gsd.py --to-uint8 离线转换为 8 位。")
            return 255.0
        with Image.open(sample_path) as sample:
            dtype = np.array(sample).dtype
        if dtype == np.uint8:
            return 255.0
        if dtype == np.uint16:
            return 65535.0
        raise ValueError(
            f"dtype_scale='auto' 无法从 dtype={dtype} 推断除数，请显式指定"
            f"（如 dtype_scale='reflectance_1e4' 或一个具体数值）")

    def _validate_augmentation(self):
        """早失败：增强强度写错时立刻报错，而不是训到一半才发现。"""
        for name in ("hflip_prob", "vflip_prob", "blur_prob", "cutout_prob"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1]，收到 {v}")
        if not 0.0 <= self.rot90_prob <= 1.0:
            raise ValueError(f"rot90_prob 必须在 [0, 1]，收到 {self.rot90_prob}")
        for name in ("brightness", "contrast", "saturation"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} 必须在 [0, 1)（作为相对抖动幅度），收到 {v}")
        if (len(self.blur_sigma) != 2 or self.blur_sigma[0] < 0
                or self.blur_sigma[1] < self.blur_sigma[0]):
            raise ValueError(
                "blur_sigma 必须是 (low, high) 且 0 <= low <= high，"
                f"收到 {self.blur_sigma}")
        if (len(self.cutout_frac) != 2 or self.cutout_frac[0] < 0
                or self.cutout_frac[1] < self.cutout_frac[0]
                or self.cutout_frac[1] > 1):
            raise ValueError(
                "cutout_frac 必须是 (low, high) 且 0 <= low <= high <= 1，"
                f"收到 {self.cutout_frac}")
        if not 0.0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError(f"scale_range 必须是 (low, high) 且 0 < low <= high，收到 {self.scale_range}")
        if self.image_pad_mode not in {"mean", "reflect", "edge", "legacy_zero_br"}:
            raise ValueError(
                "image_pad_mode 必须是 'mean'、'reflect'、'edge' 或 "
                "'legacy_zero_br'，"
                f"收到 {self.image_pad_mode!r}"
            )
        if self.cutout_label_mode not in {"ignore", "keep"}:
            raise ValueError(
                "cutout_label_mode 必须是 'ignore' 或 'keep'，"
                f"收到 {self.cutout_label_mode!r}"
            )
        if self.cutout_n < 1:
            raise ValueError(f"cutout_n 必须 >= 1，收到 {self.cutout_n}")
        if self.crop_size < 1 or self.pad_divisor < 1:
            raise ValueError("crop_size 与 pad_divisor 必须是正整数")
        n_bands = len(self.image_bands) if self.image_bands is not None else 3
        if (self.mean.shape != (n_bands,) or self.std.shape != (n_bands,)
                or np.any(self.std <= 0)):
            raise ValueError(
                f"mean/std 必须是长度为 {n_bands} 的序列（与通道数一致）且 std 全部 > 0；"
                f"收到 mean.shape={tuple(self.mean.shape)} std.shape={tuple(self.std.shape)}")
        if not isinstance(self.dtype_scale, (int, float)) or isinstance(self.dtype_scale, bool):
            if self.dtype_scale not in DTYPE_SCALE_DIVISORS and self.dtype_scale != "auto":
                raise ValueError(
                    f"dtype_scale 必须是 'auto'、{sorted(DTYPE_SCALE_DIVISORS)} 之一，"
                    f"或一个正数除数；收到 {self.dtype_scale!r}")
        elif not float(self.dtype_scale) > 0:
            raise ValueError(f"dtype_scale 作为除数必须 > 0，收到 {self.dtype_scale!r}")
        if self.image_bands is not None:
            if len(self.image_bands) < 1:
                raise ValueError("image_bands 不能为空")
            if min(self.image_bands) < 0:
                raise ValueError(f"image_bands 不能有负值，收到 {self.image_bands}")
            if self.dtype_scale == "auto" and len(self.image_bands) != 3:
                pass  # auto 会在读到影像后按真实 dtype 解析，无需在此限制

    def __len__(self):
        return len(self.samples)

    def _validate_label_file(self, path):
        """Validate one materialised index mask against the declared logits.

        ``ignore_index`` may deliberately sit outside ``[0, num_classes)``;
        The protocol uses 255 so the ignored value can never become an argmax channel.
        RGB masks and undeclared IDs fail before a training worker/GPU is
        started instead of being silently reduced to channel zero.
        """
        with Image.open(path) as image:
            arr = np.asarray(image)
        if arr.ndim != 2:
            raise ValueError(
                f"严格标签协议要求二维单通道索引图：{path} shape={arr.shape}"
            )
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"标签必须是整数 dtype：{path} dtype={arr.dtype}")
        valid = (arr >= 0) & (arr < self.num_classes)
        valid |= arr == self.ignore_index
        if not valid.all():
            bad = np.unique(arr[~valid]).tolist()
            raise ValueError(
                f"标签含模型 taxonomy 外 ID：{path} -> {bad[:16]}；"
                f"允许 0..{self.num_classes - 1} 或 ignore={self.ignore_index}"
            )

    def _validate_all_labels(self):
        for _, label_path, _ in self.samples:
            self._validate_label_file(label_path)

    # ------------------------------------------------------------ 增强
    def _load_image(self, path):
        """bands=None → PIL 8 位 RGB（与旧行为逐位相同）；否则按原始位深读成 ndarray 并选通道。"""
        if self.image_bands is None:
            with Image.open(path) as image:
                return image.convert("RGB")
        with Image.open(path) as image:
            arr = np.array(image)
        if arr.ndim == 2:
            arr = arr[..., None]
        return np.ascontiguousarray(arr[..., list(self.image_bands)])

    def _random_scale(self, img, lab, rng):
        s = rng.uniform(*self.scale_range)
        if isinstance(img, np.ndarray):
            # 多波段/非 8 位路径：PIL 的 BILINEAR resize 只工作于 8 位，
            # 改用 torch 插值并保持原始 dtype。
            h, w = img.shape[:2]
            nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
            t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
            t = t.unsqueeze(0).float()
            t = torch.nn.functional.interpolate(t, size=(nh, nw), mode="bilinear",
                                                align_corners=False)
            img = t[0].permute(1, 2, 0).numpy().astype(self._raw_dtype, copy=False)
            lab = lab.resize((nw, nh), Image.NEAREST)
            return img, lab
        w, h = img.size
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = img.resize((nw, nh), Image.BILINEAR)
        lab = lab.resize((nw, nh), Image.NEAREST)
        return img, lab

    def _photometric(self, img, rng):
        """光度抖动；固定预取随机量，避免开关某一策略改变其余策略的采样。"""
        brightness_factor = rng.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        contrast_factor = rng.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        saturation_factor = rng.uniform(1.0 - self.saturation, 1.0 + self.saturation)
        blur_draw = rng.random()
        blur_radius = rng.uniform(*self.blur_sigma)
        if isinstance(img, np.ndarray):
            # 多波段路径没有 PIL 的光度算子。亮度/对比度可以线性实现，
            # 饱和度与高斯模糊在这里不适用——显式跳过并只警告一次，
            # 而不是静默给出与 8 位路径不同的增强语义。
            if (self.saturation > 0 or self.blur_prob > 0) and not self._warned_photo:
                print("[dataset] 多波段路径跳过 saturation/blur（PIL 算子不可用）；"
                      "brightness/contrast 仍按线性方式生效", flush=True)
                self._warned_photo = True
            out = img.astype(np.float32)
            if self.brightness > 0:
                out = out * brightness_factor
            if self.contrast > 0:
                m = float(out.mean())
                out = (out - m) * contrast_factor + m
            out = np.clip(out, 0, self._scale)
            return out.astype(self._raw_dtype, copy=False)
        if self.brightness > 0:
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        if self.contrast > 0:
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        if self.saturation > 0:
            img = ImageEnhance.Color(img).enhance(saturation_factor)
        if blur_draw < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return img

    def _pad_to(self, img, lab, th, tw):
        """居中补到 ``th x tw``；标签补 ignore，图像按显式策略补边。"""
        h, w = lab.shape
        if h >= th and w >= tw:
            return img, lab
        ph, pw = max(th - h, 0), max(tw - w, 0)
        if self.image_pad_mode == "legacy_zero_br":
            # 仅供没有 PROTOCOL_ID 的历史配置复现旧实验。新 benchmark/HPO
            # 必须显式使用 mean/reflect/edge，避免单侧黑边成为尺度混杂因素。
            padded_img = np.pad(
                img, ((0, ph), (0, pw), (0, 0)),
                mode="constant", constant_values=0,
            )
            padded_lab = np.pad(
                lab, ((0, ph), (0, pw)), mode="constant",
                constant_values=self.ignore_index,
            )
            return padded_img, padded_lab

        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        pad_width_img = ((top, bottom), (left, right), (0, 0))

        if self.image_pad_mode == "mean":
            # 使用 float32 精确保存 mean*scale；后续 /scale - mean 后补边约等于 0。
            padded_img = np.empty((h + ph, w + pw, self.n_channels), dtype=np.float32)
            padded_img[...] = self.mean * self._scale
            padded_img[top:top + h, left:left + w] = img
        else:
            mode = self.image_pad_mode
            # np.pad(reflect) 对长度为 1 的轴不安全；退化为 edge，避免验证阶段崩溃。
            if mode == "reflect" and ((ph and h < 2) or (pw and w < 2)):
                mode = "edge"
            padded_img = np.pad(img, pad_width_img, mode=mode)

        padded_lab = np.pad(
            lab,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=self.ignore_index,
        )
        return padded_img, padded_lab

    def _random_crop(self, img, lab, rng):
        th = tw = self.crop_size
        h, w = lab.shape
        y = rng.randint(0, h - th)
        x = rng.randint(0, w - tw)
        return img[y:y + th, x:x + tw], lab[y:y + th, x:x + tw]

    def _cutout(self, img, lab, rng):
        """随机擦除：图像填数据均值；标签按 ignore/keep 的显式策略处理。"""
        if rng.random() >= self.cutout_prob:
            return img, lab
        # 整数 dtype 无法精确表示 mean*scale，转为 float32 后再填充。
        img = img.astype(np.float32, copy=True)
        lab = lab.copy()
        h, w = lab.shape
        for _ in range(self.cutout_n):
            ch = int(round(h * rng.uniform(*self.cutout_frac)))
            cw = int(round(w * rng.uniform(*self.cutout_frac)))
            if ch < 1 or cw < 1:
                continue
            ch, cw = min(ch, h), min(cw, w)
            y = rng.randint(0, h - ch)
            x = rng.randint(0, w - cw)
            img[y:y + ch, x:x + cw] = self.mean * self._scale
            if self.cutout_label_mode == "ignore":
                lab[y:y + ch, x:x + cw] = self.ignore_index
        return img, lab

    def _sample_rotation(self, rng):
        """返回 k∈{0,1,2,3}；非 legacy 模式下概率语义在 [0,1] 连续。"""
        if self.legacy_rot90:
            return rng.randint(0, 3)
        return rng.randint(1, 3) if rng.random() < self.rot90_prob else 0

    def _normalize_to_tensor(self, img, lab):
        img = (img.astype(np.float32) / self._scale - self.mean) / self.std
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        lab = torch.from_numpy(np.ascontiguousarray(lab.astype(np.int64)))
        return img, lab

    # ------------------------------------------------------------ 取样
    def __getitem__(self, i):
        img_path, lab_path, name = self.samples[i]
        img = self._load_image(img_path)
        with Image.open(lab_path) as label:
            lab = label.copy()

        if self.training:
            # 无论哪些增强开关打开，每个样本都固定提取三个子流种子。几何流不会
            # 因 brightness/blur/cutout 的开关或内部随机数消耗而漂移。
            geometry_rng = random.Random(random.getrandbits(64))
            photo_rng = random.Random(random.getrandbits(64))
            cutout_rng = random.Random(random.getrandbits(64))

            img, lab = self._random_scale(img, lab, geometry_rng)
            img = self._photometric(img, photo_rng)
            img = np.array(img)
            lab = np.array(lab)
            if lab.ndim == 3:  # 个别标签存成多通道时取第一通道
                lab = lab[..., 0]
            img, lab = self._pad_to(img, lab, self.crop_size, self.crop_size)
            img, lab = self._random_crop(img, lab, geometry_rng)
            img, lab = self._cutout(img, lab, cutout_rng)

            if geometry_rng.random() < self.hflip_prob:
                img, lab = np.fliplr(img), np.fliplr(lab)
            if geometry_rng.random() < self.vflip_prob:
                img, lab = np.flipud(img), np.flipud(lab)
            k = self._sample_rotation(geometry_rng)
            if k:
                img, lab = np.rot90(img, k), np.rot90(lab, k)
        else:
            img = np.array(img)
            lab = np.array(lab)
            if lab.ndim == 3:
                lab = lab[..., 0]
            # 全图评估：补边到 pad_divisor 的整数倍（OEM 标准 1024 图无需补）
            h, w = lab.shape
            d = self.pad_divisor
            th = (h + d - 1) // d * d
            tw = (w + d - 1) // d * d
            img, lab = self._pad_to(img, lab, th, tw)

        img, lab = self._normalize_to_tensor(img, lab)
        return img, lab, name

# 从 config 读取增强参数的唯一入口，新增键都带安全默认值。
AUG_KEYS = (
    ("HFLIP_PROB", "hflip_prob", 0.5),
    ("VFLIP_PROB", "vflip_prob", 0.5),
    ("ROT90_PROB", "rot90_prob", 0.75),
    ("AUG_LEGACY_ROT90", "legacy_rot90", False),
    ("BRIGHTNESS", "brightness", 0.0),
    ("CONTRAST", "contrast", 0.0),
    ("SATURATION", "saturation", 0.0),
    ("BLUR_PROB", "blur_prob", 0.0),
    ("BLUR_SIGMA", "blur_sigma", (0.5, 1.5)),
    ("CUTOUT_PROB", "cutout_prob", 0.0),
    ("CUTOUT_FRAC", "cutout_frac", (0.02, 0.10)),
    ("CUTOUT_LABEL_MODE", "cutout_label_mode", "ignore"),
    ("CUTOUT_N", "cutout_n", 1),
)

def resolve_aug_kwargs(cfg):
    """按 AUG_KEYS 从 cfg 取出增强参数；缺失的键用默认值。"""
    return {kw: getattr(cfg, cfg_key, default) for cfg_key, kw, default in AUG_KEYS}


def seed_dataloader_worker(worker_id):
    """由 PyTorch 分配的 worker seed 同步初始化 Python 与 NumPy RNG。"""
    del worker_id  # seed 已经包含 worker id；保留参数以符合 DataLoader 回调签名。
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _seeded_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class StreamingEpochSampler(Sampler):
    """Fixed-length virtual epochs cut from one stream of seeded permutations.

    Pass ``p`` over the training list is ``randperm`` seeded by ``(seed, p)``;
    virtual epoch ``e`` yields stream positions ``[e*L, (e+1)*L)``.  Every
    sample is therefore seen once per pass whatever the epoch length (a small
    dataset spans several passes per epoch, FLAIR one pass over several
    epochs), and an epoch's indices depend only on ``(seed, epoch)``, so a
    resumed run continues the same stream.  ``set_epoch`` pins the epoch; each
    iteration otherwise advances it by one.
    """

    def __init__(self, n_samples, samples_per_epoch, seed):
        if n_samples < 1 or samples_per_epoch < 1:
            raise ValueError("n_samples 与 samples_per_epoch 必须是正整数")
        self.n_samples = int(n_samples)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def _permutation(self, index):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + index)
        return torch.randperm(self.n_samples, generator=generator).tolist()

    def __iter__(self):
        start = self.epoch * self.samples_per_epoch
        stop = start + self.samples_per_epoch
        indices = []
        index = start // self.n_samples
        while len(indices) < self.samples_per_epoch:
            offset = index * self.n_samples
            indices.extend(self._permutation(index)[max(start - offset, 0):stop - offset])
            index += 1
        self.epoch += 1
        return iter(indices)


class LabelOffsetDataset(Dataset):
    """Shift a dataset's labels by ``offset`` (ignore stays ignore).

    The unified protocol stacks the native classes of several datasets into
    one label space; dataset d's class c becomes ``offset_d + c``.
    """

    def __init__(self, dataset, offset, ignore_index=255):
        self.dataset = dataset
        self.offset = int(offset)
        self.ignore_index = int(ignore_index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, lab, name = self.dataset[i]
        lab = torch.where(lab == self.ignore_index, lab, lab + self.offset)
        return img, lab, name


class BalancedEpochSampler(Sampler):
    """Fixed-length virtual epochs taking a fixed share from each dataset.

    Indices address a ``ConcatDataset`` of datasets with ``sizes``.  Epoch e
    takes ``counts[d]`` samples from dataset d -- the epoch-e slice of that
    dataset's own ``StreamingEpochSampler`` stream -- and shuffles them
    together with a generator seeded by ``(seed, e)``.  Each dataset keeps the
    coverage of the single-dataset protocol (every sample once per pass), and
    an epoch depends only on ``(seed, epoch)``, so resuming continues the
    same streams.  ``counts`` splits ``samples_per_epoch`` in proportion to
    ``weights`` (largest remainder; equal weights by default).
    """

    def __init__(self, sizes, samples_per_epoch, seed, weights=None):
        sizes = [int(n) for n in sizes]
        weights = [1.0] * len(sizes) if weights is None else [float(w) for w in weights]
        if not sizes or len(weights) != len(sizes) or min(weights) <= 0:
            raise ValueError("sizes 与 weights 必须一一对应且权重为正")
        total = int(samples_per_epoch)
        quota = [total * w / sum(weights) for w in weights]
        counts = [int(q) for q in quota]
        for d in sorted(range(len(sizes)), key=lambda d: counts[d] - quota[d])[:total - sum(counts)]:
            counts[d] += 1
        if min(counts) < 1:
            raise ValueError(f"每个 epoch 的 {total} 个样本不够分给 {len(sizes)} 个数据集")
        self.counts = counts
        self.offsets = np.cumsum([0] + sizes[:-1]).tolist()
        self.seed = int(seed)
        self.streams = [StreamingEpochSampler(n, c, self.seed + 1009 * (d + 1))
                        for d, (n, c) in enumerate(zip(sizes, counts))]
        self.samples_per_epoch = total
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        indices = []
        for offset, stream in zip(self.offsets, self.streams):
            stream.set_epoch(self.epoch)
            indices.extend(offset + i for i in stream)
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        order = torch.randperm(len(indices), generator=generator).tolist()
        self.epoch += 1
        return iter([indices[i] for i in order])


def build_unified_dataloaders(cfg):
    """Train loader over every dataset of ``cfg.UNIFIED_DATASETS``, one val loader each.

    Training samples carry global labels (``cfg.LABEL_OFFSETS``); the
    selection loaders keep native labels, scored per dataset by
    ``common.unified_protocol.evaluate_unified``.  Native labels are checked
    against each dataset's own class count before any offset is applied.
    """
    aug = resolve_aug_kwargs(cfg)
    seed = getattr(cfg, "SEED", 0)
    train_sets, val_loaders = [], {}
    for index, (dataset_id, entry) in enumerate(cfg.UNIFIED_DATASETS.items()):
        data_kw = dict(
            image_glob=entry["image_glob"], label_replace=entry["label_replace"],
            label_suffix=entry["label_suffix"], image_bands=entry["image_bands"],
            dtype_scale=cfg.DTYPE_SCALE, num_classes=entry["num_classes"],
            strict_label_schema=cfg.STRICT_LABEL_SCHEMA)
        common = dict(mean=cfg.MEAN, std=cfg.STD, ignore_index=cfg.IGNORE_INDEX,
                      image_pad_mode=cfg.IMAGE_PAD_MODE, pad_divisor=cfg.PAD_DIVISOR, **data_kw)
        train_set = OpenEarthMapDataset(
            entry["root"], entry["train_list"], training=True, crop_size=cfg.CROP_SIZE,
            scale_range=cfg.SCALE_RANGE, **aug, **common)
        train_sets.append(LabelOffsetDataset(train_set, cfg.LABEL_OFFSETS[dataset_id],
                                             cfg.IGNORE_INDEX))
        val_set = OpenEarthMapDataset(entry["root"], entry["val_list"], training=False, **common)
        val_loaders[dataset_id] = DataLoader(
            val_set, batch_size=entry["val_batch_size"], shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True,
            worker_init_fn=seed_dataloader_worker,
            generator=_seeded_generator(seed + 1 + index))
    sampler = BalancedEpochSampler(
        [len(s) for s in train_sets], int(cfg.ITERS_PER_EPOCH) * cfg.BATCH_SIZE, seed,
        [cfg.DATASET_WEIGHTS[d] for d in cfg.UNIFIED_DATASETS])
    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=cfg.BATCH_SIZE, sampler=sampler,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        worker_init_fn=seed_dataloader_worker, generator=_seeded_generator(seed))
    return train_loader, val_loaders
