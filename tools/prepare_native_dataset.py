#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_native_dataset.py — 生成 FLAIR-HUB / Global-LoveDA / DeepGlobe / OpenEarthMap /
ISPRS Potsdam 的原生类别数据副本。

类别、原始标签编码、文件配对和划分规则都定义在 ``common/native_datasets.py``；
本工具只负责按定义落盘：

  * 影像**硬链接**到 ``<dst>/<group>/images/``（要求与原始数据在同一文件系统，不复制）；
    Potsdam 的 6000² 图块切成不重叠的 1000² 小块，存为无损 PNG；
  * 标签重编码为 PNG（模型类别 0..C-1，ignore=255），写到 ``<dst>/<group>/labels/``；
  * 每个划分一份列表（``train/val/test.txt``；OpenEarthMap 为
    ``train/select/test_city/test_indomain.txt``）与 ``native_dataset.json``（类别、映射、
    划分配方、split_id、逐类像素数、列表 SHA256、生成代码哈希）。

OpenEarthMap 用固定的城市划分（方案 B），完整生成时数量和列表哈希必须与代码中冻结的
一致，否则拒绝生成；与训练集重叠的 30 张官方 val 图块不进入任何划分。Potsdam 用
BANet / UNetFormer 的固定图块划分，标注有误的 7_10 不进入任何划分。FLAIR-HUB 官方
发布中无法解码的影像（``FLAIR_UNREADABLE_IMAGES``）在划分之后剔除。

每张影像都完整解码一次；解不开的文件直接报错并放弃生成，不会被跳过。

全部内容先写进同目录下的隐藏临时目录，所有样本成功后才原子改名为 ``--dst``；
目标已存在时拒绝运行（准备好的数据视为只读，要改就生成新版本目录）。

用法
=============================================================================
    # 只看划分与数量（不写任何文件）
    python tools/prepare_native_dataset.py --dataset flair_hub --dry-run

    # 正式生成
    python tools/prepare_native_dataset.py --dataset flair_hub \\
        --dst /data/yfk/datasets/prepared/FLAIR-HUB_native_v1 --workers 16

``--src`` 默认取 ``$LAND_COVER_DATASETS/raw/<数据集目录>``。
``--limit-per-group N`` 只取每组前 N 张，用于冒烟测试（会记入元数据）。
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import native_datasets as nd  # noqa: E402
from common.data_ownership import assert_not_other_land_cover_repo  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # DeepGlobe 2448² is fine, keep PIL quiet


def _process(task):
    """Link (or cut) one image and write its re-encoded label; return pixel statistics."""
    dataset_id, sample, root = task
    spec = nd.get_spec(dataset_id)
    with Image.open(sample.label) as image:
        label_size = image.size
        raw = np.asarray(image.crop(sample.window) if sample.window else image)
    encoded = nd.encode_label(spec, raw)
    alpha_labelled = 0
    with Image.open(sample.image) as image:
        image_size = image.size
        if image_size != label_size:
            raise ValueError(f"{sample.name}: 影像 {image_size} 与标签 {label_size} 尺寸不同")
        try:
            image.load()  # an undecodable file fails here, not later in training
        except (OSError, ValueError) as exc:  # decoder error / truncated raw strips
            raise OSError(f"{sample.image}: 影像无法完整解码（{exc}）") from exc
        if sample.window:
            # Patch of a large tile: written as lossless PNG, not linked.
            image.crop(sample.window).save(
                os.path.join(root, sample.group, "images", sample.name))
        if spec.alpha_is_nodata and image.mode == "RGBA":
            alpha = np.asarray(image)[..., 3]
            alpha_labelled = int(((alpha == 0) & (encoded != nd.IGNORE_INDEX)).sum())

    if not sample.window:
        os.link(sample.image, os.path.join(root, sample.group, "images", sample.name))
    Image.fromarray(encoded, mode="L").save(
        os.path.join(root, sample.group, "labels", nd.label_name(sample.name)))
    if spec.label_mode == "index":
        raw_counts = np.bincount(raw.ravel(), minlength=256)
    else:
        raw_counts = None
    return (sample.name, np.bincount(encoded.ravel(), minlength=256), raw_counts,
            alpha_labelled)


def _git_state():
    try:
        head = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(spec, samples, split, recipe=None):
    rows = Counter((split[s.name], s.group) for s in samples)
    by_split = Counter(split[s.name] for s in samples)
    print(f"数据集   : {spec.title}（{spec.num_classes} 类，ignore={nd.IGNORE_INDEX}）")
    print(f"样本总数 : {len(samples)}")
    for name in spec.split_names:
        groups = sorted({g for (sp, g) in rows if sp == name})
        print(f"  {name:<13} {by_split.get(name, 0):>7} 张，{len(groups)} 组")
    excluded = (recipe or {}).get("excluded", {}).get("names")
    if excluded:
        print(f"  不进入任何划分 {len(excluded)} 张（{recipe['excluded']['rule']}）")
    return by_split


def build_metadata(spec, src, samples, split, recipe, stats, alpha_labelled, root, limit):
    splits = {}
    for name in spec.split_names:
        members = sorted(s.name for s in samples if split[s.name] == name)
        if not members:
            continue
        list_path = os.path.join(root, f"{name}.txt")
        with open(list_path, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{m}\n" for m in members))
        model_counts, raw_counts = stats[name]
        pixels = {cls: int(model_counts[i]) for i, cls in enumerate(spec.class_names)}
        pixels["ignore"] = int(model_counts[nd.IGNORE_INDEX])
        entry = {
            "samples": len(members),
            "groups": sorted({s.group for s in samples if split[s.name] == name}),
            "list_sha256": nd.sha256_file(list_path),
            "pixels": pixels,
        }
        if raw_counts is not None:
            entry["raw_value_pixels"] = {str(v): int(c) for v, c in enumerate(raw_counts) if c}
        splits[name] = entry
    module_path = PROJECT_ROOT / "common" / "native_datasets.py"
    return {
        "format": nd.NATIVE_FORMAT,
        "dataset_id": spec.dataset_id,
        "title": spec.title,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": os.path.realpath(src),
        "class_names": list(spec.class_names),
        "num_classes": spec.num_classes,
        "ignore_index": nd.IGNORE_INDEX,
        "label_mode": spec.label_mode,
        "raw_to_model": {str(k): v for k, v in spec.raw_to_model.items()},
        "spec_sha256": nd.spec_fingerprint(spec),
        "layout": {
            "image_glob": f"*/images/*{spec.image_ext}",
            "label_replace": [["images", "labels"]],
            "label_suffix": nd.LABEL_SUFFIX,
            "rgb_bands": spec.rgb_bands,
            "all_bands": spec.all_bands,
        },
        "split_recipe": recipe,
        "subset": {"limit_per_group": limit} if limit else None,
        "split_id": nd.split_id({"splits": splits}),
        "splits": splits,
        "audit": ({"labelled_pixels_under_alpha0": alpha_labelled}
                  if spec.alpha_is_nodata else {}),
        "generator": {
            "tool_sha256": nd.sha256_file(__file__),
            "module_sha256": nd.sha256_file(module_path),
            "git": _git_state(),
        },
    }


def prepare(spec, src, dst, samples, split, recipe, workers, limit):
    dst = os.path.abspath(dst)
    if os.path.lexists(dst):
        sys.exit(f"❌ 目标已存在，拒绝覆盖：{dst}（准备好的数据只读；请换一个版本目录名）")
    assert_not_other_land_cover_repo(os.path.dirname(dst), PROJECT_ROOT)
    tmp = os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.tmp-{os.getpid()}")
    os.makedirs(tmp)
    try:
        if (samples[0].window is None
                and os.stat(tmp).st_dev != os.stat(samples[0].image).st_dev):
            raise OSError("目标与原始影像不在同一文件系统，无法硬链接")
        for group in sorted({s.group for s in samples}):
            os.makedirs(os.path.join(tmp, group, "images"))
            os.makedirs(os.path.join(tmp, group, "labels"))
        stats = {name: [np.zeros(256, np.int64),
                        np.zeros(256, np.int64) if spec.label_mode == "index" else None]
                 for name in spec.split_names}
        alpha_labelled = 0
        tasks = [(spec.dataset_id, s, tmp) for s in samples]
        step = max(1, len(tasks) // 20)
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for done, (name, model_counts, raw_counts, alpha) in enumerate(
                    pool.imap_unordered(_process, tasks, chunksize=32), 1):
                entry = stats[split[name]]
                entry[0] += model_counts
                if raw_counts is not None:
                    entry[1] += raw_counts
                alpha_labelled += alpha
                if done % step == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)}", flush=True)
        meta = build_metadata(spec, src, samples, split, recipe, stats, alpha_labelled,
                              tmp, limit)
        with open(os.path.join(tmp, nd.METADATA_NAME), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        nd.load_metadata(tmp, spec.dataset_id)  # self-check before publishing
        os.rename(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # links and new labels only; raw data untouched
        raise
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(nd.SPECS))
    ap.add_argument("--src", help="原始数据目录（默认 $LAND_COVER_DATASETS/raw/<目录>）")
    ap.add_argument("--dst", help="输出目录（必须不存在）；--dry-run 时可省略")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, help="覆盖默认的 val 比例")
    ap.add_argument("--test-fraction", type=float, help="覆盖默认的 test 比例（仅 DeepGlobe）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-per-group", type=int, default=0,
                    help="每组只取前 N 张（冒烟测试）")
    ap.add_argument("--dry-run", action="store_true", help="只打印划分统计，不写文件")
    args = ap.parse_args(argv)

    spec = nd.get_spec(args.dataset)
    src = args.src
    if src is None:
        base = os.environ.get("LAND_COVER_DATASETS")
        if not base:
            sys.exit("❌ 未给 --src，且没有设置 LAND_COVER_DATASETS")
        src = os.path.join(base, "raw", spec.raw_dir)
    fractions = dict(spec.default_fractions)
    if args.val_fraction is not None:
        fractions["val"] = args.val_fraction
    if args.test_fraction is not None:
        fractions["test"] = args.test_fraction

    samples = nd.list_samples(spec.dataset_id, src)
    split, recipe = nd.assign_splits(spec, samples, args.seed, fractions)
    samples = [s for s in samples if s.name in split]  # drop excluded tiles
    if args.limit_per_group:
        kept, seen = [], Counter()
        for s in samples:
            if seen[s.group] < args.limit_per_group:
                kept.append(s)
                seen[s.group] += 1
        samples = kept
    by_split = summarize(spec, samples, split, recipe)
    if recipe["held_out_units"]:
        for part, units in recipe["held_out_units"].items():
            print(f"  {part} 单元：{', '.join(units)}")
    if spec.expected_counts and not args.limit_per_group:
        actual = {name: by_split.get(name, 0) for name in spec.split_names}
        if actual != spec.expected_counts:
            sys.exit(f"❌ 划分数量 {actual} 与代码中冻结的 {spec.expected_counts} 不一致，拒绝生成")
        print("  数量与冻结的划分一致")
    if args.dry_run:
        print("Dry-run：未写任何文件。")
        return 0
    if not args.dst:
        sys.exit("❌ 正式生成需要 --dst")
    meta = prepare(spec, src, args.dst, samples, split, recipe, args.workers,
                   args.limit_per_group)
    print(f"✅ 已生成 {args.dst}")
    for name, info in meta["splits"].items():
        total = sum(info["pixels"].values())
        print(f"  {name}: {info['samples']} 张，ignore 像素占 "
              f"{info['pixels']['ignore'] / max(total, 1):.2%}")
    if meta["audit"]:
        print(f"  audit: {meta['audit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
