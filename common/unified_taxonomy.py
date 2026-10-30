# -*- coding: utf-8 -*-
"""Unified fine-class taxonomy: fine classes and the native classes made of them.

A taxonomy file (``configs/class_maps/unified_fine_v2.json``) lists fine
classes U and, for every fine class and dataset, the native class(es) of that
dataset the fine class belongs to (``map.<dataset>.to``).  The native class c
of dataset d is then the set S_d(c) of fine classes mapped to it, and the 0/1
matrix ``membership(d)`` (C_d x |U|) is what training (set cross-entropy) and
evaluation (sum of fine probabilities per native class) use.

A fine class should belong to exactly one native class per dataset: set
cross-entropy never penalises a fine class on pixels of either of its two
native classes, and ``car`` in taxonomy v1 (road and developed space in OEM,
road and background in Global-LoveDA) took 92-95% of the road pixels that way.
The loader still accepts such a file so that v1 stays readable; a test holds
the current file free of overlaps.  ``load_taxonomy`` rejects a file in which
the native sets cannot work:

* native class names must be those of ``common.native_datasets`` (same order);
* every fine class is mapped in every dataset, and every native class gets at
  least one fine class;
* no native set may contain another native set of the same dataset (the
  larger set would always win);
* every two fine classes must be told apart by at least one dataset, otherwise
  nothing in the data can teach the difference.

``python -m common.unified_taxonomy`` prints the mapping tables and the
review list (uncertain cells, overlaps, classes no dataset labels alone).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import native_datasets as nd  # noqa: E402

BASIS = ("official", "inferred", "uncertain")
# A draft may be trained for smoke tests and pilots; formal (--run-tag) runs need final.
STATUS = ("draft", "final")
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "class_maps", "unified_fine_v2.json")


@dataclass(frozen=True)
class Cell:
    """Where one fine class goes in one dataset."""
    to: tuple
    basis: str
    alternatives: tuple = ()
    note: str = ""


@dataclass(frozen=True)
class Taxonomy:
    path: str
    sha256: str
    taxonomy_id: str
    version: int
    status: str
    datasets: tuple
    fine_keys: tuple
    fine_names: tuple
    descriptions: tuple
    cells: dict  # (fine key, dataset id) -> Cell

    def class_names(self, dataset_id):
        return nd.get_spec(dataset_id).class_names

    def native_sets(self, dataset_id):
        """Native class name -> tuple of fine keys, in native class order."""
        sets = {name: [] for name in self.class_names(dataset_id)}
        for key in self.fine_keys:
            for name in self.cells[key, dataset_id].to:
                sets[name].append(key)
        return {name: tuple(keys) for name, keys in sets.items()}

    def membership(self, dataset_id):
        """0/1 matrix [C_d, |U|]: row c marks the fine classes of native class c."""
        index = {key: i for i, key in enumerate(self.fine_keys)}
        sets = self.native_sets(dataset_id)
        matrix = np.zeros((len(sets), len(self.fine_keys)), dtype=np.int64)
        for row, keys in enumerate(sets.values()):
            matrix[row, [index[key] for key in keys]] = 1
        return matrix

    def overlaps(self):
        """dataset -> fine keys mapped to more than one native class."""
        return {d: tuple(k for k in self.fine_keys if len(self.cells[k, d].to) > 1)
                for d in self.datasets}

    def anchored_by(self, key):
        """Datasets with a native class made of this fine class alone."""
        return tuple(d for d in self.datasets
                     if any(keys == (key,) for keys in self.native_sets(d).values()))


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_taxonomy(path=DEFAULT_PATH):
    """Read and validate a taxonomy file; any violation raises ValueError."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    for field in ("taxonomy_id", "version", "status", "datasets", "fine_classes"):
        if field not in raw:
            raise ValueError(f"{path} 缺少字段 {field}")
    if raw["status"] not in STATUS:
        raise ValueError(f"status 必须是 {'/'.join(STATUS)}，收到 {raw['status']!r}")
    datasets = tuple(raw["datasets"])
    if len(set(datasets)) != len(datasets):
        raise ValueError("datasets 有重复")
    for dataset_id in datasets:
        nd.get_spec(dataset_id)  # unknown dataset ids fail here
    keys, names, descriptions, cells = [], [], [], {}
    for entry in raw["fine_classes"]:
        key = entry.get("key")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError(f"细类 key 缺失或重复：{key!r}")
        keys.append(key)
        names.append(str(entry.get("name", key)))
        descriptions.append(str(entry.get("description", "")))
        mapping = entry.get("map", {})
        extra = set(mapping) - set(datasets)
        if extra:
            raise ValueError(f"细类 {key} 映射了未登记的数据集：{sorted(extra)}")
        for dataset_id in datasets:
            if dataset_id not in mapping:
                raise ValueError(f"细类 {key} 没有给出在 {dataset_id} 中的归属")
            cells[key, dataset_id] = _parse_cell(key, dataset_id, mapping[dataset_id])
    taxonomy = Taxonomy(
        path=path, sha256=_sha256(path), taxonomy_id=str(raw["taxonomy_id"]),
        version=int(raw["version"]), status=str(raw["status"]), datasets=datasets,
        fine_keys=tuple(keys), fine_names=tuple(names), descriptions=tuple(descriptions),
        cells=cells)
    _check_sets(taxonomy)
    return taxonomy


def _parse_cell(key, dataset_id, value):
    native = nd.get_spec(dataset_id).class_names
    where = f"细类 {key} 在 {dataset_id} 中"
    to = tuple(value.get("to", ()))
    if not to or len(set(to)) != len(to):
        raise ValueError(f"{where}的归属为空或重复：{list(to)}")
    alternatives = tuple(value.get("alternatives", ()))
    for name in to + alternatives:
        if name not in native:
            raise ValueError(f"{where}引用了不存在的原生类 {name!r}（{dataset_id}: {', '.join(native)}）")
    if set(to) & set(alternatives):
        raise ValueError(f"{where}的备选与归属重复")
    basis = value.get("basis")
    if basis not in BASIS:
        raise ValueError(f"{where}的 basis 必须是 {'/'.join(BASIS)}，收到 {basis!r}")
    if basis == "official" and alternatives:
        raise ValueError(f"{where}标为 official 却给了备选")
    return Cell(to=to, basis=basis, alternatives=alternatives, note=str(value.get("note", "")))


def _check_sets(taxonomy):
    for dataset_id in taxonomy.datasets:
        sets = taxonomy.native_sets(dataset_id)
        empty = [name for name, keys in sets.items() if not keys]
        if empty:
            raise ValueError(f"{dataset_id} 的原生类 {empty} 没有任何细类")
        for (a, sa), (b, sb) in itertools.permutations(sets.items(), 2):
            if set(sb) <= set(sa):
                raise ValueError(
                    f"{dataset_id} 的 {a!r} 包含 {b!r} 的全部细类，按集合概率评估时 {b!r} 永远不会被选中")
    for a, b in itertools.combinations(taxonomy.fine_keys, 2):
        if all(set(taxonomy.cells[a, d].to) == set(taxonomy.cells[b, d].to)
               for d in taxonomy.datasets):
            raise ValueError(f"细类 {a} 和 {b} 在每个数据集里的归属都相同，数据无法区分，应合并")


# ---------------------------------------------------------------- report
_BASIS_MARK = {"official": "", "inferred": " †", "uncertain": " ?"}


def markdown(taxonomy):
    """Mapping tables and review lists in Markdown (the review document's body)."""
    short = {"openearthmap": "OEM", "flair_hub": "FLAIR-HUB", "global_loveda": "Global-LoveDA",
             "deepglobe": "DeepGlobe", "isprs_potsdam": "Potsdam"}
    title = lambda d: short.get(d, d)  # noqa: E731
    lines = ["| # | 细类 | " + " | ".join(title(d) for d in taxonomy.datasets) + " |",
             "|---|---|" + "---|" * len(taxonomy.datasets)]
    for i, key in enumerate(taxonomy.fine_keys):
        cells = []
        for d in taxonomy.datasets:
            cell = taxonomy.cells[key, d]
            cells.append(" + ".join(cell.to) + _BASIS_MARK[cell.basis])
        lines.append(f"| {i} | `{key}` | " + " | ".join(cells) + " |")

    lines += ["", "| 数据集 | 原生类 = 细类集合 |", "|---|---|"]
    for d in taxonomy.datasets:
        parts = [f"**{name}** = {{{', '.join(keys)}}}"
                 for name, keys in taxonomy.native_sets(d).items()]
        lines.append(f"| {title(d)} | " + "<br>".join(parts) + " |")

    uncertain = [(key, d, taxonomy.cells[key, d]) for key in taxonomy.fine_keys
                 for d in taxonomy.datasets if taxonomy.cells[key, d].basis == "uncertain"]
    lines += ["", f"待核对的格子（{len(uncertain)} 个）：", "",
              "| 细类 | 数据集 | 草案 | 备选 | 说明 |", "|---|---|---|---|---|"]
    for key, d, cell in uncertain:
        lines.append(f"| `{key}` | {title(d)} | {' + '.join(cell.to)} | "
                     f"{', '.join(cell.alternatives) or '—'} | {cell.note or '—'} |")

    overlaps = {d: keys for d, keys in taxonomy.overlaps().items() if keys}
    lines += ["", "一个细类对应两个原生类（训练时两者都接受）：" + (
        "；".join(f"{title(d)}：{', '.join(keys)}" for d, keys in overlaps.items()) or "无")]
    alone = [(key, taxonomy.anchored_by(key)) for key in taxonomy.fine_keys]
    lines += ["", "有数据集单独标注的细类：" + "，".join(
        f"`{key}`（{'、'.join(title(d) for d in ds)}）" for key, ds in alone if ds)]
    lines += ["", "只能靠数据集之间的交集学到的细类：" + (
        "，".join(f"`{key}`" for key, ds in alone if not ds) or "无")]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="校验并打印统一细类体系")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    taxonomy = load_taxonomy(args.path)
    print(f"{taxonomy.taxonomy_id} v{taxonomy.version} ({taxonomy.status}) "
          f"sha256={taxonomy.sha256}: {len(taxonomy.fine_keys)} 个细类，"
          f"{len(taxonomy.datasets)} 个数据集\n")
    print(markdown(taxonomy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
