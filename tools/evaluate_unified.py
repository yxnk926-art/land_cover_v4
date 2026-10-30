#!/usr/bin/env python3
"""Final test-split evaluation for the unified fine-class protocol v4.

A checkpoint trained by ``common/train.py --config config_unified_<model>``
is evaluated on every dataset's test splits with that dataset's own classes:
the fine-class probabilities are summed over each native class's set (see
``configs/class_maps/unified_fine_v2.json``)::

    python tools/evaluate_unified.py --model unet_r50 \\
        --checkpoint checkpoints/unified_v4/all/baselines/<tag>/unet_r50/best_model.pth

By default the datasets the checkpoint was trained on are evaluated;
``--datasets`` may name others of the taxonomy (held-out evaluation).  The
checkpoint must carry the same taxonomy, label stacking, model and training
splits as the current config.  Results go to ``evaluations/unified_v4/<scope>/
<model>/<checkpoint sha256[:12]>/`` (never overwritten): ``metrics.json`` with
per-split native metrics plus the fine-class confusion (rows = native ground
truth, columns = fine argmax), and ``scores.txt``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import tempfile
import types
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import unified_protocol as up  # noqa: E402
from common.unified_taxonomy import load_taxonomy  # noqa: E402

# A checkpoint is only evaluated under the config it was trained with.
CHECKPOINT_GUARD_KEYS = (
    "PROTOCOL_ID", "UNIFIED_TAXONOMY_ID", "UNIFIED_TAXONOMY_SHA256", "FINE_CLASS_NAMES",
    "UNIFIED_DATASET_IDS", "LABEL_OFFSETS", "SET_MEMBERSHIP", "MODEL_NAME",
    "IN_CHANNELS", "NUM_CLASSES", "IGNORE_INDEX",
)
# Per training dataset: the same prepared copy, split and class definition.
DATASET_GUARD_KEYS = ("split_id", "spec_sha256", "class_names")
SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return repr(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_snapshot(config) -> dict[str, Any]:
    return {
        name: _jsonable(getattr(config, name))
        for name in sorted(dir(config))
        if name.isupper()
    }


def _is_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unwrap_checkpoint(payload: Any) -> tuple[dict[str, torch.Tensor], Mapping[str, Any] | None, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint 必须是 state_dict mapping 或含 model/state_dict 的 mapping")
    if payload.get("kind") == "numerical_failure":
        raise ValueError("数值失败快照仅供诊断，禁止作为评估 checkpoint")
    if isinstance(payload.get("model"), Mapping):
        state = dict(payload["model"])
        metadata: Mapping[str, Any] | None = payload
        kind = "model_payload"
    elif isinstance(payload.get("state_dict"), Mapping):
        state = dict(payload["state_dict"])
        metadata = payload
        kind = "state_dict_payload"
    elif payload and all(isinstance(key, str) and torch.is_tensor(value)
                         for key, value in payload.items()):
        state = dict(payload)
        metadata = None
        kind = "raw_state_dict"
    else:
        raise ValueError("checkpoint 中找不到有效 model/state_dict")
    if not state or not all(isinstance(key, str) and torch.is_tensor(value)
                            for key, value in state.items()):
        raise ValueError("checkpoint state_dict 含非 tensor 项或为空")
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state, metadata, kind


def configure_runtime(config) -> torch.device:
    seed = int(getattr(config, "SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic = bool(getattr(config, "DETERMINISTIC", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=bool(getattr(config, "DETERMINISTIC_WARN_ONLY", False)),
    )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "pillow": _version("Pillow"),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def safe_component(value: str) -> str:
    cleaned = SAFE_COMPONENT.sub("-", value).strip("-._")
    return cleaned or "unnamed"


def serialize_scores(scores: Mapping[str, Any], config) -> dict[str, Any]:
    names = list(config.CLASS_NAMES)
    iou = np.asarray(scores["per_class_iou"], dtype=float)
    f1 = np.asarray(scores["per_class_f1"], dtype=float)
    present = np.asarray(scores["valid_classes"], dtype=bool)
    matrix = np.asarray(scores["confusion_matrix"], dtype=np.int64)
    payload = {
        "miou_present": float(scores["miou"]),
        "miou_fixed": float(scores["miou_fixed"]),
        "mf1_present": float(scores["mf1"]),
        "mf1_fixed": float(scores["mf1_fixed"]),
        "oa": float(scores["oa"]),
        "classes": [{
            "id": i, "name": name, "support": int(scores["per_class_support"][i]),
            "present": bool(present[i]), "iou": float(iou[i]), "f1": float(f1[i]),
        } for i, name in enumerate(names)],
        "confusion_matrix": matrix.tolist(),
    }
    excluded = tuple(getattr(config, "REPORT_EXCLUDE_CLASSES", ()))
    if excluded:
        reported = np.array([name not in excluded for name in names])
        kept = present & reported
        payload["excluded_classes"] = list(excluded)
        payload["miou_present_excluding"] = float(iou[kept].mean()) if kept.any() else 0.0
        payload["mf1_present_excluding"] = float(f1[kept].mean()) if kept.any() else 0.0
        # Pixels whose ground truth is a reported class; predicting an excluded
        # class there counts as an error.
        rows = int(matrix[reported].sum())
        payload["oa_excluding"] = float(np.diag(matrix)[reported].sum() / max(rows, 1))
    return payload


def write_artifacts(output: Path, result: Mapping[str, Any], text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".unified-eval-", dir=output.parent))
    try:
        with open(stage / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(_jsonable(result), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        (stage / "scores.txt").write_text(text.rstrip() + "\n", encoding="utf-8")
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(up.MODEL_CONFIGS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="默认评估 checkpoint 训练用的全部数据集；可加入细类体系中的其他数据集")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def read_checkpoint(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    state, metadata, _ = _unwrap_checkpoint(_torch_load(path))
    saved = metadata.get("config") if metadata is not None else None
    if not isinstance(saved, Mapping) or "UNIFIED_DATASET_IDS" not in saved:
        raise ValueError("checkpoint 缺少统一细类协议的 config 元数据")
    return path, state, metadata, saved


def check_checkpoint(saved: Mapping[str, Any], config) -> None:
    for key in CHECKPOINT_GUARD_KEYS:
        if key not in saved:
            raise ValueError(f"checkpoint 元数据缺少 {key}")
        expected = _jsonable(getattr(config, key))
        if _jsonable(saved[key]) != expected:
            raise ValueError(f"checkpoint {key} 与当前 config 不一致")
    for dataset_id, entry in config.UNIFIED_DATASETS.items():
        old = saved.get("UNIFIED_DATASETS", {}).get(dataset_id, {})
        for key in DATASET_GUARD_KEYS:
            if _jsonable(old.get(key)) != _jsonable(entry[key]):
                raise ValueError(f"checkpoint 中 {dataset_id} 的 {key} 与当前准备副本不一致")


def resolve_datasets(config, taxonomy, requested: Sequence[str] | None):
    dataset_ids = list(requested or config.UNIFIED_DATASET_IDS)
    unknown = [d for d in dataset_ids if d not in taxonomy.datasets]
    if unknown or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError(f"--datasets 只能取细类体系中的数据集：{', '.join(taxonomy.datasets)}")
    entries = {}
    for dataset_id in dataset_ids:
        entries[dataset_id] = (config.UNIFIED_DATASETS.get(dataset_id)
                               or up.dataset_entry(config.DATASETS_ROOT, dataset_id,
                                                   config.NATIVE_DATASETS))
    return entries


def build_loader(config, entry: Mapping[str, Any], list_path: str,
                 num_workers: int | None) -> DataLoader:
    from common.dataset import OpenEarthMapDataset

    dataset = OpenEarthMapDataset(
        entry["root"], list_path, training=False, mean=config.MEAN, std=config.STD,
        pad_divisor=config.PAD_DIVISOR, ignore_index=config.IGNORE_INDEX,
        image_pad_mode=config.IMAGE_PAD_MODE, image_glob=entry["image_glob"],
        label_replace=entry["label_replace"], label_suffix=entry["label_suffix"],
        image_bands=entry["image_bands"], dtype_scale=config.DTYPE_SCALE,
        num_classes=entry["num_classes"], strict_label_schema=config.STRICT_LABEL_SCHEMA)
    workers = config.NUM_WORKERS if num_workers is None else num_workers
    return DataLoader(dataset, batch_size=entry["val_batch_size"], shuffle=False,
                      num_workers=workers, pin_memory=True)


def resolve_output_dir(args: argparse.Namespace, config, checkpoint_sha: str,
                       entries: Mapping[str, Any]) -> Path:
    if args.output_dir is None:
        output = (PROJECT_ROOT / "evaluations" / "unified_v4" / safe_component(config.RUN_SCOPE)
                  / safe_component(config.MODEL_SLUG) / checkpoint_sha[:12])
    else:
        output = args.output_dir.expanduser()
        if not output.is_absolute():
            output = PROJECT_ROOT / output
    output = output.resolve()
    protected = [config.CKPT_DIR, config.OUT_DIR] + [e["root"] for e in entries.values()]
    for parent in protected:
        parent = Path(parent).expanduser().resolve()
        if _is_within(output, parent):
            raise ValueError(f"评估输出不得写入数据或训练目录：{output} 位于 {parent}")
    if output.exists():
        raise FileExistsError(f"评估输出已存在，拒绝覆盖：{output}")
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    from common.engine import evaluate, format_scores
    from common.models import build_model

    path, state, metadata, saved = read_checkpoint(args.checkpoint)
    config = up.import_unified_config(up.MODEL_CONFIGS[args.model],
                                      list(saved["UNIFIED_DATASET_IDS"]))
    check_checkpoint(saved, config)
    taxonomy = load_taxonomy(config.TAXONOMY_PATH)
    entries = resolve_datasets(config, taxonomy, args.datasets)
    checkpoint = {
        "path": str(path), "sha256": file_sha256(path), "kind": metadata.get("kind", "last"),
        "epoch": metadata.get("epoch"), "protocol_fingerprint": metadata.get("protocol_fingerprint"),
        "selection_miou": metadata.get("source_selection_value", metadata.get("miou")),
        "selection_by_dataset": metadata.get("source_selection_by_dataset"),
    }
    output = resolve_output_dir(args, config, checkpoint["sha256"], entries)
    device = configure_runtime(config)
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    results, texts = {}, []
    for dataset_id, entry in entries.items():
        membership = taxonomy.membership(dataset_id)
        report = types.SimpleNamespace(CLASS_NAMES=entry["class_names"],
                                       REPORT_EXCLUDE_CLASSES=tuple(entry["report_exclude"]))
        splits = {}
        for split, list_path in entry["test_lists"].items():
            scores = evaluate(model, build_loader(config, entry, list_path, args.num_workers),
                              device, entry["num_classes"], ignore_index=int(config.IGNORE_INDEX),
                              amp=bool(config.EVAL_AMP), window=entry["eval_window"],
                              stride=entry["eval_stride"], projection=membership)
            metrics = serialize_scores(scores, report)
            metrics["fine_confusion"] = {
                "rows": list(entry["class_names"]), "columns": list(config.FINE_CLASS_NAMES),
                "counts": np.asarray(scores["fine_confusion"]).tolist(),
            }
            splits[split] = {"list": list_path, "list_sha256": file_sha256(list_path),
                             "metrics": metrics}
            text = (f"== {dataset_id} / {split}\n"
                    + format_scores(scores, entry["class_names"]))
            if "excluded_classes" in metrics:
                text += (f"\n不含 {', '.join(metrics['excluded_classes'])}："
                         f"mIoU {metrics['miou_present_excluding']:.4f} | "
                         f"mF1 {metrics['mf1_present_excluding']:.4f} | "
                         f"OA {metrics['oa_excluding']:.4f}")
            texts.append(text)
        results[dataset_id] = {
            "trained_on": dataset_id in config.UNIFIED_DATASET_IDS,
            "dataset": {key: entry[key] for key in (
                "root", "split_id", "spec_sha256", "metadata_sha256", "selection_split")},
            "membership": membership.tolist(),
            "splits": splits,
        }

    resolved = config_snapshot(config)
    result = {
        "schema_version": 1,
        "evaluation_kind": "unified_test",
        "selection_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy": {"id": config.UNIFIED_TAXONOMY_ID, "version": config.UNIFIED_TAXONOMY_VERSION,
                     "status": config.UNIFIED_TAXONOMY_STATUS,
                     "sha256": config.UNIFIED_TAXONOMY_SHA256,
                     "fine_classes": list(config.FINE_CLASS_NAMES)},
        "config": {"module": config.__name__, "sha256": canonical_sha256(resolved),
                   "model_name": config.MODEL_NAME, "resolved": resolved},
        "checkpoint": checkpoint,
        "evaluation": {"eval_amp": bool(config.EVAL_AMP)},
        "datasets": results,
        "metric_policy": {
            "native_scores": "each native class scores the summed probability of its "
                             "fine-class set; the highest-scoring native class is the prediction",
            "miou_present": "mean over classes present in the split's ground truth",
            "miou_fixed": "mean over every class of the dataset; absent class IoU=0",
            "fine_confusion": "pixel counts, rows = native ground truth, columns = fine-class "
                              "argmax; ignored pixels not counted",
            "model_selection": "forbidden on test splits; training selects on the mean "
                               "native mIoU of the selection splits",
        },
        "runtime": runtime_versions(device),
    }
    text = "\n\n".join(texts)
    write_artifacts(output, result, text)
    result["output_dir"] = str(output)
    print(text)
    print(f"[unified-eval] checkpoint sha256={checkpoint['sha256']}")
    print(f"[unified-eval] 输出：{output}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
