# -*- coding: utf-8 -*-
"""Unified fine-class protocol v4: registry, per-dataset entries, evaluation.

One model is trained on several native prepared copies at once and predicts
the fine classes of a taxonomy (``common/unified_taxonomy.py``).  Labels stay
native: during training a per-dataset offset turns dataset d's class c into
the global label ``LABEL_OFFSETS[d] + c``, whose row of ``SET_MEMBERSHIP``
lists the fine classes allowed there.  Evaluation scores every dataset on its
own classes by summing the fine probabilities of each native class's set.

``configs/config_unified_v4.py`` reads the training datasets from an
environment variable at import time; tools go through ``import_unified_config``,
which drops the cached unified modules first.
"""
import importlib
import os
import sys

import numpy as np
import torch

from common import native_datasets as nd

DATASETS_ENV = "LAND_COVER_UNIFIED_DATASETS"
CONFIG_PREFIX = "configs.config_unified_"

# Launcher / evaluator key -> leaf module, fastest model first (the default queue).
MODEL_CONFIGS = {
    "ddrnet": "config_unified_ddrnet",
    "mobilenetv3_lraspp_in1k": "config_unified_mobilenetv3_lraspp_in1k",
    "pidnet_s_in1k": "config_unified_pidnet_s_in1k",
    "segformer_b0_in1k": "config_unified_segformer_b0_in1k",
    "segmenter_s16_in1k": "config_unified_segmenter_s16_in1k",
    "deeplabv3plus_r50": "config_unified_deeplabv3plus_r50",
    "unet_r50": "config_unified_unet_r50",
    "segformer_b2_in1k": "config_unified_segformer_b2_in1k",
    "unetpp_r50": "config_unified_unetpp_r50",
    "swin_upernet": "config_unified_swin_upernet",
}


# ---------------------------------------------------------------- config helpers
def training_datasets(taxonomy, requested=None):
    """Training datasets in taxonomy order: all of them, or the comma list ``requested``."""
    if not requested:
        return tuple(taxonomy.datasets)
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = sorted(set(names) - set(taxonomy.datasets))
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(f"{DATASETS_ENV}={requested!r} 无效：只能是 "
                         f"{', '.join(taxonomy.datasets)} 中不重复的若干个")
    return tuple(d for d in taxonomy.datasets if d in names)


def run_scope(taxonomy, dataset_ids):
    """Artifact directory name: ``all`` or the datasets joined by ``+``."""
    return "all" if tuple(dataset_ids) == tuple(taxonomy.datasets) else "+".join(dataset_ids)


def dataset_entry(datasets_root, dataset_id, native_datasets):
    """Everything the unified protocol needs about one native prepared copy.

    ``native_datasets`` is the prepared-copy table of ``config_unified_v4``.
    Training reads ``train_list`` and ``val_list`` (the selection split);
    ``test_lists`` are for the final evaluation only.
    """
    settings = native_datasets[dataset_id]
    root = os.path.join(datasets_root, "prepared", settings["prepared"])
    values = nd.config_values(root, bands="rgb")
    meta = nd.load_metadata(root, dataset_id, verify_lists=False)
    if "VAL_LIST" not in values:
        raise ValueError(f"{root} 没有选择划分 {values['SELECTION_SPLIT']}.txt，无法选模型")
    window, stride = settings.get("eval_window", (None, None))
    return {
        "prepared": settings["prepared"],
        "root": root,
        "split_id": values["SPLIT_ID"],
        "spec_sha256": meta["spec_sha256"],
        "metadata_sha256": nd.sha256_file(os.path.join(root, nd.METADATA_NAME)),
        "class_names": list(values["CLASS_NAMES"]),
        "num_classes": int(values["NUM_CLASSES"]),
        "image_glob": values["IMAGE_GLOB"],
        "label_replace": values["LABEL_REPLACE"],
        "label_suffix": values["LABEL_SUFFIX"],
        "image_bands": values["IMAGE_BANDS"],
        "selection_split": values["SELECTION_SPLIT"],
        "train_list": values["TRAIN_LIST"],
        "val_list": values["VAL_LIST"],
        "train_list_sha256": meta["splits"]["train"]["list_sha256"],
        "val_list_sha256": meta["splits"][values["SELECTION_SPLIT"]]["list_sha256"],
        "test_lists": {name: values[f"{name.upper()}_LIST"] for name in values["TEST_SPLITS"]},
        "val_batch_size": int(settings.get("val_batch_size", 1)),
        "eval_window": window,
        "eval_stride": stride,
        "report_exclude": list(settings.get("report_exclude", ())),
    }


def global_membership(taxonomy, dataset_ids):
    """Label offsets and the stacked membership matrix of ``dataset_ids``.

    Dataset d's native class c becomes global label ``offsets[d] + c``; row
    ``offsets[d] + c`` of the matrix marks the fine classes of that class.
    """
    offsets, rows = {}, []
    for dataset_id in dataset_ids:
        offsets[dataset_id] = len(rows)
        rows.extend(taxonomy.membership(dataset_id).tolist())
    if len(rows) >= nd.IGNORE_INDEX:
        raise ValueError(f"全局标签数 {len(rows)} 超过 ignore 值 {nd.IGNORE_INDEX}")
    return offsets, rows


def import_unified_config(config_name, dataset_ids=None):
    """Import a unified leaf for ``dataset_ids`` (None: every taxonomy dataset)."""
    if not config_name.startswith("config_unified_"):
        raise ValueError(f"{config_name} 不是统一细类配置（config_unified_*）")
    if dataset_ids:
        os.environ[DATASETS_ENV] = ",".join(dataset_ids)
    else:
        os.environ.pop(DATASETS_ENV, None)
    for name in [name for name in sys.modules if name.startswith(CONFIG_PREFIX)]:
        del sys.modules[name]
    config = importlib.import_module(f"configs.{config_name}")
    if dataset_ids and set(config.UNIFIED_DATASET_IDS) != set(dataset_ids):
        raise ValueError(f"{config_name} 读到的数据集是 {config.UNIFIED_DATASET_IDS}")
    return config


# ---------------------------------------------------------------- training logs
class DatasetLossMeter:
    """Set cross-entropy of the main output per training dataset, for the history only.

    The training loss mixes the datasets of a batch.  The label offset of each
    pixel tells its dataset, so the loss can be split to show which dataset
    overfits.  Training calls this every ``LOG_INTERVAL`` iterations.
    """

    def __init__(self, cfg, device):
        self.names = list(cfg.LABEL_OFFSETS)
        self.starts = torch.as_tensor([cfg.LABEL_OFFSETS[d] for d in self.names], device=device)
        self.allowed = torch.as_tensor(cfg.SET_MEMBERSHIP, device=device).bool()
        self.ignore_index = cfg.IGNORE_INDEX
        self.loss = torch.zeros(len(self.names), dtype=torch.float64, device=device)
        self.pixels = torch.zeros(len(self.names), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits, target):
        logits = logits.float()
        if logits.shape[-2:] != target.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=target.shape[-2:],
                                                     mode="bilinear", align_corners=False)
        valid = target != self.ignore_index
        labels = target[valid]
        logits = logits.movedim(1, -1)[valid]
        in_set = self.allowed[labels]
        pixel_loss = (torch.logsumexp(logits, dim=1)
                      - torch.logsumexp(logits.masked_fill(~in_set, float("-inf")), dim=1))
        index = torch.bucketize(labels, self.starts, right=True) - 1
        self.loss.index_add_(0, index, pixel_loss.double())
        self.pixels.index_add_(0, index, torch.ones_like(pixel_loss, dtype=torch.float64))

    def means(self):
        """Mean loss per dataset since the last call; None for a dataset with no pixels."""
        result = {name: (loss / pixels).item() if pixels else None
                  for name, loss, pixels in zip(self.names, self.loss, self.pixels)}
        self.loss.zero_()
        self.pixels.zero_()
        return result


# ---------------------------------------------------------------- evaluation
def projection(taxonomy_membership, device):
    """Float [C_d, |U|] matrix that sums fine probabilities per native class."""
    return torch.as_tensor(np.asarray(taxonomy_membership), dtype=torch.float32, device=device)


def evaluate_unified(model, loaders, device, cfg, amp=False):
    """Native scores of every dataset's selection split and their mean.

    ``scores["miou"]`` (what training selects on) is the mean over datasets of
    the per-dataset native mIoU; ``scores["datasets"][d]`` holds the full
    ``ConfusionMatrix.scores()`` of dataset d plus its fine-class confusion.
    """
    from common.engine import evaluate

    per_dataset = {}
    for dataset_id, loader in loaders.items():
        entry = cfg.UNIFIED_DATASETS[dataset_id]
        offset = cfg.LABEL_OFFSETS[dataset_id]
        rows = cfg.SET_MEMBERSHIP[offset:offset + entry["num_classes"]]
        per_dataset[dataset_id] = evaluate(
            model, loader, device, entry["num_classes"], ignore_index=cfg.IGNORE_INDEX,
            amp=amp, window=entry["eval_window"], stride=entry["eval_stride"],
            projection=projection(rows, device))
    mious = [scores["miou"] for scores in per_dataset.values()]
    return {"miou": float(np.mean(mious)), "datasets": per_dataset}


def format_unified_scores(scores, cfg):
    from common.engine import format_scores

    blocks = []
    for dataset_id, dataset_scores in scores["datasets"].items():
        names = cfg.UNIFIED_DATASETS[dataset_id]["class_names"]
        blocks.append(f"-- {dataset_id}\n" + format_scores(dataset_scores, names))
    summary = " | ".join(f"{d} {s['miou']:.4f}" for d, s in scores["datasets"].items())
    blocks.append(f"选模指标（各数据集原生 mIoU 的均值）{scores['miou']:.4f} | {summary}")
    return "\n".join(blocks)
