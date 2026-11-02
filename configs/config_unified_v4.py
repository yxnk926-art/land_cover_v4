# -*- coding: utf-8 -*-
"""Unified fine-class protocol v4: one model, several datasets, native evaluation.

The model predicts the fine classes of ``configs/class_maps/unified_fine_v2.json``
(see ``common/unified_taxonomy.py``).  Every dataset keeps its own labels: a
pixel of native class c in dataset d is supervised with the set S_d(c) of fine
classes that make up c (set cross-entropy, -log sum_{u in S_d(c)} p_u), and
dataset d is evaluated on its own classes after summing the fine probabilities
over each S_d(c).  The native prepared copies are read unchanged; training adds
a per-dataset label offset so one label map names both dataset and class.

Training datasets: every dataset of the taxonomy, or the comma-separated
subset in ``LAND_COVER_UNIFIED_DATASETS`` (tests; later leave-one-dataset-out
runs).  Each virtual epoch draws the same number of samples from every
dataset, and 40 epochs per dataset give each dataset the iterations x batch
exposure of its native run (5 datasets: 200 x 1000 iterations).  Model
selection uses the mean of the per-dataset native mIoU on the selection
splits; test splits are left to ``tools/evaluate_unified.py``.
"""
import os

from common import native_datasets as _nd
from common import unified_protocol as _up
from common import unified_taxonomy as _ut


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROLE = "source"


# ---------------------------------------------------------------- protocol
PROTOCOL_ID = "unified-fine-v4"
DATASETS_ENV = _up.DATASETS_ENV
# $LAND_COVER_DATASETS, default datasets/ next to the repository.
DATASETS_ROOT = os.path.abspath(os.environ.get(
    "LAND_COVER_DATASETS", os.path.join(os.path.dirname(PROJECT_ROOT), "datasets")))

TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "configs", "class_maps", "unified_fine_v2.json")
_TAXONOMY = _ut.load_taxonomy(TAXONOMY_PATH)
UNIFIED_TAXONOMY_ID = _TAXONOMY.taxonomy_id
UNIFIED_TAXONOMY_VERSION = _TAXONOMY.version
UNIFIED_TAXONOMY_STATUS = _TAXONOMY.status
UNIFIED_TAXONOMY_SHA256 = _TAXONOMY.sha256
FINE_CLASS_NAMES = list(_TAXONOMY.fine_keys)

UNIFIED_DATASET_IDS = list(_up.training_datasets(_TAXONOMY, os.environ.get(DATASETS_ENV)))
RUN_SCOPE = _up.run_scope(_TAXONOMY, UNIFIED_DATASET_IDS)
EXPERIMENT_FAMILY = "unified_v4"
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints", EXPERIMENT_FAMILY, RUN_SCOPE)
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", EXPERIMENT_FAMILY, RUN_SCOPE)


# dataset id -> prepared copy under $LAND_COVER_DATASETS/prepared and the
# per-dataset evaluation settings (everything else is shared).
NATIVE_DATASETS = {
    "openearthmap": {"prepared": "OpenEarthMap_native_v1"},
    # 152k training tiles: validating the 38k-tile selection split is costly,
    # so it runs every 8k iterations (5 times) instead of every 2k.
    "flair_hub": {"prepared": "FLAIR-HUB_native_v1", "eval_interval": 8,
                  "val_batch_size": 16},
    "global_loveda": {"prepared": "Global-LoveDA_native_v1"},
    # 2448² scenes: full-image inference of the Transformers does not fit in
    # memory, so they are evaluated with overlapping 1024² windows.
    "deepglobe": {"prepared": "DeepGlobe_native_v1", "eval_window": (1024, 768)},
    # Literature reports Potsdam without clutter; it stays a class here and the
    # evaluator adds the five-class mean.
    "isprs_potsdam": {"prepared": "ISPRS_Potsdam_native_v1", "val_batch_size": 4,
                      "report_exclude": ("clutter",)},
}

DTYPE_SCALE = "auto"
PAD_DIVISOR = 32


# ----------------------------------------------------------- shared recipe
# A model leaf overrides the optimiser and loss keys where its recipe differs.
IN_CHANNELS = 3
CROP_SIZE = 512
SCALE_RANGE = (0.75, 1.5)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

IMAGE_PAD_MODE = "mean"
HFLIP_PROB = 0.5
VFLIP_PROB = 0.5
ROT90_PROB = 0.75
AUG_LEGACY_ROT90 = False
BRIGHTNESS = 0.0
CONTRAST = 0.0
SATURATION = 0.0
BLUR_PROB = 0.0
BLUR_SIGMA = (0.1, 1.5)
CUTOUT_PROB = 0.0
CUTOUT_FRAC = (0.1, 0.25)
CUTOUT_LABEL_MODE = "ignore"
CUTOUT_N = 1

BATCH_SIZE = 8
NUM_WORKERS = 8
LR = 5e-4
WEIGHT_DECAY = 1e-2
ITERS_PER_EPOCH = 1000
EARLY_STOP_PATIENCE = None
EARLY_STOP_MIN_DELTA = 0.0
WARMUP_ITERS = 200
POLY_POWER = 0.9
CLIP_GRAD = 1.0
AMP = True
EVAL_AMP = False
SEED = 42
# Not bitwise reproducible: cuDNN autotuning is about 4x faster on the 4090
# (DDRNet 34 vs 130 ms/iter).  Seeds and the sample order stay fixed.
DETERMINISTIC = False
CUDNN_BENCHMARK = True
DETERMINISTIC_DATA_LOADER = True

LOSS_NAME = "seg"
DEEP_SUP_WEIGHTS = (1.0,)
DICE_WEIGHT = 0.0
BACKBONE_LR_MULT = 1.0
NO_DECAY_BN_BIAS = True

INIT_CHECKPOINT = None
RESUME = False
ALLOW_UNVERIFIED_RESUME = False
ALLOW_CONFIG_MISMATCH_ON_RESUME = False
LOG_INTERVAL = 20


# ---------------------------------------------------------------- data
# Per dataset: prepared copy, split identity, native classes, train and
# selection lists (with their SHA256), test lists, evaluation settings.
UNIFIED_DATASETS = {d: _up.dataset_entry(DATASETS_ROOT, d, NATIVE_DATASETS)
                    for d in UNIFIED_DATASET_IDS}
# Global label of dataset d's class c = LABEL_OFFSETS[d] + c; its row of
# SET_MEMBERSHIP marks the fine classes that make up that native class.
LABEL_OFFSETS, SET_MEMBERSHIP = _up.global_membership(_TAXONOMY, UNIFIED_DATASET_IDS)
# What these datasets can teach about the fine classes; a report, not in the fingerprint.
UNIFIED_IDENTIFIABILITY = _ut.identifiability(_TAXONOMY, UNIFIED_DATASET_IDS)
DATASET_WEIGHTS = {d: 1.0 for d in UNIFIED_DATASET_IDS}

NUM_CLASSES = len(FINE_CLASS_NAMES)          # model outputs: the fine classes
CLASS_NAMES = FINE_CLASS_NAMES
IGNORE_INDEX = _nd.IGNORE_INDEX
STRICT_LABEL_SCHEMA = True                   # native labels are checked before the offset
SELECTION_METRIC = "mean over datasets of the native mIoU on each selection split"


# ----------------------------------------------------------- schedule
EPOCHS = 40 * len(UNIFIED_DATASET_IDS)
SCHEDULE_EPOCHS = EPOCHS
# Every 20k iterations; one validation reads every selection split (FLAIR:
# 38k tiles), so it runs 10 times in the five-dataset schedule.
EVAL_INTERVAL = 20
VAL_BATCH_SIZE = 1                           # per-dataset values: UNIFIED_DATASETS
EVAL_WINDOW = EVAL_STRIDE = None             # per-dataset values: UNIFIED_DATASETS
