# -*- coding: utf-8 -*-
"""PIDNet-S with the authors' ImageNet weights.

The I and P branches start from ImageNet, the D branch, PAPPM, fusion and heads
from scratch (as in the official code).  Optimisation stays equal to the
DDRNet-23-slim leaf.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "pidnet"
PIDNET_VARIANT = "s"
PIDNET_AUXILIARY = True
LOSS_NAME = "pidnet"
DEEP_SUP_WEIGHTS = (1.0, 0.4)
BOUNDARY_WEIGHT = 20.0
BOUNDARY_AWARE_WEIGHT = 1.0
BOUNDARY_THRESHOLD = 0.8
BOUNDARY_DILATION = 3
PRETRAINED = "pidnet_s.imagenet"
LR = 1e-3
WEIGHT_DECAY = 5e-4

MODEL_SLUG = "pidnet_s_in1k"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
