# -*- coding: utf-8 -*-
"""Swin-T + UperNet with ImageNet-1k Swin-T weights.

Optimisation follows the mmsegmentation Swin-T UperNet recipe adapted to
AdamW/poly in this repository: lr 6e-5, weight decay 0.01, poly power 1.0,
stochastic depth 0.3 and an FCN auxiliary head on stage 3 with weight 0.4.
Warm-up stays at the shared value (mmseg uses 1500 iterations).  The relative
position bias tables are excluded from weight decay through NO_DECAY_BN_BIAS.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "swin_upernet"
SWIN_VARIANT = "tiny"
SWIN_WINDOW_SIZE = 7
DROP_PATH_RATE = 0.3
UPERNET_CHANNELS = 512
UPERNET_POOL_SCALES = (1, 2, 3, 6)
UPERNET_AUXILIARY = True
AUX_CHANNELS = 256
MODEL_DROPOUT = 0.1
DEEP_SUP_WEIGHTS = (1.0, 0.4)
PRETRAINED = "swin_tiny_patch4_window7_224.ms_in1k"
LR = 6e-5
POLY_POWER = 1.0

MODEL_SLUG = "swin_t_upernet"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
