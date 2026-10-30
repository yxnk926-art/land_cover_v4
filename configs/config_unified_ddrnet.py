# -*- coding: utf-8 -*-
"""DDRNet-23-slim with the authors' ImageNet weights.

Optimisation and auxiliary weight match the PIDNet-S leaf so that the
DDRNet/PIDNet pair differs only in PIDNet's boundary branch and fusion
modules.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "ddrnet"
DDRNET_VARIANT = "23_slim"
DDRNET_AUXILIARY = True
DEEP_SUP_WEIGHTS = (1.0, 0.4)
PRETRAINED = "ddrnet23_slim.imagenet"
LR = 1e-3
WEIGHT_DECAY = 5e-4

MODEL_SLUG = "ddrnet23_slim"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
