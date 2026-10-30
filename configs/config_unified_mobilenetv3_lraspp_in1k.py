# -*- coding: utf-8 -*-
"""MobileNetV3-LRASPP with timm's ImageNet-1k MobileNetV3-Large.

Two architecture details follow the weights rather than the paper: the
full-width tail (960 channels; the paper's reduced tail has no ImageNet
release, torchvision's LR-ASPP also uses the full tail) and BatchNorm eps
1e-5, the value the timm weights were trained with.  Learning rate and weight
decay are the shared defaults.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "mobilenetv3_lraspp"
MOBILENET_WIDTH_MULT = 1.0
MOBILENET_REDUCED_TAIL = False
MOBILENET_BN_EPS = 1e-5
OUTPUT_STRIDE = 16
LRASPP_CHANNELS = 128
LRASPP_CONTEXT_POOL = "adaptive"
MODEL_DROPOUT = 0.0
PRETRAINED = "mobilenetv3_large_100.ra_in1k"

MODEL_SLUG = "mobilenetv3_lraspp_in1k"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
