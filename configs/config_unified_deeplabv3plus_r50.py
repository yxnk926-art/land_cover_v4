# -*- coding: utf-8 -*-
"""DeepLabV3+ with an ImageNet ResNet-50 encoder (dilated layer4, output stride 16).
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "deeplabv3plus"
DEEPLAB_BACKBONE = "resnet50"
OUTPUT_STRIDE = 16
ASPP_RATES = (6, 12, 18)
ASPP_CHANNELS = 256
LOW_LEVEL_CHANNELS = 48
DECODER_CHANNELS = 256
MODEL_DROPOUT = 0.1
PRETRAINED = "resnet50.tv_in1k"
LR = 1e-4

MODEL_SLUG = "deeplabv3plus_r50"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
