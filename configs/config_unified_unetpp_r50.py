# -*- coding: utf-8 -*-
"""Single-head UNet++ with an ImageNet ResNet-50 encoder.

Shares encoder, decoder widths and head with the U-Net R50 leaf; only the
nested skip topology differs.  No deep supervision.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "unetpp_resnet"
ENCODER_NAME = "resnet50"
UNET_DECODER_CHANNELS = (256, 128, 64, 32, 16)
PRETRAINED = "resnet50.tv_in1k"
LR = 1e-4

MODEL_SLUG = "unetpp_r50"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
