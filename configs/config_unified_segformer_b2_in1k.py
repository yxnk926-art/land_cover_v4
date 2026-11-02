# -*- coding: utf-8 -*-
"""SegFormer-B2 with NVIDIA's ImageNet-1k MiT-B2 encoder.

Learning rate 6e-5 and poly 1.0 follow the official fine-tuning recipe, except
that the official decode head has lr_mult=10 and this one does not; a 40k pilot
on SegFormer-B0 gained 0.004, within the noise, so the recipe stays.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "segformer"
SEGFORMER_VARIANT = "b2"
SEGFORMER_DECODER_DIM = 768
PRETRAINED = "mit_b2.nvidia_in1k"
MODEL_DROPOUT = 0.1
DROP_PATH_RATE = 0.1
LR = 6e-5
POLY_POWER = 1.0

MODEL_SLUG = "segformer_b2_in1k"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
