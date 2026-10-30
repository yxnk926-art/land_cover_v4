# -*- coding: utf-8 -*-
"""SegFormer-B0 with the ImageNet-1k MiT-B0 encoder.

Pairs with the B2 leaf for the encoder-scale study: optimisation is shared,
only the variant and the official decoder width (256 for B0) change.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "segformer"
SEGFORMER_VARIANT = "b0"
SEGFORMER_DECODER_DIM = 256
PRETRAINED = "mit_b0.nvidia_in1k"
MODEL_DROPOUT = 0.1
DROP_PATH_RATE = 0.1
LR = 6e-5
POLY_POWER = 1.0

MODEL_SLUG = "segformer_b0_in1k"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
