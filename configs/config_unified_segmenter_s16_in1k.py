# -*- coding: utf-8 -*-
"""Segmenter-S/16 (mask decoder) with ImageNet-1k DeiT-S weights.

The paper fine-tunes ImageNet-21k ViT-S with SGD (lr 1e-3, no weight decay).
Here the weights are ImageNet-1k like every other leaf, and the optimiser is
that of SegFormer-B2 and Swin-T (AdamW, lr 6e-5, weight decay 0.01, poly 1.0),
so the three Transformers differ in architecture only.  This deviates from the
paper and is provisional until a pilot confirms it.  Stochastic depth 0.1 on the
encoder and dropout 0.1 in the decoder are the official values.  The position
embeddings, class token and class embeddings are excluded from weight decay
through NO_DECAY_BN_BIAS.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "segmenter"
SEGMENTER_VARIANT = "s16"
SEGMENTER_DECODER_LAYERS = 2
DROP_PATH_RATE = 0.1
MODEL_DROPOUT = 0.1
PRETRAINED = "deit_small_patch16_224.fb_in1k"
LR = 6e-5
POLY_POWER = 1.0

MODEL_SLUG = "segmenter_s16_in1k"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
