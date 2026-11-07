# -*- coding: utf-8 -*-
"""YOLOv3 (Darknet-53) with the authors' ImageNet weights.

YOLOv3 is a detector.  As a segmenter (``common/models/yolov3.py``) it keeps
Darknet-53 and the neck of ``yolov3.cfg``; the stride-8 output layer predicts
class scores per grid cell instead of box, objectness and class terms per
anchor, the two coarser output branches are dropped, and the logits are
upsampled bilinearly.  The backbone starts from ``darknet53.conv.74``, the
file the authors distribute for training YOLOv3; neck and head start from
darknet's initialisation.

The optimiser is that of the ResNet-50 leaves (U-Net, UNet++, DeepLabV3+):
AdamW, lr 1e-4, weight decay 0.01, poly 0.9.  The paper's detection recipe
(SGD with burn-in and step decay) is not carried over.  Provisional until a
pilot confirms it.
"""
import os

from .config_unified_v4 import *


MODEL_NAME = "yolov3"
PRETRAINED = "darknet53.imagenet"
LR = 1e-4

MODEL_SLUG = "yolov3_in1k"
EXP_NAME = f"{EXPERIMENT_FAMILY}_{RUN_SCOPE}_{MODEL_SLUG}"
CKPT_DIR = os.path.join(CHECKPOINT_ROOT, MODEL_SLUG)
OUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_SLUG)
