# -*- coding: utf-8 -*-
"""Output contracts of the models: shapes, train/eval outputs, losses."""
import unittest
from types import SimpleNamespace

import torch

from common.engine import SegLoss, build_optimizer, primary_logits
from common.models import build_model
from common.models.mobilenetv3 import MobileNetV3Backbone
from common.models.resnet import ResNetBackbone


def _cfg(**kwargs):
    base = dict(NUM_CLASSES=6, IN_CHANNELS=3)
    base.update(kwargs)
    return SimpleNamespace(**base)


def _count(module):
    return sum(p.numel() for p in module.parameters())


class ResNetDecoderModelsTest(unittest.TestCase):
    def test_unet_and_unetpp_share_everything_but_skips(self):
        unet = build_model(_cfg(MODEL_NAME="unet_resnet"))
        unetpp = build_model(_cfg(MODEL_NAME="unetpp_resnet"))
        self.assertEqual(_count(unet.backbone), _count(unetpp.backbone))
        self.assertEqual(_count(unet.final), _count(unetpp.final))
        self.assertEqual(_count(unet.classifier), _count(unetpp.classifier))
        self.assertGreater(_count(unetpp.nodes), _count(unet.blocks))
        # X^{3,1} of UNet++ is the U-Net level-3 block.
        self.assertEqual(_count(unetpp.nodes["x3_1"]), _count(unet.blocks[3]))

    def test_full_resolution_logits_for_awkward_sizes(self):
        for name in ("unet_resnet", "unetpp_resnet"):
            model = build_model(_cfg(MODEL_NAME=name)).eval()
            for size in ((64, 96), (70, 90)):
                with torch.inference_mode():
                    out = model(torch.zeros(1, 3, *size))
                self.assertIsInstance(out, list)
                self.assertEqual(tuple(out[0].shape), (1, 6, *size), name)

    def test_deeplab_backbone_switch(self):
        default = build_model(_cfg(MODEL_NAME="deeplabv3plus"))
        self.assertIsInstance(default.backbone, MobileNetV3Backbone)
        resnet = build_model(_cfg(MODEL_NAME="deeplabv3plus", DEEPLAB_BACKBONE="resnet50",
                                  OUTPUT_STRIDE=16)).eval()
        self.assertIsInstance(resnet.backbone, ResNetBackbone)
        self.assertEqual(resnet.backbone.output_stride, 16)
        self.assertEqual(resnet.aspp.branches[0][0].in_channels, 2048)
        with torch.inference_mode():
            out = resnet(torch.zeros(1, 3, 64, 80))
        self.assertEqual(tuple(out[0].shape), (1, 6, 64, 80))
        self.assertEqual(default.pretrained_spec()["arch"], "mobilenetv3_large_100")
        self.assertIs(default.pretrained_spec()["module"], default.backbone)
        with self.assertRaises(ValueError):
            build_model(_cfg(MODEL_NAME="deeplabv3plus", DEEPLAB_BACKBONE="vgg16"))


class AuxiliaryHeadModelsTest(unittest.TestCase):
    def _check_train_eval(self, cfg, size=(64, 64)):
        model = build_model(cfg)
        model.train()
        outputs = model(torch.randn(2, 3, *size))
        self.assertEqual(len(outputs), 2)
        for tensor in outputs:
            self.assertEqual(tuple(tensor.shape), (2, 6, *size))
        target = torch.randint(0, 6, (2, *size))
        target[:, :4] = 255
        loss = SegLoss(ignore_index=255, dice_weight=0.0,
                       deep_sup_weights=(1.0, 0.4))(outputs, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        with torch.inference_mode():
            outputs = model(torch.zeros(1, 3, *size))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(tuple(primary_logits(outputs).shape), (1, 6, *size))
        return model

    def test_swin_upernet_outputs(self):
        model = self._check_train_eval(_cfg(MODEL_NAME="swin_upernet", DROP_PATH_RATE=0.1))
        no_aux = build_model(_cfg(MODEL_NAME="swin_upernet", UPERNET_AUXILIARY=False))
        no_aux.train()
        self.assertEqual(len(no_aux(torch.randn(2, 3, 64, 64))), 1)
        # Swin-T backbone 27.5M + UperNet head; the 1x1 PPM branch has no BatchNorm.
        self.assertAlmostEqual(_count(model.backbone) / 1e6, 27.5, delta=0.3)
        first_branch = model.decode_head.ppm.branches[0]
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in first_branch.modules()))

    def test_ddrnet_outputs_and_size(self):
        model = self._check_train_eval(_cfg(MODEL_NAME="ddrnet"), size=(72, 88))
        # The authors report 5.7M parameters for DDRNet-23-slim (19 classes).
        self.assertAlmostEqual(_count(model) / 1e6, 5.7, delta=0.15)

    def test_swin_bias_tables_skip_weight_decay(self):
        model = build_model(_cfg(MODEL_NAME="swin_upernet"))
        optimizer = build_optimizer(model, SimpleNamespace(
            LR=6e-5, WEIGHT_DECAY=0.01, NO_DECAY_BN_BIAS=True, BACKBONE_LR_MULT=1.0
        ))
        no_decay = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0.0
                    for p in g["params"]}
        tables = [p for n, p in model.named_parameters() if "relative_position_bias_table" in n]
        self.assertEqual(len(tables), 12)
        self.assertTrue(all(id(p) in no_decay for p in tables))
        qkv = model.backbone.layers[0].blocks[0].attn.qkv.weight
        self.assertNotIn(id(qkv), no_decay)


if __name__ == "__main__":
    unittest.main()
