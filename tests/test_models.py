# -*- coding: utf-8 -*-
"""Output contracts of the models: shapes, train/eval outputs, losses."""
import unittest
from types import SimpleNamespace

import torch

from common.engine import SegLoss, build_optimizer, primary_logits
from common.models import build_model
from common.models.mobilenetv3 import MobileNetV3Backbone
from common.models.resnet import ResNetBackbone
from common.models.segmenter import POS_GRID, resize_pos_embed


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


class SegmenterModelTest(unittest.TestCase):
    def test_outputs_loss_and_gradients(self):
        model = build_model(_cfg(MODEL_NAME="segmenter")).train()
        outputs = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(tuple(outputs[0].shape), (2, 6, 64, 64))
        target = torch.randint(0, 6, (2, 64, 64))
        target[:, :4] = 255
        loss = SegLoss(ignore_index=255, dice_weight=0.0, deep_sup_weights=(1.0,))(outputs, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_set_cross_entropy_accepts_the_masks(self):
        model = build_model(_cfg(MODEL_NAME="segmenter")).train()
        membership = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]]
        loss = SegLoss(ignore_index=255, dice_weight=0.0, deep_sup_weights=(1.0,),
                       set_membership=membership)(
            model(torch.randn(2, 3, 64, 64)), torch.randint(0, 2, (2, 64, 64)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_full_resolution_logits_for_awkward_sizes(self):
        model = build_model(_cfg(MODEL_NAME="segmenter")).eval()
        for size in ((64, 96), (70, 90), (33, 129), (8, 8)):
            with torch.inference_mode():
                logits = primary_logits(model(torch.zeros(1, 3, *size)))
            self.assertEqual(tuple(logits.shape), (1, 6, *size))

    def test_size_matches_the_paper(self):
        model = build_model(_cfg(MODEL_NAME="segmenter"))
        # The paper lists about 22M for Seg-S/16; the 32x32 position grid adds 0.32M to DeiT-S.
        self.assertAlmostEqual(_count(model.backbone) / 1e6, 21.99, delta=0.01)
        self.assertEqual(model.backbone.pos_embed.shape[1], 1 + POS_GRID ** 2)
        self.assertEqual(len(model.decode_head.blocks), 2)
        self.assertEqual(model.decode_head.blocks[0].attn.num_heads, 6)

    def test_position_grid_resampling(self):
        pos_embed = torch.randn(1, 1 + 4 ** 2, 8)
        self.assertIs(resize_pos_embed(pos_embed, (4, 4)), pos_embed)
        resized = resize_pos_embed(pos_embed, (3, 6))
        self.assertEqual(tuple(resized.shape), (1, 1 + 3 * 6, 8))
        torch.testing.assert_close(resized[:, 0], pos_embed[:, 0])

    def test_mask_grid_is_row_major(self):
        """Without decoder blocks, one patch token changes exactly one mask cell."""
        head = build_model(_cfg(MODEL_NAME="segmenter")).decode_head.eval()
        head.blocks = torch.nn.ModuleList()
        height, width, row, column = 2, 3, 1, 2
        tokens = torch.randn(1, height * width, 384)
        changed = tokens.clone()
        changed[0, row * width + column] += 1.0
        with torch.inference_mode():
            difference = (head(tokens, (height, width)) - head(changed, (height, width))).abs()
        cells = difference.sum(1)[0] > 0
        self.assertEqual(cells.nonzero().tolist(), [[row, column]])

    def test_masks_stay_single_precision_under_amp(self):
        head = build_model(_cfg(MODEL_NAME="segmenter")).decode_head.eval()
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            masks = head(torch.randn(1, 16, 384), (4, 4))
        self.assertEqual(masks.dtype, torch.float32)

    def test_embeddings_skip_weight_decay(self):
        model = build_model(_cfg(MODEL_NAME="segmenter"))
        optimizer = build_optimizer(model, SimpleNamespace(
            LR=6e-5, WEIGHT_DECAY=0.01, NO_DECAY_BN_BIAS=True, BACKBONE_LR_MULT=1.0
        ))
        no_decay = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0.0
                    for p in g["params"]}
        for parameter in (model.backbone.cls_token, model.backbone.pos_embed,
                          model.decode_head.cls_emb):
            self.assertIn(id(parameter), no_decay)
        self.assertNotIn(id(model.backbone.blocks[0].attn.qkv.weight), no_decay)
        self.assertNotIn(id(model.decode_head.proj_patch), no_decay)


if __name__ == "__main__":
    unittest.main()
