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
from common.models.yolov3 import IMAGENET_MEAN, IMAGENET_STD


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


class YOLOv3ModelTest(unittest.TestCase):
    def test_outputs_loss_and_gradients(self):
        model = build_model(_cfg(MODEL_NAME="yolov3")).train()
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

    def test_set_cross_entropy_accepts_the_logits(self):
        model = build_model(_cfg(MODEL_NAME="yolov3")).train()
        membership = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]]
        loss = SegLoss(ignore_index=255, dice_weight=0.0, deep_sup_weights=(1.0,),
                       set_membership=membership)(
            model(torch.randn(2, 3, 64, 64)), torch.randint(0, 2, (2, 64, 64)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_full_resolution_logits_for_awkward_sizes(self):
        model = build_model(_cfg(MODEL_NAME="yolov3")).eval()
        for size in ((64, 96), (70, 90), (33, 129), (8, 8)):
            with torch.inference_mode():
                logits = primary_logits(model(torch.zeros(1, 3, *size)))
            self.assertEqual(tuple(logits.shape), (1, 6, *size))

    def test_logits_come_from_the_stride_8_grid(self):
        model = build_model(_cfg(MODEL_NAME="yolov3")).eval()
        shapes = []
        model.head.classifier.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
        with torch.inference_mode():
            model(torch.zeros(1, 3, 64, 96))
            model(torch.zeros(1, 3, 100, 100))  # padded to 128, like 1000 -> 1024 on Potsdam
        self.assertEqual(shapes, [(1, 6, 8, 12), (1, 6, 16, 16)])

    def test_padding_is_the_mean_colour_and_cropped_away(self):
        model = build_model(_cfg(MODEL_NAME="yolov3")).eval()
        x = torch.randn(1, 3, 40, 52)
        with torch.inference_mode():
            padded = primary_logits(model(torch.nn.functional.pad(x, (0, 12, 0, 24))))
            direct = primary_logits(model(x))
        self.assertTrue(torch.equal(direct, padded[..., :40, :52]))

    def test_backbone_sees_rgb_in_unit_range(self):
        """The darknet weights expect [0, 1] RGB; the pipeline feeds (rgb - mean) / std."""
        model = build_model(_cfg(MODEL_NAME="yolov3", MEAN=IMAGENET_MEAN, STD=IMAGENET_STD)).eval()
        seen = []
        model.backbone.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        rgb = torch.rand(1, 3, 40, 64)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        with torch.inference_mode():
            model((rgb - mean) / std)
        self.assertEqual(tuple(seen[0].shape), (1, 3, 64, 64))
        torch.testing.assert_close(seen[0][..., :40, :], rgb, rtol=0, atol=1e-6)
        torch.testing.assert_close(seen[0][..., 40:, :], mean.expand(1, 3, 24, 64))
        with self.assertRaisesRegex(ValueError, "MEAN/STD"):
            build_model(_cfg(MODEL_NAME="yolov3", IN_CHANNELS=4))

    def test_layers_and_size_match_yolov3_cfg(self):
        model = build_model(_cfg(MODEL_NAME="yolov3"))
        convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        # yolov3.cfg has 75 convolutions; the box branches 80-81 and 92-93 are absent.
        self.assertEqual(len(convs), 71)
        self.assertEqual(_count(model.backbone), 40_584_928)  # = timm darknet53 without classifier
        # yolov3.cfg: 61,949,149 parameters, of which layers 80-81, 92-93 and the
        # 255-channel layer 105 hold 6,359,037; layer 105 here has 257 per class.
        self.assertEqual(_count(model), 61_949_149 - 6_359_037 + 257 * 6)
        for module in model.modules():
            if isinstance(module, torch.nn.LeakyReLU):
                self.assertEqual(module.negative_slope, 0.1)
            elif isinstance(module, torch.nn.BatchNorm2d):
                self.assertEqual((module.eps, module.momentum), (1e-5, 0.01))
        spec = model.pretrained_spec()
        self.assertEqual((spec["arch"], spec["allowed_missing"]), ("darknet53", ()))
        self.assertIs(spec["module"], model.backbone)

    def test_batchnorm_and_biases_skip_weight_decay(self):
        model = build_model(_cfg(MODEL_NAME="yolov3"))
        optimizer = build_optimizer(model, SimpleNamespace(
            LR=1e-4, WEIGHT_DECAY=0.01, NO_DECAY_BN_BIAS=True, BACKBONE_LR_MULT=1.0
        ))
        no_decay = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0.0
                    for p in g["params"]}
        for parameter in (model.backbone.stem.conv1.bn.weight, model.neck.block3[4].bn.bias,
                          model.head.classifier.bias):
            self.assertIn(id(parameter), no_decay)
        for parameter in (model.backbone.stem.conv1.conv.weight, model.head.classifier.weight):
            self.assertNotIn(id(parameter), no_decay)


if __name__ == "__main__":
    unittest.main()
