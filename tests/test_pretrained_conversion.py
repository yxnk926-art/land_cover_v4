# -*- coding: utf-8 -*-
"""Key conversion of the MiT, MobileNetV3, PIDNet and DeiT ImageNet releases,
and the reader of darknet's Darknet-53 weight files.

Where a reference implementation is installed (timm, HF transformers), the
converted weights must reproduce its features numerically; this proves the
key layout, the fused key/value projection and the block order rather than
only matching names and shapes.
"""
import dataclasses
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from common import pretrained
from common.models import build_model
from common.models.mobilenetv3 import MobileNetV3Backbone
from common.models.segformer import MixVisionTransformer, SEGFORMER_VARIANTS
from common.models.yolov3 import Darknet53, darknet53_layout
from common.pretrained import (DARKNET53_CLASSIFIER, apply_pretrained, convert_hf_mit,
                               convert_pidnet_official, convert_timm_deit,
                               convert_timm_mobilenetv3, read_darknet53)

try:
    import timm
except ImportError:  # pragma: no cover - optional reference
    timm = None
try:
    import transformers
except ImportError:  # pragma: no cover - optional reference
    transformers = None


def _cfg(**kwargs):
    return SimpleNamespace(NUM_CLASSES=6, IN_CHANNELS=3, **kwargs)


def _file(name):
    path = pretrained.default_pretrained_dir() / pretrained.REGISTRY[name].filename
    return path if path.is_file() else None


class MobileNetV3ConversionTest(unittest.TestCase):
    @unittest.skipIf(timm is None, "timm is not installed")
    def test_matches_timm_features(self):
        """Random timm weights -> converted -> same 1/4, 1/8 and 960-channel maps."""
        torch.manual_seed(0)
        reference = timm.create_model("mobilenetv3_large_100", pretrained=False).eval()
        for module in reference.modules():  # non-trivial BN statistics
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.5, 2.0)
        state = {k: v for k, v in reference.state_dict().items()
                 if not k.startswith(("conv_head.", "classifier."))}
        model = MobileNetV3Backbone(output_stride=32, norm_eps=1e-5).eval()
        model.load_state_dict(convert_timm_mobilenetv3(state), strict=True)

        x = torch.randn(2, 3, 96, 128)
        with torch.inference_mode():
            h = reference.bn1(reference.conv_stem(x))
            stages = []
            for stage in reference.blocks:
                h = stage(h)
                stages.append(h)
            ours = model(x)
        torch.testing.assert_close(ours["low"], stages[1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ours["low8"], stages[2], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ours["high"], stages[6], rtol=1e-4, atol=1e-4)

    def test_reduced_tail_and_width_are_refused(self):
        name = "mobilenetv3_large_100.ra_in1k"
        if _file(name) is None:
            self.skipTest("MobileNetV3 weights not downloaded")
        with self.assertRaisesRegex(ValueError, "MOBILENET_REDUCED_TAIL"):
            apply_pretrained(build_model(_cfg(MODEL_NAME="mobilenetv3_lraspp")), name)
        narrow = build_model(_cfg(MODEL_NAME="deeplabv3plus", MOBILENET_WIDTH_MULT=0.75))
        with self.assertRaisesRegex(ValueError, "mobilenetv3_large_75"):
            apply_pretrained(narrow, name)

    def test_default_eps_keeps_the_v3_models_unchanged(self):
        for model in (build_model(_cfg(MODEL_NAME="mobilenetv3_lraspp")),
                      build_model(_cfg(MODEL_NAME="deeplabv3plus"))):
            eps = {m.eps for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)}
            self.assertEqual(eps, {1e-3})
        lraspp = build_model(_cfg(MODEL_NAME="mobilenetv3_lraspp"))
        self.assertTrue(lraspp.backbone.reduced_tail)
        self.assertEqual(lraspp.backbone.high_channels, 480)


class MiTConversionTest(unittest.TestCase):
    def test_key_and_value_are_fused_in_that_order(self):
        key = torch.randn(4, 4)
        value = torch.randn(4, 4)
        state = {"segformer.encoder.block.0.0.attention.self.key.weight": key,
                 "segformer.encoder.block.0.0.attention.self.value.weight": value}
        out = convert_hf_mit(state)
        torch.testing.assert_close(out["stages.0.0.attention.kv.weight"],
                                   torch.cat((key, value)))

    def test_unknown_keys_fail(self):
        with self.assertRaisesRegex(ValueError, "未覆盖"):
            convert_hf_mit({"segformer.encoder.extra.weight": torch.zeros(1)})

    @unittest.skipIf(transformers is None, "transformers is not installed")
    def test_matches_transformers_segformer_encoder(self):
        from transformers import SegformerConfig, SegformerModel

        spec = SEGFORMER_VARIANTS["b0"]
        config = SegformerConfig(
            num_channels=3, num_encoder_blocks=4, depths=list(spec["depths"]),
            hidden_sizes=list(spec["dims"]), num_attention_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4], hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0, drop_path_rate=0.0,
        )
        torch.manual_seed(0)
        reference = SegformerModel(config).eval()
        model = MixVisionTransformer(3, spec["dims"], spec["depths"],
                                     drop_path_rate=0.0).eval()
        state = {"segformer." + k: v for k, v in reference.state_dict().items()}
        model.load_state_dict(convert_hf_mit(state), strict=True)
        # Compare the layout, not LayerNorm eps conventions: give every norm
        # the eps of its HF counterpart, matched through the converted name.
        ours = dict(model.named_modules())
        hf_norms = {name: m for name, m in reference.named_modules()
                    if isinstance(m, nn.LayerNorm)}
        self.assertEqual(len(hf_norms),
                         sum(isinstance(m, nn.LayerNorm) for m in ours.values()))
        for name, module in hf_norms.items():
            (key,) = convert_hf_mit({f"segformer.{name}.weight": module.weight})
            ours[key[:-len(".weight")]].eps = module.eps

        x = torch.randn(1, 3, 96, 128)
        with torch.inference_mode():
            expected = reference(x, output_hidden_states=True).hidden_states
            actual = model(x)
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-4)


class PIDNetConversionTest(unittest.TestCase):
    @staticmethod
    def _stem(bias):
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=bias), nn.BatchNorm2d(8),
            nn.ReLU(), nn.Conv2d(8, 8, 3, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(8), nn.ReLU(),
        )

    def test_stem_bias_fold_is_exact(self):
        torch.manual_seed(0)
        official = self._stem(bias=True)
        for module in official.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.uniform_(-1, 1)
                module.running_var.uniform_(0.5, 2.0)
            if isinstance(module, nn.Conv2d):
                module.bias.data.uniform_(-1, 1)
        state = {"conv1." + k: v for k, v in official.state_dict().items()}
        converted = convert_pidnet_official(state)
        ours = self._stem(bias=False)
        ours.load_state_dict({k[len("conv1."):]: v for k, v in converted.items()})
        x = torch.randn(2, 3, 32, 32)
        for train in (False, True):
            official.train(train)
            ours.train(train)
            torch.testing.assert_close(ours(x), official(x), rtol=1e-5, atol=1e-5)

    def test_branch_names_are_mapped(self):
        state = {"conv1.0.bias": torch.zeros(1), "conv1.1.running_mean": torch.zeros(1),
                 "conv1.3.bias": torch.zeros(1), "conv1.4.running_mean": torch.zeros(1),
                 "layer3_.0.conv1.weight": torch.zeros(1),
                 "pag3.f_x.0.weight": torch.zeros(1), "pag4.f_y.1.bias": torch.zeros(1),
                 "layer3.0.conv1.weight": torch.zeros(1)}
        self.assertEqual(set(convert_pidnet_official(state)), {
            "conv1.1.running_mean", "conv1.4.running_mean", "layer3_p.0.conv1.weight",
            "pag3.f_p.0.weight", "pag4.f_i.1.bias", "layer3.0.conv1.weight"})

    def test_real_weights_leave_only_declared_modules_new(self):
        name = "pidnet_s.imagenet"
        if _file(name) is None:
            self.skipTest("PIDNet-S weights not downloaded")
        torch.manual_seed(0)
        model = build_model(_cfg(MODEL_NAME="pidnet", PIDNET_VARIANT="s"))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        report = apply_pretrained(model, name)
        after = model.state_dict()
        changed = {k for k in after if not torch.equal(after[k], before[k])}
        prefixes = model.pretrained_spec()["allowed_missing"]
        self.assertFalse(any(k.startswith(prefixes) for k in changed))
        self.assertTrue(any(k.startswith("layer3_p.") for k in changed))
        self.assertEqual(report["dropped_tensors"], 20)


class DeiTConversionTest(unittest.TestCase):
    @unittest.skipIf(timm is None, "timm is not installed")
    def test_position_grid_matches_timm_resampling(self):
        torch.manual_seed(0)
        reference = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
        state = dict(reference.state_dict())
        converted = convert_timm_deit(state)
        expected = timm.layers.resample_abs_pos_embed(
            reference.pos_embed, new_size=[32, 32], old_size=[14, 14], num_prefix_tokens=1,
            interpolation="bilinear", antialias=False)
        torch.testing.assert_close(converted["pos_embed"], expected)
        self.assertEqual(set(converted), set(state))
        torch.testing.assert_close(converted["blocks.11.attn.qkv.weight"],
                                   state["blocks.11.attn.qkv.weight"])

    def test_unknown_keys_are_refused(self):
        with self.assertRaisesRegex(ValueError, "未覆盖"):
            convert_timm_deit({"pos_embed": torch.zeros(1, 197, 384), "dist_token": torch.zeros(1)})

    def test_real_checkpoint_fills_the_encoder(self):
        path = _file("deit_small_patch16_224.fb_in1k")
        if path is None:
            self.skipTest("DeiT-S checkpoint not downloaded")
        model = build_model(_cfg(MODEL_NAME="segmenter"))
        decoder_before = model.decode_head.cls_emb.detach().clone()
        report = apply_pretrained(model, "deit_small_patch16_224.fb_in1k")
        self.assertEqual(report["loaded_tensors"], 150)
        self.assertEqual(report["dropped_tensors"], 2)
        torch.testing.assert_close(model.decode_head.cls_emb, decoder_before)


def _write_darknet(path, state, layers, header=(0, 2, 0), seen=12345):
    """Write ``state`` in darknet's .weights layout (the inverse of the reader)."""
    major, minor, revision = header
    with open(path, "wb") as handle:
        handle.write(struct.pack("<3i", major, minor, revision))
        handle.write(struct.pack("<q" if major * 10 + minor >= 2 else "<i", seen))
        for prefix, _, _, _, batch_norm in layers:
            names = (("bn.bias", "bn.weight", "bn.running_mean", "bn.running_var", "conv.weight")
                     if batch_norm else ("bias", "weight"))
            for name in names:
                handle.write(state[f"{prefix}.{name}"].numpy().astype("<f4").tobytes())


class DarknetReaderTest(unittest.TestCase):
    """The darknet format has no names: every value is placed by layer order alone."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        torch.manual_seed(0)
        self.layers = [(*layer, True) for layer in darknet53_layout()]
        self.state = {k: torch.randn(v.shape) for k, v in Darknet53().state_dict().items()
                      if not k.endswith("num_batches_tracked")}
        self.path = Path(self.tmp.name) / "darknet53.conv.74"

    def test_round_trip_of_the_backbone(self):
        _write_darknet(self.path, self.state, self.layers)
        self.assertEqual(self.path.stat().st_size, 20 + 4 * 40_620_640)  # the released size
        loaded = read_darknet53(self.path)
        self.assertEqual(set(loaded), set(self.state))
        for key, value in self.state.items():
            self.assertTrue(torch.equal(loaded[key], value), key)

    def test_classifier_tail_is_read_and_dropped(self):
        self.state["fc.weight"] = torch.randn(1000, 1024, 1, 1)
        self.state["fc.bias"] = torch.randn(1000)
        _write_darknet(self.path, self.state, self.layers + [DARKNET53_CLASSIFIER])
        loaded = read_darknet53(self.path)
        self.assertTrue(torch.equal(loaded["fc.weight"], self.state["fc.weight"]))
        self.assertTrue(torch.equal(loaded["fc.bias"], self.state["fc.bias"]))
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        spec = dataclasses.replace(pretrained.REGISTRY["darknet53.imagenet"], sha256=sha)
        with mock.patch.dict(pretrained.REGISTRY, {"darknet53.imagenet": spec}):
            model = build_model(_cfg(MODEL_NAME="yolov3"))
            report = apply_pretrained(model, "darknet53.imagenet", self.tmp.name)
        self.assertEqual((report["loaded_tensors"], report["dropped_tensors"]), (260, 2))
        torch.testing.assert_close(model.backbone.stages[4].blocks[3].conv2.bn.running_var,
                                   self.state["stages.4.blocks.3.conv2.bn.running_var"])

    def test_old_header_with_32_bit_seen(self):
        _write_darknet(self.path, self.state, self.layers, header=(0, 1, 0))
        loaded = read_darknet53(self.path)
        self.assertTrue(torch.equal(loaded["stem.conv1.conv.weight"],
                                    self.state["stem.conv1.conv.weight"]))

    def test_files_of_another_length_are_refused(self):
        _write_darknet(self.path, self.state, self.layers)
        with open(self.path, "ab") as handle:
            handle.write(b"\0\0\0\0")
        with self.assertRaisesRegex(ValueError, "40620641 个权重值"):
            read_darknet53(self.path)

    def test_real_checkpoint_fills_the_backbone(self):
        path = _file("darknet53.imagenet")
        if path is None:
            self.skipTest("darknet53.conv.74 not downloaded")
        model = build_model(_cfg(MODEL_NAME="yolov3"))
        head_before = model.head.classifier.weight.detach().clone()
        report = apply_pretrained(model, "darknet53.imagenet")
        self.assertEqual((report["loaded_tensors"], report["dropped_tensors"]), (260, 0))
        self.assertTrue(torch.equal(model.head.classifier.weight, head_before))
        variances = [b for n, b in model.backbone.named_buffers() if n.endswith("running_var")]
        self.assertTrue(all(bool((v > 0).all()) for v in variances))


if __name__ == "__main__":
    unittest.main()
