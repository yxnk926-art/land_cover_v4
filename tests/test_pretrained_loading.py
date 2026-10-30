# -*- coding: utf-8 -*-
"""Pinned, strict ImageNet initialisation (common/pretrained.py)."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from common import pretrained
from common.models import build_model
from common.pretrained import PretrainedSpec, apply_pretrained, resolve_pretrained


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cfg(**kwargs):
    return SimpleNamespace(NUM_CLASSES=6, IN_CHANNELS=3, **kwargs)


class RegistryTest(unittest.TestCase):
    def test_entries_are_pinned(self):
        self.assertEqual(
            set(pretrained.REGISTRY),
            {"resnet50.tv_in1k", "swin_tiny_patch4_window7_224.ms_in1k",
             "ddrnet23_slim.imagenet", "mit_b2.nvidia_in1k", "mit_b0.nvidia_in1k",
             "mobilenetv3_large_100.ra_in1k", "pidnet_s.imagenet",
             "deit_small_patch16_224.fb_in1k"},
        )
        for name, spec in pretrained.REGISTRY.items():
            self.assertEqual(spec.name, name)
            self.assertRegex(spec.sha256, r"^[0-9a-f]{64}$")
            self.assertTrue(spec.url.startswith("https://"))
            self.assertTrue(spec.drop_prefixes)

    def test_default_directory_honours_environment(self):
        with mock.patch.dict(os.environ, {"LAND_COVER_PRETRAINED_DIR": "/x/weights"}):
            self.assertEqual(pretrained.default_pretrained_dir(), Path("/x/weights"))
        with mock.patch.dict(os.environ, {}, clear=True):
            root = Path(pretrained.__file__).resolve().parents[1]
            self.assertEqual(pretrained.default_pretrained_dir(), root.parent / "pretrained")


class SyntheticCheckpointTest(unittest.TestCase):
    """Exercise the loader on a fake ResNet-50 checkpoint in a temp directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        torch.manual_seed(0)
        source = build_model(_cfg(MODEL_NAME="unet_resnet")).backbone
        self.state = dict(source.state_dict())
        self.state["fc.weight"] = torch.zeros(1000, 2048)
        self.state["fc.bias"] = torch.zeros(1000)
        self._write(self.state)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, state):
        path = self.dir / "fake.pth"
        torch.save(state, path)
        spec = PretrainedSpec("fake.in1k", "resnet50", "fake.pth", "https://example.invalid",
                              _sha(path), ("fc.",))
        patcher = mock.patch.dict(pretrained.REGISTRY, {"fake.in1k": spec})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_loads_backbone_and_leaves_decoder_untouched(self):
        torch.manual_seed(1)
        model = build_model(_cfg(MODEL_NAME="unetpp_resnet"))
        decoder_before = model.classifier.weight.detach().clone()
        report = apply_pretrained(model, "fake.in1k", self.dir)
        self.assertEqual(report["dropped_tensors"], 2)
        self.assertEqual(report["new_tensors"], 0)
        torch.testing.assert_close(model.backbone.layer3[2].conv2.weight,
                                   self.state["layer3.2.conv2.weight"])
        torch.testing.assert_close(model.classifier.weight, decoder_before)

    def test_hash_mismatch_is_refused(self):
        with open(self.dir / "fake.pth", "ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            resolve_pretrained("fake.in1k", self.dir)

    def test_missing_file_and_unknown_name(self):
        with self.assertRaises(FileNotFoundError):
            resolve_pretrained("fake.in1k", self.dir / "nowhere")
        with self.assertRaisesRegex(ValueError, "未登记"):
            resolve_pretrained("no_such_weights", self.dir)

    def test_architecture_mismatch_is_refused(self):
        model = build_model(_cfg(MODEL_NAME="ddrnet"))
        with self.assertRaisesRegex(ValueError, "resnet50"):
            apply_pretrained(model, "fake.in1k", self.dir)

    def test_incomplete_or_extra_tensors_are_refused(self):
        incomplete = {k: v for k, v in self.state.items() if not k.startswith("layer4.2.")}
        self._write(incomplete)
        model = build_model(_cfg(MODEL_NAME="unet_resnet"))
        with self.assertRaisesRegex(ValueError, "未覆盖"):
            apply_pretrained(model, "fake.in1k", self.dir)
        extra = dict(self.state, **{"layer5.0.conv1.weight": torch.zeros(1)})
        self._write(extra)
        with self.assertRaisesRegex(ValueError, "多余"):
            apply_pretrained(model, "fake.in1k", self.dir)

    def test_models_without_spec_are_refused(self):
        model = torch.nn.Conv2d(3, 6, 1)
        with self.assertRaisesRegex(ValueError, "pretrained_spec"):
            apply_pretrained(model, "fake.in1k", self.dir)


class RealCheckpointTest(unittest.TestCase):
    """Run only where the registered files have been downloaded."""

    CASES = (
        ("resnet50.tv_in1k", dict(MODEL_NAME="unet_resnet")),
        ("resnet50.tv_in1k", dict(MODEL_NAME="unetpp_resnet")),
        ("resnet50.tv_in1k", dict(MODEL_NAME="deeplabv3plus", DEEPLAB_BACKBONE="resnet50")),
        ("swin_tiny_patch4_window7_224.ms_in1k", dict(MODEL_NAME="swin_upernet")),
        ("ddrnet23_slim.imagenet", dict(MODEL_NAME="ddrnet")),
        ("mit_b2.nvidia_in1k", dict(MODEL_NAME="segformer", SEGFORMER_VARIANT="b2")),
        ("mit_b0.nvidia_in1k", dict(MODEL_NAME="segformer", SEGFORMER_VARIANT="b0")),
        ("mobilenetv3_large_100.ra_in1k",
         dict(MODEL_NAME="mobilenetv3_lraspp", MOBILENET_REDUCED_TAIL=False,
              MOBILENET_BN_EPS=1e-5)),
        ("mobilenetv3_large_100.ra_in1k", dict(MODEL_NAME="deeplabv3plus")),
        ("pidnet_s.imagenet", dict(MODEL_NAME="pidnet", PIDNET_VARIANT="s")),
        ("deit_small_patch16_224.fb_in1k", dict(MODEL_NAME="segmenter")),
    )

    def test_registered_files_cover_their_backbones(self):
        directory = pretrained.default_pretrained_dir()
        ran = 0
        for name, overrides in self.CASES:
            if not (directory / pretrained.REGISTRY[name].filename).is_file():
                continue
            with self.subTest(name=name, model=overrides["MODEL_NAME"]):
                model = build_model(_cfg(**overrides))
                report = apply_pretrained(model, name)
                self.assertGreater(report["loaded_tensors"], 100)
                ran += 1
        if ran == 0:
            self.skipTest(f"no registered checkpoints under {directory}")

    def test_swin_derived_buffers_match_the_checkpoint(self):
        """The dropped relative_position_index must equal what the model recomputes."""
        spec = pretrained.REGISTRY["swin_tiny_patch4_window7_224.ms_in1k"]
        path = pretrained.default_pretrained_dir() / spec.filename
        if not path.is_file():
            self.skipTest(f"{path} not downloaded")
        state = pretrained.load_checkpoint_state(path)
        model = build_model(_cfg(MODEL_NAME="swin_upernet"))
        for name, buffer in model.backbone.named_buffers():
            if name.endswith("relative_position_index"):
                torch.testing.assert_close(buffer, state[name])


if __name__ == "__main__":
    unittest.main()
