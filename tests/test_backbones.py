# -*- coding: utf-8 -*-
"""Numerical equivalence of the self-contained backbones with their references.

ResNet-50 is checked against torchvision and Swin-T against timm on shared
random weights, which proves both the computation and the checkpoint key
layout without needing any downloaded file.
"""
import unittest

import torch

from common.models.resnet import ResNetBackbone
from common.models.swin import SwinTransformer

try:
    import torchvision
except ImportError:  # pragma: no cover - optional reference
    torchvision = None
try:
    import timm
except ImportError:  # pragma: no cover - optional reference
    timm = None


class ResNetBackboneTest(unittest.TestCase):
    def test_output_strides_and_channels(self):
        x = torch.zeros(1, 3, 96, 96)
        expected_last = {32: 3, 16: 6, 8: 12}
        for stride, size in expected_last.items():
            model = ResNetBackbone("resnet50", output_stride=stride).eval()
            with torch.inference_mode():
                features = model(x)
            self.assertEqual([f.shape[1] for f in features], [64, 256, 512, 1024, 2048])
            self.assertEqual(features[0].shape[-1], 48)
            self.assertEqual(features[1].shape[-1], 24)
            self.assertEqual(features[4].shape[-1], size)

    @unittest.skipIf(torchvision is None, "torchvision is not installed")
    def test_matches_torchvision_resnet50(self):
        torch.manual_seed(0)
        reference = torchvision.models.resnet50(weights=None).eval()
        model = ResNetBackbone("resnet50").eval()
        state = {k: v for k, v in reference.state_dict().items() if not k.startswith("fc.")}
        model.load_state_dict(state, strict=True)
        x = torch.randn(2, 3, 80, 112)
        with torch.inference_mode():
            stem = reference.relu(reference.bn1(reference.conv1(x)))
            c1 = reference.layer1(reference.maxpool(stem))
            c2 = reference.layer2(c1)
            c3 = reference.layer3(c2)
            c4 = reference.layer4(c3)
            ours = model(x)
        for expected, actual in zip((stem, c1, c2, c3, c4), ours):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    @unittest.skipIf(torchvision is None, "torchvision is not installed")
    def test_dilation_matches_torchvision(self):
        torch.manual_seed(0)
        reference = torchvision.models.resnet50(
            weights=None, replace_stride_with_dilation=[False, False, True]
        ).eval()
        model = ResNetBackbone("resnet50", output_stride=16).eval()
        model.load_state_dict(
            {k: v for k, v in reference.state_dict().items() if not k.startswith("fc.")}
        )
        x = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            h = reference.maxpool(reference.relu(reference.bn1(reference.conv1(x))))
            h = reference.layer4(reference.layer3(reference.layer2(reference.layer1(h))))
            torch.testing.assert_close(model(x)[4], h, rtol=1e-4, atol=1e-4)


class SwinBackboneTest(unittest.TestCase):
    def test_any_input_size_gives_stride_4_to_32_maps(self):
        model = SwinTransformer("tiny", drop_path_rate=0.0).eval()
        with torch.inference_mode():
            features = model(torch.zeros(1, 3, 70, 90))
        self.assertEqual([f.shape[1] for f in features], [96, 192, 384, 768])
        self.assertEqual([tuple(f.shape[-2:]) for f in features],
                         [(18, 23), (9, 12), (5, 6), (3, 3)])

    @unittest.skipIf(timm is None, "timm is not installed")
    def test_matches_timm_swin_tiny(self):
        torch.manual_seed(0)
        reference = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, img_size=448
        ).eval()
        model = SwinTransformer("tiny", drop_path_rate=0.0).eval()
        state = {k: v for k, v in reference.state_dict().items()
                 if not k.startswith(("head.", "norm."))}
        result = model.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(all(k.startswith("out_norms.") for k in result.missing_keys))

        x = torch.randn(1, 3, 448, 448)
        with torch.inference_mode():
            h = reference.patch_embed(x)
            expected = []
            for stage in reference.layers:
                h = stage(h)
                expected.append(h.permute(0, 3, 1, 2))
            actual = model.forward_stages(x, normalize=False)
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e, rtol=1e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
