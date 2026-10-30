# -*- coding: utf-8 -*-
"""Regression tests for LR-ASPP pooling and configuration forwarding."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from common.models import build_model
from common.models.mobilenetv3_lraspp import MobileNetV3LRASPP


class _ShapeOnlyBackbone(nn.Module):
    """Return inexpensive feature maps while preserving LR-ASPP dimensions."""

    def __init__(self, source, output_stride):
        super().__init__()
        self.high_channels = source.high_channels
        self.low8_expanded_channels = source.low8_expanded_channels
        self.output_stride = output_stride

    def forward(self, x):
        batch, _, height, width = x.shape
        high_size = (
            max(1, height // self.output_stride),
            max(1, width // self.output_stride),
        )
        low8_size = (max(1, height // 8), max(1, width // 8))
        kwargs = {"device": x.device, "dtype": x.dtype}
        return {
            "high": torch.zeros(
                batch, self.high_channels, *high_size, **kwargs
            ),
            "low8_expanded": torch.zeros(
                batch, self.low8_expanded_channels, *low8_size, **kwargs
            ),
        }


class MobileNetV3LRASPPTests(unittest.TestCase):
    def test_factory_forwards_output_stride_and_context_pool(self):
        cfg = SimpleNamespace(
            MODEL_NAME="mobilenetv3_lraspp",
            NUM_CLASSES=5,
            IN_CHANNELS=4,
            MOBILENET_WIDTH_MULT=0.25,
            OUTPUT_STRIDE=8,
            LRASPP_CHANNELS=24,
            MODEL_DROPOUT=0.2,
            LRASPP_CONTEXT_POOL="fixed",
        )

        model = build_model(cfg)

        self.assertEqual(model.output_stride, 8)
        self.assertEqual(model.backbone.output_stride, 8)
        self.assertEqual(model.context_pool, "fixed")
        self.assertEqual(model.high_classifier.out_channels, 5)
        self.assertEqual(model.high_proj[0].in_channels, model.backbone.high_channels)

    def test_adaptive_forward_preserves_768_and_1024_input_shapes(self):
        model = MobileNetV3LRASPP(
            num_classes=3,
            width_mult=0.25,
            output_stride=16,
            head_channels=8,
            context_pool="adaptive",
        ).eval()
        model.backbone = _ShapeOnlyBackbone(model.backbone, output_stride=16)

        with torch.inference_mode():
            for size in (768, 1024):
                with self.subTest(size=size):
                    output = model(torch.empty(1, 3, size, size))
                    self.assertEqual(len(output), 1)
                    self.assertEqual(tuple(output[0].shape), (1, 3, size, size))

    def test_adaptive_context_includes_bottom_right_of_64_feature_map(self):
        model = MobileNetV3LRASPP(
            width_mult=0.25,
            head_channels=1,
            context_pool="adaptive",
        ).eval()
        with torch.no_grad():
            model.context_conv.weight.fill_(1.0)
            model.context_conv.bias.zero_()

        channels = model.backbone.high_channels
        baseline = torch.zeros(1, channels, 64, 64)
        bottom_right = baseline.clone()
        bottom_right[:, :, -1, -1] = 4096.0 / channels

        with torch.inference_mode():
            baseline_gate = model._context_gate(baseline)
            bottom_right_gate = model._context_gate(bottom_right)

        self.assertGreater(
            bottom_right_gate[0, 0, 0, 0].item(),
            baseline_gate[0, 0, 0, 0].item(),
        )
        self.assertTrue(
            torch.equal(
                bottom_right_gate[..., :1, :1],
                bottom_right_gate[..., -1:, -1:],
            )
        )

    def test_invalid_options_and_small_fixed_feature_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "context_pool"):
            MobileNetV3LRASPP(context_pool="auto")
        with self.assertRaisesRegex(ValueError, "output_stride"):
            MobileNetV3LRASPP(output_stride=12)

        model = MobileNetV3LRASPP(
            width_mult=0.25,
            head_channels=1,
            context_pool="fixed",
        ).eval()
        high = torch.zeros(1, model.backbone.high_channels, 48, 48)
        with self.assertRaisesRegex(RuntimeError, "min\(H, W\) >= 49"):
            model._context_gate(high)


if __name__ == "__main__":
    unittest.main()
