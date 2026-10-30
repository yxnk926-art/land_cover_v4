# -*- coding: utf-8 -*-
"""Regression tests for padding, augmentation RNG isolation and loader seeding."""
import importlib
import os
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from common.dataset import OpenEarthMapDataset, resolve_aug_kwargs


def _write_tiny_oem(base, count=1, shape=(12, 14)):
    root = os.path.join(base, "oem")
    image_dir = os.path.join(root, "city", "images")
    label_dir = os.path.join(root, "city", "labels")
    os.makedirs(image_dir)
    os.makedirs(label_dir)
    names = []
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    for index in range(count):
        name = f"city_{index}.tif"
        names.append(name)
        image = np.stack(
            ((xx * 11 + index) % 256, (yy * 17 + index) % 256,
             (xx + yy * 3 + index) % 256),
            axis=-1,
        ).astype(np.uint8)
        label = ((xx + yy * w + index) % 8 + 1).astype(np.uint8)
        Image.fromarray(image, mode="RGB").save(os.path.join(image_dir, name))
        Image.fromarray(label).save(os.path.join(label_dir, name))
    split = os.path.join(root, "split.txt")
    with open(split, "w", encoding="utf-8") as handle:
        handle.write("\n".join(names) + "\n")
    return root, split, names


class DatasetAugmentationTests(unittest.TestCase):
    def _dataset(self, root, split, **kwargs):
        defaults = dict(
            training=True,
            crop_size=8,
            scale_range=(1.0, 1.0),
            mean=(0.2, 0.4, 0.6),
            std=(0.1, 0.2, 0.3),
            ignore_index=255,
            hflip_prob=0.5,
            vflip_prob=0.5,
            rot90_prob=0.75,
        )
        defaults.update(kwargs)
        return OpenEarthMapDataset(root, split, **defaults)

    def test_mean_padding_is_centered_and_normalizes_to_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root, split, _ = _write_tiny_oem(temp)
            dataset = self._dataset(root, split)
            image = np.full((2, 2, 3), 17, dtype=np.uint8)
            label = np.full((2, 2), 3, dtype=np.uint8)

            padded_image, padded_label = dataset._pad_to(image, label, 5, 6)

            self.assertEqual(padded_image.shape, (5, 6, 3))
            np.testing.assert_array_equal(padded_label[1:3, 2:4], label)
            self.assertTrue(np.all(padded_label[padded_label != 3] == 255))
            normalized, _ = dataset._normalize_to_tensor(padded_image, padded_label)
            pad_mask = torch.from_numpy(padded_label == 255)
            padded_values = normalized.permute(1, 2, 0)[pad_mask]
            self.assertTrue(torch.allclose(padded_values, torch.zeros_like(padded_values), atol=1e-6))

    def test_strict_label_schema_accepts_only_six_logits_and_external_ignore(self):
        with tempfile.TemporaryDirectory() as temp:
            root, split, names = _write_tiny_oem(temp, shape=(2, 4))
            label_path = os.path.join(root, "city", "labels", names[0])
            valid = np.array([[0, 1, 2, 3], [4, 5, 255, 0]], dtype=np.uint8)
            Image.fromarray(valid).save(label_path)
            dataset = self._dataset(
                root, split, training=False, pad_divisor=1,
                num_classes=6, strict_label_schema=True,
            )
            _, label, _ = dataset[0]
            self.assertEqual(set(torch.unique(label).tolist()), {0, 1, 2, 3, 4, 5, 255})

            for bad in (6, 254):
                Image.fromarray(np.full((2, 4), bad, dtype=np.uint8)).save(label_path)
                with self.subTest(bad=bad), self.assertRaisesRegex(
                        ValueError, "taxonomy 外 ID"):
                    self._dataset(
                        root, split, training=False, pad_divisor=1,
                        num_classes=6, strict_label_schema=True,
                    )

            Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8), mode="RGB").save(label_path)
            with self.assertRaisesRegex(ValueError, "二维单通道"):
                self._dataset(
                    root, split, training=False, pad_divisor=1,
                    num_classes=6, strict_label_schema=True,
                )

    def test_reflect_padding_falls_back_safely_for_single_pixel_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            root, split, _ = _write_tiny_oem(temp)
            dataset = self._dataset(root, split, image_pad_mode="reflect")
            image = np.array([[[10, 20, 30]]], dtype=np.uint8)
            label = np.array([[7]], dtype=np.uint8)

            padded_image, padded_label = dataset._pad_to(image, label, 3, 3)

            np.testing.assert_array_equal(padded_image, np.tile(image, (3, 3, 1)))
            self.assertEqual(padded_label[1, 1], 7)
            self.assertEqual(int(np.count_nonzero(padded_label == 255)), 8)

    def test_cutout_uses_mean_fill_and_explicit_label_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root, split, _ = _write_tiny_oem(temp)
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            label = np.full((4, 4), 5, dtype=np.uint8)
            ignored = self._dataset(
                root, split, cutout_prob=1.0, cutout_frac=(0.5, 0.5),
                cutout_label_mode="ignore",
            )
            kept = self._dataset(
                root, split, cutout_prob=1.0, cutout_frac=(0.5, 0.5),
                cutout_label_mode="keep",
            )

            cut_image, cut_label = ignored._cutout(image, label, random.Random(7))
            changed = cut_label == 255
            self.assertEqual(int(changed.sum()), 4)
            np.testing.assert_allclose(
                cut_image[changed],
                np.tile(ignored.mean * 255.0, (int(changed.sum()), 1)),
            )
            _, kept_label = kept._cutout(image, label, random.Random(7))
            np.testing.assert_array_equal(kept_label, label)

            with self.assertRaisesRegex(ValueError, "high <= 1"):
                self._dataset(root, split, cutout_frac=(0.2, 1.2))

    def test_photo_switch_does_not_change_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root, split, _ = _write_tiny_oem(temp)
            plain = self._dataset(root, split, brightness=0.0, contrast=0.0)
            photo = self._dataset(root, split, brightness=0.4, contrast=0.3)

            random.seed(20260930)
            _, plain_label, _ = plain[0]
            random.seed(20260930)
            _, photo_label, _ = photo[0]

            self.assertTrue(torch.equal(plain_label, photo_label))

    def test_rot90_default_is_continuous_semantic_probability(self):
        defaults = resolve_aug_kwargs(SimpleNamespace())
        self.assertEqual(defaults["rot90_prob"], 0.75)
        self.assertFalse(defaults["legacy_rot90"])

        class StubRng:
            def __init__(self, draw):
                self.draw = draw

            def random(self):
                return self.draw

            def randint(self, low, high):
                self.last_range = (low, high)
                return 2

        with tempfile.TemporaryDirectory() as temp:
            root, split, _ = _write_tiny_oem(temp)
            dataset = self._dataset(root, split, rot90_prob=0.25)
            hit = StubRng(0.24)
            miss = StubRng(0.25)
            self.assertEqual(dataset._sample_rotation(hit), 2)
            self.assertEqual(hit.last_range, (1, 3))
            self.assertEqual(dataset._sample_rotation(miss), 0)

    def test_deterministic_mode_disables_cudnn_benchmark(self):
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        old_benchmark = torch.backends.cudnn.benchmark
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        try:
            train = importlib.import_module("common.train")
            state = train.configure_reproducibility(SimpleNamespace(
                DETERMINISTIC=True,
                CUDNN_BENCHMARK=True,
                DETERMINISTIC_WARN_ONLY=True,
            ))
            self.assertTrue(state["deterministic"])
            self.assertFalse(state["cudnn_benchmark"])
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
        finally:
            torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
            torch.backends.cudnn.benchmark = old_benchmark
            torch.backends.cudnn.deterministic = old_cudnn_deterministic


if __name__ == "__main__":
    unittest.main()
