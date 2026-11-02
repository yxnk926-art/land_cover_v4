# -*- coding: utf-8 -*-
"""PIDNet boundary target across ignore bands, and per-dataset boundary-aware CE."""
import random
import types
import unittest

import torch

from common.engine import (PIDNetLoss, SetCrossEntropyLoss, build_criterion,
                           make_boundary_target)

IGN = 255


def legacy_target(target, dilation=3):
    """The behaviour before the bridge existed, written out pixel by pixel."""
    height, width = target.shape[-2:]
    edge = torch.zeros(height, width, dtype=torch.bool)
    for y in range(height):
        for x in range(width):
            for ny, nx in ((y, x + 1), (y + 1, x)):
                if ny >= height or nx >= width:
                    continue
                first, second = int(target[0, y, x]), int(target[0, ny, nx])
                if first != IGN and second != IGN and first != second:
                    edge[y, x] = edge[ny, nx] = True
    return edge, dilate(edge, target, dilation)


def dilate(edge, target, dilation):
    height, width = edge.shape
    radius = dilation // 2
    out = torch.zeros_like(edge)
    for y in range(height):
        for x in range(width):
            if target[0, y, x] == IGN:
                continue
            out[y, x] = edge[max(y - radius, 0):y + radius + 1,
                             max(x - radius, 0):x + radius + 1].any()
    return out


def brute_bridge(target, radius, dilation=3):
    """Ordinary edges plus valid pixels touching ignore with another class in reach."""
    height, width = target.shape[-2:]
    edge, _ = legacy_target(target, dilation)
    edge = edge.clone()
    for y in range(height):
        for x in range(width):
            label = int(target[0, y, x])
            if label == IGN:
                continue
            near = target[0, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
            if not (near == IGN).any():
                continue
            reach = target[0, max(y - radius, 0):y + radius + 1,
                           max(x - radius, 0):x + radius + 1]
            if ((reach != IGN) & (reach != label)).any():
                edge[y, x] = True
    return dilate(edge, target, dilation)


def banded_map(width=20, band=(8, 12)):
    """Class 1 and class 2 side by side with an ignore band between them."""
    target = torch.full((1, 6, width), 1)
    target[0, :, band[1]:] = 2
    target[0, :, band[0]:band[1]] = IGN
    return target


class BoundaryTargetTest(unittest.TestCase):
    def test_default_keeps_the_old_target(self):
        rng = random.Random(0)
        for _ in range(20):
            target = torch.tensor([[[rng.choice((0, 1, 2, IGN)) for _ in range(12)]
                                    for _ in range(10)]])
            _, expected = legacy_target(target)
            boundary, valid = make_boundary_target(target, IGN, 3)
            self.assertTrue(torch.equal(boundary[0, 0] > 0.5, expected))
            self.assertTrue(torch.equal(valid[0, 0], target[0] != IGN))

    def test_band_alone_has_no_positive_without_the_bridge(self):
        boundary, _ = make_boundary_target(banded_map(), IGN, 3)
        self.assertEqual(boundary.sum().item(), 0)

    def test_bridge_marks_both_rims_and_nothing_inside_the_band(self):
        boundary, valid = make_boundary_target(banded_map(), IGN, 3, bridge_radius=6)
        columns = sorted(set(boundary[0, 0].nonzero()[:, 1].tolist()))
        self.assertEqual(columns, [6, 7, 12, 13])
        self.assertEqual((boundary * (~valid).float()).sum().item(), 0)

    def test_bridge_needs_another_class_within_the_radius(self):
        boundary, _ = make_boundary_target(banded_map(), IGN, 3, bridge_radius=4)
        self.assertEqual(boundary.sum().item(), 0)

    def test_padding_and_same_class_are_not_boundaries(self):
        padded = torch.full((1, 6, 12), 1)
        padded[0, :, 8:] = IGN
        same = banded_map()
        same[same == 2] = 1
        for target in (padded, same):
            boundary, _ = make_boundary_target(target, IGN, 3, bridge_radius=6)
            self.assertEqual(boundary.sum().item(), 0)

    def test_matches_a_pixelwise_implementation(self):
        rng = random.Random(1)
        for radius in (1, 3, 5):
            for _ in range(10):
                target = torch.tensor([[[rng.choice((1, 2, 3, IGN, IGN))
                                         for _ in range(14)] for _ in range(11)]])
                boundary, _ = make_boundary_target(target, IGN, 3, bridge_radius=radius)
                self.assertTrue(torch.equal(boundary[0, 0] > 0.5,
                                            brute_bridge(target, radius)))

    def test_label_offset_does_not_change_the_target(self):
        target = banded_map()
        shifted = target.clone()
        shifted[shifted != IGN] += 36
        first, _ = make_boundary_target(target, IGN, 3, bridge_radius=6)
        second, _ = make_boundary_target(shifted, IGN, 3, bridge_radius=6)
        self.assertTrue(torch.equal(first, second))

    def test_rejects_a_negative_radius(self):
        with self.assertRaises(ValueError):
            make_boundary_target(banded_map(), IGN, 3, bridge_radius=-1)


class PerDatasetBoundaryAwareTest(unittest.TestCase):
    def setUp(self):
        # Labels 0-1 are dataset A, labels 2-3 dataset B; identity membership.
        self.membership = torch.eye(4).int().tolist()

    def outputs(self, target, active):
        torch.manual_seed(0)
        main = torch.randn(1, 4, *target.shape[-2:], requires_grad=True)
        boundary = torch.full((1, 1, *target.shape[-2:]), -20.0)
        boundary[0, 0][active] = 20.0
        return main, boundary

    def test_pixel_losses_average_to_the_forward_value(self):
        loss = SetCrossEntropyLoss(self.membership, ignore_index=IGN)
        logits = torch.randn(2, 4, 5, 5)
        target = torch.randint(0, 4, (2, 5, 5))
        target[0, 0, :3] = IGN
        pixels = loss.pixel_losses(logits, target)
        self.assertEqual(pixels.numel(), int((target != IGN).sum()))
        self.assertTrue(torch.equal(pixels.mean(), loss(logits, target)))

    def test_each_dataset_weighs_the_same_whatever_its_pixel_count(self):
        target = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 2, IGN]]])
        active = target[0] != IGN
        main, boundary = self.outputs(target, active)
        balanced = PIDNetLoss(ignore_index=IGN, set_membership=self.membership,
                              boundary_aware_label_starts=(0, 2))
        value = balanced._boundary_aware_ce(main, boundary, target)
        ce = SetCrossEntropyLoss(self.membership, ignore_index=IGN)
        only_a = target.masked_fill(target != 0, IGN)
        only_b = target.masked_fill(target != 2, IGN)
        expected = (ce(main, only_a) + ce(main, only_b)) / 2
        self.assertAlmostEqual(value.item(), expected.item(), places=5)
        pooled = PIDNetLoss(ignore_index=IGN, set_membership=self.membership)
        self.assertNotAlmostEqual(value.item(),
                                  pooled._boundary_aware_ce(main, boundary, target).item(), places=3)
        value.backward()
        self.assertGreater(main.grad.abs().sum().item(), 0)

    def test_a_dataset_absent_from_the_batch_is_skipped(self):
        target = torch.tensor([[[0, 1, 0, 1]]])
        main, boundary = self.outputs(target, target[0] != IGN)
        balanced = PIDNetLoss(ignore_index=IGN, set_membership=self.membership,
                              boundary_aware_label_starts=(0, 2))
        pooled = PIDNetLoss(ignore_index=IGN, set_membership=self.membership)
        self.assertAlmostEqual(balanced._boundary_aware_ce(main, boundary, target).item(),
                               pooled._boundary_aware_ce(main, boundary, target).item(), places=5)

    def test_no_selected_pixel_gives_a_differentiable_zero(self):
        target = torch.tensor([[[0, 2, 0, 2]]])
        main, boundary = self.outputs(target, torch.zeros_like(target[0]).bool())
        balanced = PIDNetLoss(ignore_index=IGN, set_membership=self.membership,
                              boundary_aware_label_starts=(0, 2))
        value = balanced._boundary_aware_ce(main, boundary, target)
        self.assertEqual(value.item(), 0)
        value.backward()

    def test_full_loss_sees_potsdam_style_boundaries(self):
        target = banded_map()
        outputs = {"main": torch.randn(1, 4, *target.shape[-2:]),
                   "aux": [torch.randn(1, 4, *target.shape[-2:])],
                   "boundary": torch.zeros(1, 1, *target.shape[-2:])}
        plain = PIDNetLoss(ignore_index=IGN, set_membership=self.membership)
        bridged = PIDNetLoss(ignore_index=IGN, set_membership=self.membership,
                             boundary_bridge_radius=6)
        plain(outputs, target)
        bridged(outputs, target)
        self.assertNotEqual(plain.last_components["boundary"].item(),
                            bridged.last_components["boundary"].item())


class CriterionConfigTest(unittest.TestCase):
    def config(self, **keys):
        base = dict(IGNORE_INDEX=IGN, DICE_WEIGHT=0.0, LOSS_NAME="pidnet",
                    DEEP_SUP_WEIGHTS=(1.0, 0.4), PIDNET_AUXILIARY=True,
                    SET_MEMBERSHIP=torch.eye(4).int().tolist())
        base.update(keys)
        return types.SimpleNamespace(**base)

    def test_new_behaviour_is_off_unless_asked_for(self):
        criterion = build_criterion(self.config())
        self.assertEqual(criterion.boundary_bridge_radius, 0)
        self.assertIsNone(criterion.boundary_aware_label_starts)

    def test_keys_reach_the_loss(self):
        criterion = build_criterion(self.config(
            BOUNDARY_BRIDGE_RADIUS=12, BOUNDARY_AWARE_PER_DATASET=True,
            LABEL_OFFSETS={"a": 0, "b": 2}))
        self.assertEqual(criterion.boundary_bridge_radius, 12)
        self.assertEqual(criterion.boundary_aware_label_starts.tolist(), [0, 2])

    def test_per_dataset_needs_label_offsets(self):
        with self.assertRaises(ValueError):
            build_criterion(self.config(BOUNDARY_AWARE_PER_DATASET=True))


if __name__ == "__main__":
    unittest.main()
