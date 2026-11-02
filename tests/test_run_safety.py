# -*- coding: utf-8 -*-
"""Interrupted runs and what a run leaves behind: resume, atomic saves, provenance, histories."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.engine import SetCrossEntropyLoss, evaluate  # noqa: E402
from common.unified_protocol import DatasetLossMeter  # noqa: E402
from common.unified_taxonomy import DEFAULT_PATH, identifiability, load_taxonomy  # noqa: E402
with mock.patch.object(sys, "argv", ["train.py", "--config", "config_unified_ddrnet"]):
    from common import train  # noqa: E402
from tools import evaluate_unified  # noqa: E402


def leaf(root):
    return types.SimpleNamespace(OUTPUT_ROOT=str(root / "outputs"), CHECKPOINT_ROOT=str(root / "ckpt"),
                                 MODEL_SLUG="m", EXP_NAME="e", RESUME=False, INIT_CHECKPOINT=None,
                                 SEED=42)


def args(**values):
    return types.SimpleNamespace(**{"seed": None, "run_tag": "t", "resume": False, **values})


class ResumeOverrideTest(unittest.TestCase):
    def test_occupied_run_needs_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "outputs" / "baselines" / "t" / "m"
            output.mkdir(parents=True)
            (output / "train_log.txt").write_text("old\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                train.apply_run_overrides(leaf(root), args())
            module = leaf(root)
            train.apply_run_overrides(module, args(resume=True))
            self.assertTrue(module.RESUME)
            self.assertEqual(module.OUT_DIR, str(output))

    def test_fresh_run_does_not_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = leaf(Path(tmp).resolve())
            train.apply_run_overrides(module, args())
            self.assertFalse(module.RESUME)

    def test_resume_needs_a_run_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "--run-tag"):
                train.apply_run_overrides(leaf(Path(tmp).resolve()), args(run_tag=None, resume=True))

    def test_a_leaf_cannot_turn_on_resume_by_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = leaf(Path(tmp).resolve())
            module.RESUME = True
            with self.assertRaisesRegex(ValueError, "RESUME"):
                train.apply_run_overrides(module, args(resume=True))


class AtomicSaveTest(unittest.TestCase):
    def test_interrupted_save_keeps_the_old_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint_last.pth")
            train.atomic_torch_save({"epoch": 1}, path)
            with mock.patch.object(torch, "save", side_effect=OSError("disk gone")):
                with self.assertRaises(OSError):
                    train.atomic_torch_save({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)


class ProvenanceTest(unittest.TestCase):
    def test_hash_follows_the_files_outside_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "common").mkdir()
            (root / "common" / "a.py").write_text("x = 1\n", encoding="utf-8")
            (root / "common" / "a.pyc").write_text("ignored", encoding="utf-8")
            first = train.code_provenance(root)
            self.assertIsNone(first["git_commit"])
            self.assertIsNone(first["git_dirty"])
            self.assertEqual(list(first["files"]), ["common/a.py"])
            (root / "common" / "a.py").write_text("x = 2\n", encoding="utf-8")
            self.assertNotEqual(train.code_provenance(root)["core_sha256"], first["core_sha256"])

    def test_this_repository(self):
        code = train.code_provenance()
        self.assertEqual(len(code["core_sha256"]), 64)
        self.assertIn("common/train.py", code["files"])
        json.dumps(code)


class HistoryTest(unittest.TestCase):
    def test_numpy_values_round_trip(self):
        record = {"epoch": 3, "scores": {"miou": np.float64(0.5), "valid": np.array([True, False]),
                                         "matrix": np.array([[1, 2], [3, 4]])}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.jsonl")
            train.append_history(path, record)
            train.append_history(path, {"epoch": 4})
            lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[0]["scores"], {"miou": 0.5, "valid": [True, False],
                                              "matrix": [[1, 2], [3, 4]]})
        self.assertEqual(lines[1], {"epoch": 4})


class DatasetLossMeterTest(unittest.TestCase):
    MEMBERSHIP = [[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]]
    CFG = types.SimpleNamespace(LABEL_OFFSETS={"a": 0, "b": 2}, SET_MEMBERSHIP=MEMBERSHIP,
                                IGNORE_INDEX=255)

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 3, 4, 4)
        self.target = torch.tensor([[0, 1], [1, 255]]).repeat_interleave(2, 0).repeat_interleave(2, 1)
        self.target = torch.stack([self.target, self.target.where(self.target == 255, self.target + 2)])

    def test_split_matches_the_training_loss(self):
        meter = DatasetLossMeter(self.CFG, torch.device("cpu"))
        meter.update(self.logits, self.target)
        means = meter.means()
        loss = SetCrossEntropyLoss(self.MEMBERSHIP, ignore_index=255)
        for index, name in enumerate(("a", "b")):
            only = torch.full_like(self.target, 255)
            only[index] = self.target[index]
            self.assertAlmostEqual(means[name], loss(self.logits, only).item(), places=5)
        self.assertAlmostEqual((means["a"] + means["b"]) / 2,
                               loss(self.logits, self.target).item(), places=5)

    def test_means_reset_and_report_missing_datasets(self):
        meter = DatasetLossMeter(self.CFG, torch.device("cpu"))
        only_a = self.target.clone()
        only_a[1] = 255
        meter.update(self.logits, only_a)
        means = meter.means()
        self.assertIsNotNone(means["a"])
        self.assertIsNone(means["b"])
        self.assertEqual(meter.means(), {"a": None, "b": None})

    def test_low_resolution_output(self):
        meter = DatasetLossMeter(self.CFG, torch.device("cpu"))
        meter.update(self.logits[..., ::2, ::2], self.target)
        self.assertGreater(meter.means()["a"], 0)


class PerImageTest(unittest.TestCase):
    def test_matrices_sum_to_the_dataset_matrix(self):
        torch.manual_seed(0)
        model = nn.Conv2d(3, 3, 1)
        labels = torch.randint(0, 3, (4, 8, 8))
        labels[0, 0, :] = 255
        loader = [(torch.randn(2, 3, 8, 8), labels[:2], ["a", "b"]),
                  (torch.randn(2, 3, 8, 8), labels[2:], ["c", "d"])]
        images = []
        scores = evaluate(model, loader, torch.device("cpu"), 3, ignore_index=255, amp=False,
                          per_image=images)
        self.assertEqual([name for name, _ in images], ["a", "b", "c", "d"])
        self.assertTrue(np.array_equal(sum(matrix for _, matrix in images),
                                       scores["confusion_matrix"]))
        self.assertEqual(images[0][1].sum(), 8 * 8 - 8)


class EvaluatorGuardTest(unittest.TestCase):
    CONFIG = types.SimpleNamespace(MEAN=(0.5, 0.5), PAD_DIVISOR=32, SEED=42, NUM_WORKERS=8)

    def test_protocol_snapshot_must_match(self):
        evaluate_unified.check_protocol({"protocol": {"MEAN": [0.5, 0.5], "PAD_DIVISOR": 32}}, self.CONFIG)
        with self.assertRaisesRegex(ValueError, "MEAN, PAD_DIVISOR"):
            evaluate_unified.check_protocol({"protocol": {"MEAN": [0.1, 0.5], "PAD_DIVISOR": 16}},
                                            self.CONFIG)
        with self.assertRaisesRegex(ValueError, "协议快照"):
            evaluate_unified.check_protocol({}, self.CONFIG)

    def test_run_settings_are_exempt(self):
        evaluate_unified.check_protocol(
            {"protocol": {"SEED": 43, "NUM_WORKERS": 2, "PRETRAINED_SHA256": "x", "UNIFIED_DATASETS": {}}},
            self.CONFIG)

    def test_last_checkpoint_needs_the_switch(self):
        evaluate_unified.check_kind({"kind": "best_model"}, False)
        evaluate_unified.check_kind({"format_version": 3}, True)
        with self.assertRaisesRegex(ValueError, "--allow-last"):
            evaluate_unified.check_kind({"format_version": 3}, False)


class IdentifiabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy(DEFAULT_PATH)

    def without(self, dataset):
        return identifiability(self.taxonomy, [d for d in self.taxonomy.datasets if d != dataset])

    def test_all_datasets_identify_every_fine_class(self):
        report = identifiability(self.taxonomy)
        self.assertEqual(report["rank"], report["fine_classes"])
        self.assertEqual(report["indistinguishable"], [])
        self.assertEqual(report["overlaps"], {})

    def test_leaving_out_flair_loses_seven_pairs(self):
        report = self.without("flair_hub")
        self.assertEqual(report["rank"], 14)
        self.assertEqual(len(report["indistinguishable"]), 7)

    def test_leaving_out_potsdam_merges_road_and_car(self):
        pairs = self.without("isprs_potsdam")["indistinguishable"]
        self.assertIn(["road", "car"], pairs)


if __name__ == "__main__":
    unittest.main()
