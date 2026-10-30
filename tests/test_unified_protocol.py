# -*- coding: utf-8 -*-
"""Unified fine-class protocol v4: taxonomy, set loss, projection, sampler, configs, tools."""
import contextlib
import copy
import dataclasses
import importlib
import io
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import native_datasets as nd
from common import train
from common import unified_protocol as up
from common import unified_taxonomy as ut
from common.dataset import BalancedEpochSampler, StreamingEpochSampler, build_unified_dataloaders
from common.engine import (ConfusionMatrix, PIDNetLoss, SegLoss, SetCrossEntropyLoss,
                           build_criterion, evaluate, primary_logits,
                           sliding_window_logits)
from common.models import build_model
from common.pretrained import REGISTRY
from tests.test_native_datasets import MAKERS, make_oem
from tools import evaluate_unified, prepare_native_dataset, run_baselines

FIVE = ("openearthmap", "flair_hub", "global_loveda", "deepglobe", "isprs_potsdam")
# Prepared copy names the base config expects under <datasets>/prepared.
PREPARED = {
    "openearthmap": "OpenEarthMap_native_v1",
    "flair_hub": "FLAIR-HUB_native_v1",
    "global_loveda": "Global-LoveDA_native_v1",
    "deepglobe": "DeepGlobe_native_v1",
    "isprs_potsdam": "ISPRS_Potsdam_native_v1",
}
SYNTHETIC = ("openearthmap", "flair_hub", "global_loveda", "deepglobe")
# The synthetic OEM release is not plan B (a recorded subset), and the two
# synthetic FLAIR train/val departements need a larger fraction to give a val split.
PREPARE_EXTRA = {"openearthmap": ["--limit-per-group", "100"],
                 "flair_hub": ["--val-fraction", "0.5"]}


def _purge_unified_configs():
    for name in [n for n in sys.modules if n.startswith(up.CONFIG_PREFIX)]:
        del sys.modules[name]


@contextlib.contextmanager
def _draft_taxonomy():
    """Unified configs imported inside see the taxonomy file with status draft."""
    load = ut.load_taxonomy
    try:
        with mock.patch.object(ut, "load_taxonomy",
                               lambda *a, **k: dataclasses.replace(load(*a, **k), status="draft")):
            yield
    finally:
        _purge_unified_configs()


class TaxonomyFileTest(unittest.TestCase):
    def setUp(self):
        self.taxonomy = ut.load_taxonomy()

    def test_taxonomy_covers_every_native_class_of_the_five_datasets(self):
        t = self.taxonomy
        self.assertEqual(t.datasets, FIVE)
        self.assertEqual(t.status, "final")
        self.assertEqual(len(t.fine_keys), 19)
        for dataset_id in FIVE:
            with self.subTest(dataset=dataset_id):
                matrix = t.membership(dataset_id)
                names = nd.get_spec(dataset_id).class_names
                self.assertEqual(tuple(t.native_sets(dataset_id)), names)
                self.assertEqual(matrix.shape, (len(names), len(t.fine_keys)))
                self.assertTrue((matrix.sum(1) >= 1).all())
                self.assertTrue((matrix.sum(0) >= 1).all())  # every fine class has a home
        # A fine class in two native classes is never penalised there (v1's ``car``).
        self.assertEqual(t.overlaps(), {d: () for d in FIVE})
        self.assertIn("isprs_potsdam", t.anchored_by("car"))

    def test_report_lists_every_uncertain_cell(self):
        text = ut.markdown(self.taxonomy)
        uncertain = sum(cell.basis == "uncertain" for cell in self.taxonomy.cells.values())
        self.assertIn(f"待核对的格子（{uncertain} 个）", text)
        self.assertEqual(text.count("\n| `"), uncertain)  # one row per uncertain cell
        for i, key in enumerate(self.taxonomy.fine_keys):
            self.assertIn(f"| {i} | `{key}` |", text)

    def _broken(self, change):
        raw = json.loads(Path(ut.DEFAULT_PATH).read_text(encoding="utf-8"))
        change({entry["key"]: entry["map"] for entry in raw["fine_classes"]}, raw)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with tmp:
            json.dump(raw, tmp, ensure_ascii=False)
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_invalid_files_are_refused(self):
        def unknown_class(m, raw):
            m["road"]["openearthmap"]["to"] = ["street"]

        def missing_dataset(m, raw):
            del m["road"]["deepglobe"]

        def empty_native_class(m, raw):
            m["snow_ice"]["flair_hub"] = {"to": ["bare soil"], "basis": "inferred"}

        def superset(m, raw):
            for key in ("deciduous_tree", "coniferous_tree"):
                m[key]["openearthmap"]["to"] = ["tree", "rangeland"]

        def indistinguishable(m, raw):
            # Potsdam alone tells cars from roads.
            m["road"]["isprs_potsdam"]["to"] = ["car"]

        def bad_basis(m, raw):
            m["road"]["openearthmap"]["basis"] = "guess"

        def official_alternative(m, raw):
            m["road"]["openearthmap"]["alternatives"] = ["developed space"]

        def duplicate_key(m, raw):
            raw["fine_classes"].append(copy.deepcopy(raw["fine_classes"][0]))

        def bad_status(m, raw):
            raw["status"] = "reviewed"

        cases = {unknown_class: "street", missing_dataset: "deepglobe",
                 empty_native_class: "没有任何细类", superset: "包含",
                 indistinguishable: "无法区分", bad_basis: "basis",
                 official_alternative: "official", duplicate_key: "重复",
                 bad_status: "status"}
        for change, message in cases.items():
            with self.subTest(case=change.__name__):
                with self.assertRaisesRegex(ValueError, message):
                    ut.load_taxonomy(self._broken(change))


class SetCrossEntropyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.target = torch.randint(0, 3, (2, 4, 5))
        self.target[0, 0, :] = 255

    def test_identity_membership_is_cross_entropy(self):
        loss = SetCrossEntropyLoss(np.eye(3), ignore_index=255)(self.logits, self.target)
        torch.testing.assert_close(loss, F.cross_entropy(self.logits, self.target, ignore_index=255))

    def test_a_set_scores_the_summed_probability(self):
        membership = [[1, 1, 0], [0, 0, 1]]           # label 0 = {0, 1}, label 1 = {2}
        target = self.target.clamp(max=1).masked_fill(self.target == 255, 255)
        loss = SetCrossEntropyLoss(membership, ignore_index=255)(self.logits, target)
        probs = self.logits.softmax(1)
        in_set = torch.where(target == 0, probs[:, 0] + probs[:, 1], probs[:, 2])
        expected = -torch.log(in_set[target != 255]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_ignored_and_out_of_range_labels(self):
        loss_fn = SetCrossEntropyLoss(np.eye(3), ignore_index=255)
        empty = loss_fn(self.logits, torch.full_like(self.target, 255))
        self.assertEqual(empty.item(), 0.0)
        empty.backward()
        with self.assertRaisesRegex(ValueError, "范围"):
            loss_fn(self.logits, torch.full_like(self.target, 3))
        with self.assertRaisesRegex(ValueError, "至少一个"):
            SetCrossEntropyLoss([[1, 0, 0], [0, 0, 0]])

    def test_segmentation_losses_accept_a_membership(self):
        membership = [[1, 1, 0], [0, 0, 1], [1, 0, 0]]
        seg = SegLoss(ignore_index=255, dice_weight=0.0, deep_sup_weights=(1.0, 0.4),
                      set_membership=membership)
        aux = torch.randn(2, 3, 2, 3, requires_grad=True)
        total = seg([self.logits, aux], self.target)
        self.assertTrue(torch.isfinite(total))
        with self.assertRaisesRegex(ValueError, "DICE_WEIGHT"):
            SegLoss(ignore_index=255, dice_weight=1.0, set_membership=membership)
        pidnet = PIDNetLoss(ignore_index=255, set_membership=membership)
        outputs = {"main": self.logits, "aux": [aux],
                   "boundary": torch.full((2, 1, 4, 5), 5.0)}  # every pixel "on a boundary"
        self.assertTrue(torch.isfinite(pidnet(outputs, self.target)))
        self.assertGreater(pidnet.last_components["boundary_aware"].item(), 0)


class _Constant(torch.nn.Module):
    """Pointwise logits from a fixed table, so expected predictions are known."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("table", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.table.view(1, -1, 1, 1).expand(x.shape[0], -1, *x.shape[-2:]) + 0 * x[:, :1]


class ProjectionEvaluateTest(unittest.TestCase):
    def test_identity_projection_matches_plain_evaluation(self):
        torch.manual_seed(2)
        model = torch.nn.Conv2d(3, 4, 1)
        loader = DataLoader(TensorDataset(torch.randn(3, 3, 16, 16),
                                          torch.randint(0, 4, (3, 16, 16))), batch_size=1)
        cpu = torch.device("cpu")
        plain = evaluate(model, loader, cpu, 4, ignore_index=255, amp=False)
        projected = evaluate(model, loader, cpu, 4, ignore_index=255, amp=False,
                             projection=np.eye(4))
        np.testing.assert_array_equal(plain["confusion_matrix"], projected["confusion_matrix"])
        np.testing.assert_array_equal(projected["fine_confusion"], plain["confusion_matrix"])

    def test_native_class_with_the_largest_set_probability_wins(self):
        # fine probabilities ~ (0.4, 0.35, 0.25): fine argmax is 0, but the set
        # {1, 2} of native class 1 holds 0.6.
        model = _Constant(np.log([0.4, 0.35, 0.25]))
        labels = torch.tensor([[[0, 1], [1, 255]]])
        loader = DataLoader(TensorDataset(torch.zeros(1, 3, 2, 2), labels), batch_size=1)
        scores = evaluate(model, loader, torch.device("cpu"), 2, ignore_index=255, amp=False,
                          projection=[[1, 0, 0], [0, 1, 1]])
        self.assertEqual(scores["confusion_matrix"].tolist(), [[0, 1], [0, 2]])
        self.assertEqual(scores["fine_confusion"].tolist(), [[1, 0, 0], [2, 0, 0]])
        with self.assertRaisesRegex(ValueError, "projection"):
            evaluate(model, loader, torch.device("cpu"), 3, ignore_index=255, amp=False,
                     projection=[[1, 0, 0], [0, 1, 1]])


class BalancedEpochSamplerTest(unittest.TestCase):
    def test_each_dataset_contributes_its_own_stream(self):
        sizes = [5, 3, 11]
        sampler = BalancedEpochSampler(sizes, 12, seed=4)
        self.assertEqual(sampler.counts, [4, 4, 4])
        self.assertEqual(len(sampler), 12)
        epochs = [list(sampler) for _ in range(3)]
        offsets = [0, 5, 8]
        for e, indices in enumerate(epochs):
            self.assertEqual(len(indices), 12)
            for d, (offset, size) in enumerate(zip(offsets, sizes)):
                mine = [i - offset for i in indices if offset <= i < offset + size]
                stream = StreamingEpochSampler(size, 4, 4 + 1009 * (d + 1))
                stream.set_epoch(e)
                self.assertEqual(sorted(mine), sorted(stream))
        resumed = BalancedEpochSampler(sizes, 12, seed=4)
        resumed.set_epoch(2)
        self.assertEqual(list(resumed), epochs[2])
        self.assertNotEqual(list(BalancedEpochSampler(sizes, 12, seed=5)), epochs[0])

    def test_weights_split_the_epoch_by_largest_remainder(self):
        self.assertEqual(BalancedEpochSampler([9, 9, 9], 10, 0, [2, 1, 1]).counts, [5, 3, 2])
        with self.assertRaises(ValueError):
            BalancedEpochSampler([9, 9, 9], 2, 0)
        with self.assertRaises(ValueError):
            BalancedEpochSampler([9, 9], 10, 0, [1, 0])


class UnifiedCopies(unittest.TestCase):
    """The four synthetic prepared copies, built once per class and trained together."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.datasets = Path(cls.tmp.name)
        for dataset_id in SYNTHETIC:
            src = cls.datasets / "raw" / dataset_id
            (make_oem if dataset_id == "openearthmap" else MAKERS[dataset_id])(src)
            with contextlib.redirect_stdout(io.StringIO()):
                code = prepare_native_dataset.main([
                    "--dataset", dataset_id, "--src", str(src),
                    "--dst", str(cls.datasets / "prepared" / PREPARED[dataset_id]),
                    "--workers", "2", *PREPARE_EXTRA.get(dataset_id, [])])
            assert code == 0, dataset_id

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.saved = {key: os.environ.get(key) for key in ("LAND_COVER_DATASETS", up.DATASETS_ENV)}
        os.environ["LAND_COVER_DATASETS"] = str(self.datasets)

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _purge_unified_configs()

    def unified(self, model="ddrnet", datasets=SYNTHETIC):
        return up.import_unified_config(up.MODEL_CONFIGS[model], list(datasets))


class UnifiedConfigTest(UnifiedCopies):
    def test_datasets_offsets_and_membership(self):
        cfg = self.unified()
        taxonomy = ut.load_taxonomy(cfg.TAXONOMY_PATH)
        self.assertEqual(cfg.UNIFIED_DATASET_IDS, list(SYNTHETIC))
        self.assertEqual(cfg.RUN_SCOPE, "+".join(SYNTHETIC))
        self.assertEqual(cfg.NUM_CLASSES, 19)
        self.assertEqual(cfg.CLASS_NAMES, list(taxonomy.fine_keys))
        self.assertEqual(cfg.LABEL_OFFSETS,
                         {"openearthmap": 0, "flair_hub": 8, "global_loveda": 23, "deepglobe": 30})
        self.assertEqual(len(cfg.SET_MEMBERSHIP), 36)
        np.testing.assert_array_equal(np.array(cfg.SET_MEMBERSHIP[8:23]),
                                      taxonomy.membership("flair_hub"))
        self.assertEqual((cfg.EPOCHS, cfg.SCHEDULE_EPOCHS, cfg.ITERS_PER_EPOCH), (160, 160, 1000))
        self.assertEqual(cfg.EPOCHS % cfg.EVAL_INTERVAL, 0)
        self.assertIsNone(cfg.EARLY_STOP_PATIENCE)
        self.assertFalse(hasattr(cfg, "TAXONOMY_ID") or hasattr(cfg, "DATA_ROOT"))
        self.assertIn(os.path.join("unified_v4", cfg.RUN_SCOPE), cfg.CKPT_DIR)
        for dataset_id in SYNTHETIC:
            with self.subTest(dataset=dataset_id):
                values = nd.config_values(str(self.datasets / "prepared" / PREPARED[dataset_id]),
                                          bands="rgb")
                entry = cfg.UNIFIED_DATASETS[dataset_id]
                self.assertEqual((entry["train_list"], entry["val_list"]),
                                 (values["TRAIN_LIST"], values["VAL_LIST"]))
                self.assertEqual((entry["split_id"], entry["class_names"]),
                                 (values["SPLIT_ID"], list(values["CLASS_NAMES"])))
                self.assertEqual(sorted(entry["test_lists"]), sorted(values["TEST_SPLITS"]))
        deepglobe = cfg.UNIFIED_DATASETS["deepglobe"]
        self.assertEqual((deepglobe["eval_window"], deepglobe["eval_stride"]), (1024, 768))
        self.assertIsNone(cfg.UNIFIED_DATASETS["openearthmap"]["eval_window"])
        self.assertEqual(cfg.UNIFIED_DATASETS["flair_hub"]["val_batch_size"], 16)

    def test_dataset_subset_must_come_from_the_taxonomy(self):
        with self.assertRaisesRegex(ValueError, "loveda"):
            self.unified(datasets=["openearthmap", "loveda"])
        with self.assertRaisesRegex(ValueError, "不是统一细类配置"):
            up.import_unified_config("config_other_ddrnet")

    def test_a_copy_without_selection_split_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, prepared = Path(tmp, "raw"), Path(tmp, "prepared", PREPARED["flair_hub"])
            MAKERS["flair_hub"](src)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare_native_dataset.main([
                    "--dataset", "flair_hub", "--src", str(src), "--dst", str(prepared)]), 0)
            self.assertNotIn("val", nd.load_metadata(str(prepared))["splits"])
            with self.assertRaisesRegex(ValueError, "选择划分"):
                up.dataset_entry(tmp, "flair_hub", {"flair_hub": {"prepared": PREPARED["flair_hub"]}})

    def test_controlled_pairs_share_their_recipe(self):
        unet, unetpp = self.unified("unet_r50"), self.unified("unetpp_r50")
        for key in ("LR", "WEIGHT_DECAY", "POLY_POWER", "ENCODER_NAME",
                    "UNET_DECODER_CHANNELS", "PRETRAINED"):
            self.assertEqual(getattr(unet, key), getattr(unetpp, key), key)
        ddrnet, pidnet = self.unified("ddrnet"), self.unified("pidnet_s_in1k")
        for key in ("LR", "WEIGHT_DECAY", "POLY_POWER", "DEEP_SUP_WEIGHTS"):
            self.assertEqual(getattr(ddrnet, key), getattr(pidnet, key), key)
        b2, b0 = self.unified("segformer_b2_in1k"), self.unified("segformer_b0_in1k")
        for key in ("LR", "WEIGHT_DECAY", "POLY_POWER", "DROP_PATH_RATE", "MODEL_DROPOUT"):
            self.assertEqual(getattr(b2, key), getattr(b0, key), key)
        self.assertEqual(self.unified("swin_upernet").DEEP_SUP_WEIGHTS, (1.0, 0.4))
        self.assertEqual(self.unified("deeplabv3plus_r50").DEEPLAB_BACKBONE, "resnet50")

    def test_leaves_name_registered_weights_and_unique_artifacts(self):
        seen = set()
        for key in up.MODEL_CONFIGS:
            cfg = self.unified(key)
            self.assertIn(cfg.PRETRAINED, REGISTRY, key)
            self.assertEqual((cfg.CROP_SIZE, cfg.BATCH_SIZE, cfg.IN_CHANNELS), (512, 8, 3))
            self.assertFalse(cfg.EVAL_AMP)
            for attr in ("EXP_NAME", "CKPT_DIR", "OUT_DIR"):
                value = (attr, os.path.normcase(getattr(cfg, attr)))
                self.assertNotIn(value, seen)
                seen.add(value)

    def test_leaves_build_fine_logits(self):
        sample = torch.zeros(1, 3, 64, 64)
        with torch.inference_mode():
            for key in up.MODEL_CONFIGS:
                logits = primary_logits(build_model(self.unified(key)).eval()(sample))
                self.assertEqual(tuple(logits.shape), (1, 19, 64, 64), key)

    def test_fingerprint_names_taxonomy_and_datasets(self):
        cfg = self.unified()
        snapshot = train.protocol_snapshot(cfg)
        for key in ("UNIFIED_TAXONOMY_ID", "UNIFIED_TAXONOMY_SHA256", "FINE_CLASS_NAMES",
                    "UNIFIED_DATASET_IDS", "UNIFIED_DATASETS", "LABEL_OFFSETS", "SET_MEMBERSHIP",
                    "DATASET_WEIGHTS", "SELECTION_METRIC", "ITERS_PER_EPOCH", "EVAL_INTERVAL"):
            self.assertIn(key, snapshot)
        smaller = self.unified(datasets=SYNTHETIC[:3])
        self.assertNotEqual(train.protocol_fingerprint(snapshot),
                            train.protocol_fingerprint(train.protocol_snapshot(smaller)))

    def test_only_a_final_taxonomy_allows_formal_runs(self):
        cfg = self.unified()
        self.assertEqual(cfg.UNIFIED_TAXONOMY_STATUS, "final")
        train.apply_run_overrides(cfg, types.SimpleNamespace(seed=7, run_tag="u_s42"))
        self.assertEqual(cfg.SEED, 7)
        self.assertEqual(Path(cfg.CKPT_DIR).parts[-3:], ("baselines", "u_s42", cfg.MODEL_SLUG))
        draft = self.unified()
        draft.UNIFIED_TAXONOMY_STATUS = "draft"
        with self.assertRaisesRegex(ValueError, "final"):
            train.apply_run_overrides(draft, types.SimpleNamespace(seed=None, run_tag="u_s42"))
        train.apply_run_overrides(draft, types.SimpleNamespace(seed=7, run_tag=None))
        self.assertEqual(draft.SEED, 7)

    def test_training_reads_only_train_and_selection_splits(self):
        cfg = self.unified()
        train.validate_training_domain(cfg)
        oem = cfg.UNIFIED_DATASETS["openearthmap"]
        oem["val_list"] = oem["test_lists"]["test_city"]
        with self.assertRaisesRegex(ValueError, "test_city"):
            train.validate_training_domain(cfg)

    def test_loaders_stack_labels_and_keep_native_selection_labels(self):
        cfg = self.unified()
        cfg.ITERS_PER_EPOCH, cfg.BATCH_SIZE, cfg.NUM_WORKERS = 4, 2, 0
        train_loader, val_loaders = build_unified_dataloaders(cfg)
        self.assertEqual(len(train_loader), 4)
        self.assertEqual(train_loader.sampler.counts, [2, 2, 2, 2])
        ranges = {d: (cfg.LABEL_OFFSETS[d], cfg.LABEL_OFFSETS[d] + e["num_classes"])
                  for d, e in cfg.UNIFIED_DATASETS.items()}
        seen = []
        for _, labels, _ in train_loader:
            for lab in labels:
                values = set(lab.unique().tolist()) - {255}
                owner = [d for d, (lo, hi) in ranges.items() if values and values <= set(range(lo, hi))]
                self.assertEqual(len(owner), 1, values)
                seen.extend(owner)
        self.assertEqual(sorted(seen), sorted(SYNTHETIC * 2))
        self.assertEqual(list(val_loaders), list(SYNTHETIC))
        for dataset_id, loader in val_loaders.items():
            _, labels, _ = next(iter(loader))
            native = set(labels.unique().tolist()) - {255}
            self.assertTrue(native <= set(range(cfg.UNIFIED_DATASETS[dataset_id]["num_classes"])))

    def test_criterion_and_selection_scores(self):
        cfg = self.unified("unet_r50")
        criterion = build_criterion(cfg)
        self.assertIsInstance(criterion.ce, SetCrossEntropyLoss)
        self.assertEqual(tuple(criterion.ce.allowed.shape), (36, 19))
        pidnet = build_criterion(self.unified("pidnet_s_in1k"))
        self.assertIsInstance(pidnet.semantic.ce, SetCrossEntropyLoss)
        cfg.NUM_WORKERS = 0
        _, val_loaders = build_unified_dataloaders(cfg)
        scores = up.evaluate_unified(build_model(cfg), val_loaders, torch.device("cpu"), cfg)
        self.assertEqual(list(scores["datasets"]), list(SYNTHETIC))
        self.assertAlmostEqual(scores["miou"],
                               np.mean([s["miou"] for s in scores["datasets"].values()]))
        self.assertEqual(scores["datasets"]["flair_hub"]["fine_confusion"].shape, (15, 19))
        self.assertIn("选模指标", up.format_unified_scores(scores, cfg))


class UnifiedToolsTest(UnifiedCopies):
    def _checkpoint(self, cfg, path, **changes):
        snapshot = {**evaluate_unified.config_snapshot(cfg), **changes}
        torch.save({"kind": "best_model", "epoch": 7, "model": build_model(cfg).state_dict(),
                    "config": snapshot, "protocol_fingerprint": "test",
                    "source_selection_value": 0.4,
                    "source_selection_by_dataset": {d: 0.4 for d in cfg.UNIFIED_DATASET_IDS}}, path)

    def test_evaluator_scores_every_test_split_on_native_classes(self):
        cfg = self.unified("unet_r50")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, out = Path(tmp, "best.pth"), Path(tmp, "eval")
            self._checkpoint(cfg, checkpoint)
            os.environ.pop(up.DATASETS_ENV, None)  # the checkpoint names its datasets
            with contextlib.redirect_stdout(io.StringIO()):
                code = evaluate_unified.main([
                    "--model", "unet_r50", "--checkpoint", str(checkpoint),
                    "--output-dir", str(out), "--num-workers", "0"])
            self.assertEqual(code, 0)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(metrics["datasets"]), sorted(SYNTHETIC))  # JSON keys are sorted
            self.assertFalse(metrics["selection_eligible"])
            self.assertEqual(metrics["taxonomy"]["status"], "final")
            oem = metrics["datasets"]["openearthmap"]
            self.assertEqual(sorted(oem["splits"]), ["test_city", "test_indomain"])
            split = oem["splits"]["test_city"]["metrics"]
            self.assertEqual([c["name"] for c in split["classes"]], list(nd.OPENEARTHMAP.class_names))
            fine = np.array(split["fine_confusion"]["counts"])
            self.assertEqual(fine.shape, (8, 19))
            self.assertEqual(fine.sum(1).tolist(), [c["support"] for c in split["classes"]])
            self.assertEqual(metrics["checkpoint"]["epoch"], 7)

    def test_evaluator_refuses_another_taxonomy(self):
        cfg = self.unified()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, out = Path(tmp, "best.pth"), Path(tmp, "eval")
            self._checkpoint(cfg, checkpoint, UNIFIED_TAXONOMY_SHA256="0" * 64)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = evaluate_unified.main([
                    "--model", "ddrnet", "--checkpoint", str(checkpoint),
                    "--output-dir", str(out), "--num-workers", "0"])
            self.assertEqual(code, 2)
            self.assertIn("UNIFIED_TAXONOMY_SHA256", err.getvalue())
            self.assertFalse(out.exists())

    def test_launcher_resolves_unified_leaves(self):
        os.environ[up.DATASETS_ENV] = ",".join(SYNTHETIC)
        runs = run_baselines.build_runs(["ddrnet", "unet_r50"], "u_s42", 42)
        self.assertEqual([r.config for r in runs], ["config_unified_ddrnet", "config_unified_unet_r50"])
        for run in runs:
            self.assertEqual(run.summary["taxonomy_status"], "final")
            self.assertEqual(run.summary["datasets"], list(SYNTHETIC))
            self.assertEqual(run.summary["max_iters"], 160_000)
            self.assertEqual(len(run.lists), 2 * len(SYNTHETIC))
            self.assertIn(os.path.join("unified_v4", "+".join(SYNTHETIC), "baselines", "u_s42"),
                          str(run.output_dir))
        run_baselines.preflight(runs)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_baselines.main(
                ["--models", "ddrnet", "--run-tag", "u_s42", "--dry-run"]), 0)
        # A draft taxonomy is planned (dry-run) but never trained by the launcher.
        with _draft_taxonomy(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(run_baselines.main(
                ["--models", "ddrnet", "--run-tag", "u_s42", "--dry-run"]), 0)
            self.assertEqual(run_baselines.main(["--models", "ddrnet", "--run-tag", "u_s42"]), 2)
        self.assertIn("draft", err.getvalue())
        self.assertFalse(runs[0].log_path.exists())
        parser = run_baselines.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--models", "unet", "--run-tag", "u_s42"])


class ScoresTest(unittest.TestCase):
    def test_excluded_classes_leave_the_means_and_oa(self):
        matrix = ConfusionMatrix(num_classes=3, ignore_index=255)
        # pred, target; the last pixel is ignored
        matrix.update(np.array([0, 0, 2, 1, 1, 2, 0, 1]), np.array([0, 0, 0, 1, 1, 2, 2, 255]))
        config = types.SimpleNamespace(CLASS_NAMES=("a", "b", "clutter"),
                                       REPORT_EXCLUDE_CLASSES=("clutter",))
        metrics = evaluate_unified.serialize_scores(matrix.scores(), config)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 1], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["oa"], 5 / 7)
        self.assertAlmostEqual(metrics["miou_present_excluding"], (1 / 2 + 1) / 2)
        self.assertAlmostEqual(metrics["mf1_present_excluding"], (2 / 3 + 1) / 2)
        # the "a" pixel predicted as clutter stays an error
        self.assertAlmostEqual(metrics["oa_excluding"], 4 / 5)
        json.dumps(metrics)

        config = types.SimpleNamespace(CLASS_NAMES=("a", "b", "clutter"))
        metrics = evaluate_unified.serialize_scores(matrix.scores(), config)
        self.assertNotIn("oa_excluding", metrics)


class StreamingEpochSamplerTest(unittest.TestCase):
    def test_every_pass_is_a_permutation(self):
        sampler = StreamingEpochSampler(10, 4, seed=1)
        stream = [i for _ in range(5) for i in sampler]
        self.assertEqual(sorted(stream[:10]), list(range(10)))
        self.assertEqual(sorted(stream[10:]), list(range(10)))
        long = StreamingEpochSampler(3, 7, seed=1)
        counts = [list(long).count(i) for i in range(3)]
        self.assertTrue(all(c in (2, 3) for c in counts))

    def test_epochs_depend_only_on_seed_and_epoch(self):
        sequential = StreamingEpochSampler(13, 5, seed=7)
        epochs = [list(sequential) for _ in range(4)]
        resumed = StreamingEpochSampler(13, 5, seed=7)
        resumed.set_epoch(2)
        self.assertEqual(list(resumed), epochs[2])
        self.assertEqual(list(resumed), epochs[3])  # advances after an epoch
        self.assertNotEqual(list(StreamingEpochSampler(13, 5, seed=8)), epochs[0])
        self.assertEqual(len(sequential), 5)


class _Pointwise(torch.nn.Module):
    """A 1x1 convolution: windowed and full-image logits must agree."""

    def __init__(self, classes=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, classes, 1)

    def forward(self, x):
        return self.conv(x)


class SlidingWindowTest(unittest.TestCase):
    def test_windows_reproduce_pointwise_logits(self):
        torch.manual_seed(0)
        model, image = _Pointwise().eval(), torch.randn(2, 3, 96, 160)
        with torch.no_grad():
            full = model(image)
            torch.testing.assert_close(sliding_window_logits(model, image, 64, 48), full)
            torch.testing.assert_close(sliding_window_logits(model, image, 256, 256), full)
        with self.assertRaises(ValueError):
            sliding_window_logits(model, image, 64, 96)

    def test_evaluate_with_window_matches_full_image(self):
        torch.manual_seed(1)
        model = _Pointwise()
        images = torch.randn(3, 3, 64, 96)
        labels = torch.randint(0, 3, (3, 64, 96))
        loader = DataLoader(TensorDataset(images, labels), batch_size=1)
        device = torch.device("cpu")
        full = evaluate(model, loader, device, 3, ignore_index=255, amp=False)
        windowed = evaluate(model, loader, device, 3, ignore_index=255, amp=False,
                            window=32, stride=16)
        self.assertEqual(full["miou"], windowed["miou"])


@unittest.skipUnless(Path("/data/yfk/datasets/prepared/ISPRS_Potsdam_native_v1").is_dir(),
                     "real prepared copies are only on the training server")
class RealPreparedCopiesTest(unittest.TestCase):
    def test_all_five_datasets_load_together(self):
        saved = {key: os.environ.pop(key, None) for key in ("LAND_COVER_DATASETS", up.DATASETS_ENV)}
        try:
            cfg = up.import_unified_config("config_unified_ddrnet")
            self.assertEqual(cfg.UNIFIED_DATASET_IDS, list(FIVE))
            self.assertEqual(cfg.RUN_SCOPE, "all")
            self.assertEqual(cfg.LABEL_OFFSETS, {"openearthmap": 0, "flair_hub": 8,
                                                 "global_loveda": 23, "deepglobe": 30,
                                                 "isprs_potsdam": 36})
            self.assertEqual(len(cfg.SET_MEMBERSHIP), 42)
            self.assertEqual(cfg.EPOCHS * cfg.ITERS_PER_EPOCH, 200_000)
            for entry in cfg.UNIFIED_DATASETS.values():
                nd.check_selection_lists(entry["root"], entry["train_list"], entry["val_list"])
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
            _purge_unified_configs()


if __name__ == "__main__":
    unittest.main()
