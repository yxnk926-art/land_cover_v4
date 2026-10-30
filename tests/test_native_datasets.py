# -*- coding: utf-8 -*-
"""Native-class readers and the prepared-copy tool on synthetic mini datasets."""
import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from common import native_datasets as nd
from common.dataset import OpenEarthMapDataset
from tools import prepare_native_dataset as tool


def _save(path, array):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def make_flair(root, size=64):
    """Two train/val departements (one with two years) and one test departement."""
    rng = np.random.default_rng(0)
    for domain in ("D004-2021", "D017-2018", "D017-2021", "D012-2019"):
        for roi in ("AA-S1-1", "AB-S1-2"):
            for patch in ("0-0", "0-1"):
                image = Path(root, f"{domain}_AERIAL_RGBI", roi,
                             f"{domain}_AERIAL_RGBI_{roi}_{patch}.tif")
                _save(image, rng.integers(0, 255, (size, size, 4), dtype=np.uint8))
                label = str(image).replace("_AERIAL_RGBI", "_AERIAL_LABEL-COSIA")
                _save(label, rng.integers(0, 19, (size, size), dtype=np.uint8))


def make_global_loveda(root, size=64):
    rng = np.random.default_rng(1)
    for split, ids in (("Train", range(7541, 7551)), ("Val", range(10279, 10283))):
        for i in ids:
            mode_rgba = i % 3 == 0
            image = rng.integers(0, 255, (size, size, 4 if mode_rgba else 3), dtype=np.uint8)
            label = rng.integers(0, 9, (size, size), dtype=np.uint8)
            if mode_rgba:
                image[..., 3] = 255
                image[:8, :, 3] = 0
                label[:8] = 0  # no-data under alpha 0
            _save(Path(root, split, "images_png", f"{i}.png"), image)
            _save(Path(root, split, "masks_png", f"{i}.png"), label)


def make_deepglobe(root, size=64, count=20):
    rng = np.random.default_rng(2)
    palette = np.array(list(nd.DEEPGLOBE.raw_to_model), dtype=np.uint8)
    for i in range(count):
        image_id = 100000 + i
        _save(Path(root, "train", f"{image_id}_sat.jpg"),
              rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
        mask = palette[rng.integers(0, len(palette), (size, size))]
        _save(Path(root, "train", f"{image_id}_mask.png"), mask)


MAKERS = {"flair_hub": make_flair, "global_loveda": make_global_loveda,
          "deepglobe": make_deepglobe}

# Synthetic OEM release: a held-out city, two selection cities (gorakhpur has a
# tile on the overlap list), two train cities (daressalaam_10 overlaps train)
# and an unlabelled official test tile per city.
OEM_TILES = {
    "mahe": ([1, 2], [3]),
    "paris": ([1, 2], [3]),
    "gorakhpur": ([1, 2], [24]),
    "daressalaam": ([1, 2, 3], [10, 11]),
    "aachen": ([1, 2, 3], [4]),
}


def make_oem(root, size=32):
    rng = np.random.default_rng(3)
    lists = {"train": [], "val": [], "test": []}
    for city, (train_ids, val_ids) in OEM_TILES.items():
        for official, ids in (("train", train_ids), ("val", val_ids), ("test", [99])):
            for i in ids:
                name = f"{city}_{i}.tif"
                lists[official].append(name)
                _save(Path(root, city, "images", name),
                      rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
                if official != "test":  # official test tiles carry no label
                    _save(Path(root, city, "labels", name),
                          rng.integers(0, 9, (size, size), dtype=np.uint8))
    for official, names in lists.items():
        Path(root, f"{official}.txt").write_text("".join(f"{n}\n" for n in names),
                                                 encoding="utf-8")


# Synthetic Potsdam: 64² tiles cut into 32² patches (the real spec cuts 6000²
# tiles into 1000² patches); one tile per role plus the excluded 7_10.
POTSDAM_TILES = ("2_10", "2_11", "2_13", "7_10")
SMALL_POTSDAM = dataclasses.replace(nd.ISPRS_POTSDAM, patch_size=32)


def small_potsdam():
    return mock.patch.dict(nd.SPECS, {"isprs_potsdam": SMALL_POTSDAM})


def make_potsdam(root, size=64):
    rng = np.random.default_rng(4)
    palette = np.array(list(nd.ISPRS_POTSDAM.raw_to_model), dtype=np.uint8)
    for tile in POTSDAM_TILES:
        _save(Path(root, "2_Ortho_RGB", "2_Ortho_RGB", f"top_potsdam_{tile}_RGB.tif"),
              rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
        label = palette[rng.integers(0, len(palette), (size, size))]
        label[:4] = (3, 2, 214)  # off-palette building colour, as in tile 4_12
        _save(Path(root, "5_Labels_all_noBoundary", f"top_potsdam_{tile}_label_noBoundary.tif"),
              label)


class EncodeLabelTest(unittest.TestCase):
    def test_flair_ignores_unscored_cosia_codes(self):
        raw = np.array([[0, 14, 15, 18]], dtype=np.uint8)
        np.testing.assert_array_equal(nd.encode_label(nd.FLAIR, raw), [[0, 14, 255, 255]])
        with self.assertRaisesRegex(ValueError, "未声明"):
            nd.encode_label(nd.FLAIR, np.array([[19]], dtype=np.uint8))

    def test_global_loveda_follows_loveda_encoding(self):
        raw = np.array([[0, 1, 2, 7, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(nd.encode_label(nd.GLOBAL_LOVEDA, raw),
                                      [[255, 0, 1, 6, 255]])

    def test_deepglobe_colours_are_thresholded(self):
        raw = np.array([[[0, 255, 255], [250, 251, 3], [255, 255, 255], [3, 2, 1]]],
                       dtype=np.uint8)
        np.testing.assert_array_equal(nd.encode_label(nd.DEEPGLOBE, raw), [[0, 1, 5, 255]])
        with self.assertRaisesRegex(ValueError, "未声明"):
            nd.encode_label(nd.DEEPGLOBE, np.array([[[255, 0, 0]]], dtype=np.uint8))

    def test_openearthmap_ignores_unknown(self):
        raw = np.array([[0, 1, 2, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(nd.encode_label(nd.OPENEARTHMAP, raw), [[255, 0, 1, 7]])
        with self.assertRaisesRegex(ValueError, "未声明"):
            nd.encode_label(nd.OPENEARTHMAP, np.array([[9]], dtype=np.uint8))

    def test_potsdam_snaps_off_palette_colours(self):
        raw = np.array([[[255, 255, 255], [0, 0, 255], [3, 2, 214], [252, 255, 0],
                         [0, 0, 0], [255, 0, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(nd.encode_label(nd.ISPRS_POTSDAM, raw),
                                      [[0, 1, 1, 4, 255, 5]])
        with self.assertRaisesRegex(ValueError, "未声明"):
            nd.encode_label(nd.ISPRS_POTSDAM, np.array([[[255, 0, 255]]], dtype=np.uint8))

    def test_wrong_label_shape_fails(self):
        with self.assertRaises(ValueError):
            nd.encode_label(nd.FLAIR, np.zeros((2, 2, 3), dtype=np.uint8))


class SplitTest(unittest.TestCase):
    @staticmethod
    def _flair_samples():
        samples = []
        units = {"D004": ["D004-2021"], "D017": ["D017-2018", "D017-2021"],
                 "D033": ["D033-2018", "D033-2021"], "D044": ["D044-2020"],
                 "D055": ["D055-2018"], "D012": ["D012-2019"]}
        for department, domains in units.items():
            official = "test" if department in nd.FLAIR_TEST_DEPARTMENTS else "trainval"
            for domain in domains:
                for k in range(10):
                    name = f"{domain}_AERIAL_RGBI_x_{k}.tif"
                    samples.append(nd.Sample(name, domain, department, official, name, name))
        return samples

    def test_flair_keeps_departements_whole_and_official_test(self):
        samples = self._flair_samples()
        split, recipe = nd.assign_splits(nd.FLAIR, samples, seed=7)
        by_unit = {}
        for s in samples:
            by_unit.setdefault(s.unit, set()).add(split[s.name])
        self.assertTrue(all(len(v) == 1 for v in by_unit.values()), by_unit)
        self.assertEqual(by_unit["D012"], {"test"})
        self.assertEqual(recipe["unit"], "departement")
        self.assertEqual(set(recipe["held_out_units"]["val"]),
                         {u for u, v in by_unit.items() if v == {"val"}})
        again, _ = nd.assign_splits(nd.FLAIR, samples, seed=7)
        self.assertEqual(split, again)

    def test_flair_drops_unreadable_images_after_the_split(self):
        samples = self._flair_samples()
        clean, _ = nd.assign_splits(nd.FLAIR, samples, seed=7)
        bad = "D033-2018_AERIAL_RGBI_x_3.tif"
        spec = dataclasses.replace(nd.FLAIR, excluded_images=frozenset((bad, "absent.tif")))
        split, recipe = nd.assign_splits(spec, samples, seed=7)
        # Every other sample keeps its split: the departements do not move.
        self.assertEqual(split, {n: p for n, p in clean.items() if n != bad})
        self.assertEqual(recipe["excluded"], {"rule": spec.excluded_rule, "names": [bad]})
        self.assertIn("D035-2020_AERIAL_RGBI_NA-S1-72_9-2.tif", nd.FLAIR.excluded_images)
        _, plain = nd.assign_splits(nd.GLOBAL_LOVEDA, samples[:20], seed=0)
        self.assertNotIn("excluded", plain)

    def test_global_loveda_val_becomes_test(self):
        samples = [nd.Sample(f"{i}.png", "train", f"{i}.png", "train", "", "") for i in range(50)]
        samples += [nd.Sample(f"v{i}.png", "val", f"v{i}.png", "val", "", "") for i in range(5)]
        split, _ = nd.assign_splits(nd.GLOBAL_LOVEDA, samples, seed=0)
        self.assertEqual({split[f"v{i}.png"] for i in range(5)}, {"test"})
        self.assertEqual(sum(split[f"{i}.png"] == "val" for i in range(50)), 10)

    def test_deepglobe_fractions_and_unknown_fraction(self):
        samples = [nd.Sample(f"{i}.jpg", "train", str(i), "train", "", "") for i in range(100)]
        split, _ = nd.assign_splits(nd.DEEPGLOBE, samples, seed=0)
        counts = {k: list(split.values()).count(k) for k in nd.SPLITS}
        self.assertEqual(counts, {"train": 70, "val": 10, "test": 20})
        with self.assertRaisesRegex(ValueError, "比例"):
            nd.assign_splits(nd.FLAIR, samples, fractions={"test": 0.1})

    def test_openearthmap_plan_b_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_oem(tmp)
            samples = nd.list_samples("openearthmap", tmp)
            split, recipe = nd.assign_splits(nd.OPENEARTHMAP, samples)
        self.assertNotIn("mahe_99.tif", split)  # unlabelled official test tile
        expected = {
            "mahe_1.tif": "test_city", "mahe_3.tif": "test_city",
            "paris_1.tif": "select", "paris_3.tif": "select",
            # on the overlap list, but gorakhpur is a selection city
            "gorakhpur_24.tif": "select",
            "daressalaam_1.tif": "train", "daressalaam_11.tif": "test_indomain",
            "aachen_2.tif": "train", "aachen_4.tif": "test_indomain",
        }
        self.assertEqual({k: split[k] for k in expected}, expected)
        self.assertNotIn("daressalaam_10.tif", split)
        self.assertEqual(recipe["excluded"]["names"], ["daressalaam_10.tif"])
        self.assertEqual(recipe["train_cities"], ["aachen", "daressalaam"])
        self.assertIsNone(recipe["seed"])
        with self.assertRaisesRegex(ValueError, "固定的划分清单"):
            nd.assign_splits(nd.OPENEARTHMAP, samples, fractions={"val": 0.2})

    def test_openearthmap_frozen_lists_are_consistent(self):
        self.assertEqual(len(nd.OEM_TEST_CITIES), 15)
        self.assertEqual(len(nd.OEM_SELECT_CITIES), 11)
        self.assertFalse(nd.OEM_TEST_CITIES & nd.OEM_SELECT_CITIES)
        v3_dev = {"buenos_aires", "christchurch", "kitsap", "paris", "pedrogao_grande",
                  "piura", "rio", "tuscaloosa"}
        self.assertLessEqual(v3_dev, nd.OEM_SELECT_CITIES)
        overlap_cities = {n.rsplit("_", 1)[0] for n in nd.OEM_VAL_TILES_OVERLAPPING_TRAIN}
        self.assertEqual(len(nd.OEM_VAL_TILES_OVERLAPPING_TRAIN), 33)
        self.assertFalse(overlap_cities & nd.OEM_TEST_CITIES)
        excluded = sum(1 for n in nd.OEM_VAL_TILES_OVERLAPPING_TRAIN
                       if n.rsplit("_", 1)[0] not in nd.OEM_SELECT_CITIES)
        spec = nd.OPENEARTHMAP
        self.assertEqual(sum(spec.expected_counts.values()) + excluded, 3500)
        self.assertEqual(set(spec.expected_counts), set(spec.split_names))
        self.assertEqual(set(spec.expected_list_sha256), set(spec.split_names))
        self.assertEqual(spec.test_splits, ("test_city", "test_indomain"))
        self.assertEqual(nd.FLAIR.test_splits, ("test",))

    def test_potsdam_tiles_keep_their_role_and_7_10_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp, small_potsdam():
            make_potsdam(tmp)
            samples = nd.list_samples("isprs_potsdam", tmp)
            split, recipe = nd.assign_splits(SMALL_POTSDAM, samples)
            with self.assertRaisesRegex(ValueError, "固定的划分清单"):
                nd.assign_splits(SMALL_POTSDAM, samples, fractions={"val": 0.2})
        self.assertEqual(len(samples), 16)
        roles = {}
        for s in samples:
            roles.setdefault(s.unit, set()).add(split.get(s.name, "excluded"))
        self.assertEqual(roles, {"2_10": {"val"}, "2_11": {"train"}, "2_13": {"test"},
                                 "7_10": {"excluded"}})
        by_name = {s.name: s for s in samples}
        self.assertEqual(by_name["top_potsdam_2_11_r1_c0.png"].window, (0, 32, 32, 64))
        self.assertEqual(recipe["excluded"]["tiles"], ["7_10"])
        self.assertEqual(len(recipe["excluded"]["names"]), 4)
        self.assertEqual(recipe["patch"], {"size": 32, "overlap": 0})

    def test_potsdam_frozen_tiles_are_consistent(self):
        test, val, excluded = (nd.POTSDAM_TEST_TILES, nd.POTSDAM_VAL_TILES,
                               nd.POTSDAM_EXCLUDED_TILES)
        self.assertEqual((len(test), len(val), len(excluded)), (14, 1, 1))
        self.assertFalse(test & val or test & excluded or val & excluded)
        per_tile = (6000 // nd.ISPRS_POTSDAM.patch_size) ** 2
        train_tiles = 38 - len(test) - len(val) - len(excluded)
        self.assertEqual(nd.ISPRS_POTSDAM.expected_counts,
                         {"train": train_tiles * per_tile, "val": per_tile,
                          "test": len(test) * per_tile})

    def test_wo_xbd_release_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_oem(tmp)
            os.remove(Path(tmp, "aachen", "images", "aachen_1.tif"))
            with self.assertRaisesRegex(FileNotFoundError, "xBD"):
                nd.list_samples("openearthmap", tmp)


class PrepareToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _prepare(self, dataset_id, *extra):
        src = self.base / f"raw_{dataset_id}"
        MAKERS[dataset_id](src)
        dst = self.base / f"prepared_{dataset_id}"
        with contextlib.redirect_stdout(io.StringIO()):
            code = tool.main(["--dataset", dataset_id, "--src", str(src), "--dst", str(dst),
                              "--workers", "2", *extra])
        self.assertEqual(code, 0)
        return src, dst

    def test_each_dataset_round_trips_through_openearthmap_dataset(self):
        for dataset_id in MAKERS:
            with self.subTest(dataset=dataset_id):
                _, dst = self._prepare(dataset_id)
                meta = nd.load_metadata(str(dst), dataset_id)
                spec = nd.get_spec(dataset_id)
                total = sum(info["samples"] for info in meta["splits"].values())
                self.assertEqual(total, len(nd.list_samples(dataset_id, meta["source_root"])))
                for split_name in meta["splits"]:
                    dataset = OpenEarthMapDataset(
                        str(dst), str(dst / f"{split_name}.txt"), training=False,
                        **nd.dataset_kwargs(str(dst)))
                    image, label, name = dataset[0]
                    self.assertEqual(image.shape[0], 3)
                    values = set(np.unique(label.numpy()).tolist())
                    self.assertLessEqual(values, set(range(spec.num_classes)) | {255})
                    self.assertTrue(name.endswith(spec.image_ext))

    def test_images_are_hard_links_and_labels_match_the_encoding(self):
        src, dst = self._prepare("flair_hub")
        sample = nd.list_samples("flair_hub", str(src))[0]
        linked = dst / sample.group / "images" / sample.name
        self.assertEqual(os.stat(linked).st_ino, os.stat(sample.image).st_ino)
        with Image.open(sample.label) as raw, \
                Image.open(dst / sample.group / "labels" / nd.label_name(sample.name)) as ours:
            np.testing.assert_array_equal(np.asarray(ours),
                                          nd.encode_label(nd.FLAIR, np.asarray(raw)))
        meta = nd.load_metadata(str(dst))
        self.assertEqual(meta["splits"]["test"]["groups"], ["D012-2019"])
        val_groups = set(meta["splits"].get("val", {}).get("groups", []))
        self.assertIn(val_groups & {"D017-2018", "D017-2021"},
                      (set(), {"D017-2018", "D017-2021"}))

    def test_flair_all_bands_gives_four_channels(self):
        _, dst = self._prepare("flair_hub")
        dataset = OpenEarthMapDataset(
            str(dst), str(dst / "train.txt"), training=True, crop_size=32,
            mean=(0.5,) * 4, std=(0.25,) * 4, **nd.dataset_kwargs(str(dst), bands="all"))
        image, label, _ = dataset[0]
        self.assertEqual(tuple(image.shape), (4, 32, 32))
        self.assertEqual(tuple(label.shape), (32, 32))

    def test_alpha_audit_and_ignored_raw_values_are_recorded(self):
        _, dst = self._prepare("global_loveda")
        meta = nd.load_metadata(str(dst))
        self.assertEqual(meta["audit"], {"labelled_pixels_under_alpha0": 0})
        self.assertIn("8", meta["splits"]["test"]["raw_value_pixels"])

    def test_config_values_name_every_split_list(self):
        _, dst = self._prepare("deepglobe")
        values = nd.config_values(str(dst))
        self.assertEqual(values["LABEL_SUFFIX"], ".png")
        self.assertEqual(values["NUM_CLASSES"], 6)
        for key in ("TRAIN_LIST", "VAL_LIST", "TEST_LIST"):
            self.assertTrue(os.path.isfile(values[key]), key)

    def test_existing_target_dry_run_and_tampering_are_refused(self):
        src, dst = self._prepare("deepglobe")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(["--dataset", "deepglobe", "--src", str(src), "--dst", str(dst)])
        dry = self.base / "dry"
        with contextlib.redirect_stdout(io.StringIO()):
            tool.main(["--dataset", "deepglobe", "--src", str(src), "--dst", str(dry),
                       "--dry-run"])
        self.assertFalse(dry.exists())
        with open(dst / "train.txt", "a", encoding="utf-8") as handle:
            handle.write("extra.jpg\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            nd.load_metadata(str(dst))

    def test_failed_run_leaves_nothing_behind(self):
        src = self.base / "raw_bad"
        make_deepglobe(src, count=3)
        _save(src / "train" / "100001_mask.png", np.full((64, 64, 3), (255, 0, 0), np.uint8))
        dst = self.base / "bad"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "未声明"):
            tool.main(["--dataset", "deepglobe", "--src", str(src), "--dst", str(dst),
                       "--workers", "1"])
        self.assertEqual([p.name for p in self.base.iterdir() if p.name.startswith((".bad", "bad"))],
                         [])

    def test_undecodable_image_fails_the_run(self):
        src = self.base / "raw_truncated"
        make_flair(src)
        image = sorted(src.glob("D004-2021_AERIAL_RGBI/*/*.tif"))[0]
        data = image.read_bytes()
        image.write_bytes(data[: len(data) // 2])
        dst = self.base / "truncated"
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(OSError, "无法完整解码"):
            tool.main(["--dataset", "flair_hub", "--src", str(src), "--dst", str(dst),
                       "--workers", "1"])
        self.assertFalse(dst.exists())

    def test_metadata_is_json_with_split_recipe(self):
        _, dst = self._prepare("global_loveda", "--seed", "3")
        meta = json.loads((dst / nd.METADATA_NAME).read_text(encoding="utf-8"))
        self.assertEqual(meta["split_recipe"]["seed"], 3)
        self.assertEqual(meta["layout"]["label_suffix"], ".png")
        self.assertEqual(meta["spec_sha256"], nd.spec_fingerprint(nd.GLOBAL_LOVEDA))
        self.assertEqual(meta["split_id"], nd.split_id(meta))

    def _prepare_oem(self, *extra):
        src = self.base / "raw_oem"
        make_oem(src)
        dst = self.base / "prepared_oem"
        with contextlib.redirect_stdout(io.StringIO()):
            code = tool.main(["--dataset", "openearthmap", "--src", str(src), "--dst", str(dst),
                              "--workers", "2", *extra])
        self.assertEqual(code, 0)
        return src, dst

    def test_openearthmap_copy_names_four_roles(self):
        # The synthetic release is not plan B, so build it as a recorded subset.
        _, dst = self._prepare_oem("--limit-per-group", "100")
        meta = nd.load_metadata(str(dst), "openearthmap")
        self.assertEqual(list(meta["splits"]), list(nd.OPENEARTHMAP.split_names))
        self.assertEqual(meta["split_recipe"]["excluded"]["names"], ["daressalaam_10.tif"])
        self.assertFalse((dst / "daressalaam" / "images" / "daressalaam_10.tif").exists())
        values = nd.config_values(str(dst))
        self.assertEqual(values["VAL_LIST"], values["SELECT_LIST"])
        self.assertEqual(values["TEST_SPLITS"], ["test_city", "test_indomain"])
        self.assertEqual(values["SPLIT_ID"], meta["split_id"])
        self.assertNotIn("TEST_LIST", values)
        for key in ("TRAIN_LIST", "SELECT_LIST", "TEST_CITY_LIST", "TEST_INDOMAIN_LIST"):
            dataset = OpenEarthMapDataset(str(dst), values[key], training=False,
                                          **nd.dataset_kwargs(str(dst)))
            image, label, _ = dataset[0]
            self.assertEqual(image.shape[0], 3)
            self.assertLessEqual(set(np.unique(label.numpy()).tolist()), set(range(8)) | {255})

    def test_openearthmap_complete_copy_must_match_frozen_split(self):
        src = self.base / "raw_oem"
        make_oem(src)
        dst = self.base / "prepared_oem"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "冻结"):
            tool.main(["--dataset", "openearthmap", "--src", str(src), "--dst", str(dst)])
        self.assertFalse(dst.exists())

    def test_pinned_lists_are_checked_on_load(self):
        _, dst = self._prepare_oem("--limit-per-group", "100")
        path = dst / nd.METADATA_NAME
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["subset"] = None  # pretend the synthetic copy is a complete release
        path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "冻结"):
            nd.load_metadata(str(dst))

    def test_potsdam_patches_are_exact_crops(self):
        src = self.base / "raw_potsdam"
        make_potsdam(src)
        dst = self.base / "prepared_potsdam"
        with small_potsdam():
            with contextlib.redirect_stdout(io.StringIO()):
                # The synthetic release is not the full one: build a recorded subset.
                code = tool.main(["--dataset", "isprs_potsdam", "--src", str(src),
                                  "--dst", str(dst), "--workers", "2",
                                  "--limit-per-group", "100"])
            self.assertEqual(code, 0)
            meta = nd.load_metadata(str(dst), "isprs_potsdam")
            values = nd.config_values(str(dst))
            sample = next(s for s in nd.list_samples("isprs_potsdam", str(src))
                          if s.name == "top_potsdam_2_13_r0_c1.png")
            dataset = OpenEarthMapDataset(str(dst), values["TRAIN_LIST"], training=False,
                                          **nd.dataset_kwargs(str(dst)))
            image, label, _ = dataset[0]
        self.assertEqual({k: v["samples"] for k, v in meta["splits"].items()},
                         {"train": 4, "val": 4, "test": 4})
        self.assertFalse((dst / "7_10").exists())
        self.assertEqual(os.path.basename(values["VAL_LIST"]), "val.txt")
        with Image.open(sample.image) as raw_image, Image.open(sample.label) as raw_label, \
                Image.open(dst / "2_13" / "images" / sample.name) as patch, \
                Image.open(dst / "2_13" / "labels" / sample.name) as patch_label:
            np.testing.assert_array_equal(np.asarray(patch),
                                          np.asarray(raw_image.crop(sample.window)))
            np.testing.assert_array_equal(
                np.asarray(patch_label),
                nd.encode_label(nd.ISPRS_POTSDAM, np.asarray(raw_label.crop(sample.window))))
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertLessEqual(set(np.unique(label.numpy()).tolist()), set(range(6)) | {255})

    def test_test_splits_never_reach_training(self):
        _, dst = self._prepare_oem("--limit-per-group", "100")
        root = str(dst)
        values = nd.config_values(root)
        self.assertIsNotNone(nd.check_selection_lists(root, values["TRAIN_LIST"],
                                                      values["VAL_LIST"]))
        for test_key in ("TEST_CITY_LIST", "TEST_INDOMAIN_LIST"):
            with self.assertRaisesRegex(ValueError, "测试集"):
                nd.check_selection_lists(root, values["TRAIN_LIST"], values[test_key])
        mixed = self.base / "train_plus_indomain.txt"
        mixed.write_text("daressalaam_1.tif\ndaressalaam_11.tif\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "test_indomain"):
            nd.check_selection_lists(root, str(mixed), values["VAL_LIST"])
        swapped = self.base / "select_as_train.txt"
        swapped.write_text("paris_1.tif\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "只能取 train.txt"):
            nd.check_selection_lists(root, str(swapped), values["VAL_LIST"])
        self.assertIsNone(nd.check_selection_lists(str(self.base), "unused", "unused"))


_REAL_OEM = os.path.join(os.environ.get("LAND_COVER_DATASETS", "/data/yfk/datasets"),
                         "prepared", "OpenEarthMap_native_v1")


@unittest.skipUnless(os.path.isfile(os.path.join(_REAL_OEM, nd.METADATA_NAME)),
                     "需要服务器上的 OEM 准备副本")
class RealOpenEarthMapCopyTest(unittest.TestCase):
    def test_prepared_copy_is_plan_b(self):
        meta = nd.load_metadata(_REAL_OEM, "openearthmap")  # also checks pinned hashes
        self.assertIsNone(meta["subset"])
        self.assertEqual({k: v["samples"] for k, v in meta["splits"].items()},
                         nd.OPENEARTHMAP.expected_counts)
        self.assertEqual(len(meta["split_recipe"]["excluded"]["names"]), 30)
        values = nd.config_values(_REAL_OEM)
        self.assertEqual(values["VAL_LIST"], values["SELECT_LIST"])
        nd.check_selection_lists(_REAL_OEM, values["TRAIN_LIST"], values["VAL_LIST"])


_REAL_POTSDAM = os.path.join(os.environ.get("LAND_COVER_DATASETS", "/data/yfk/datasets"),
                             "prepared", "ISPRS_Potsdam_native_v1")


@unittest.skipUnless(os.path.isfile(os.path.join(_REAL_POTSDAM, nd.METADATA_NAME)),
                     "需要服务器上的 Potsdam 准备副本")
class RealPotsdamCopyTest(unittest.TestCase):
    def test_prepared_copy_follows_the_tile_split(self):
        meta = nd.load_metadata(_REAL_POTSDAM, "isprs_potsdam")
        self.assertIsNone(meta["subset"])
        self.assertEqual({k: v["samples"] for k, v in meta["splits"].items()},
                         nd.ISPRS_POTSDAM.expected_counts)
        self.assertEqual(set(meta["splits"]["test"]["groups"]), nd.POTSDAM_TEST_TILES)
        self.assertEqual(meta["splits"]["val"]["groups"], ["2_10"])
        self.assertEqual(meta["split_recipe"]["excluded"]["tiles"], ["7_10"])


_REAL_FLAIR = os.path.join(os.environ.get("LAND_COVER_DATASETS", "/data/yfk/datasets"),
                           "prepared", "FLAIR-HUB_native_v1")


@unittest.skipUnless(os.path.isfile(os.path.join(_REAL_FLAIR, nd.METADATA_NAME)),
                     "需要服务器上的 FLAIR-HUB 准备副本")
class RealFlairCopyTest(unittest.TestCase):
    def test_unreadable_release_image_is_excluded(self):
        meta = nd.load_metadata(_REAL_FLAIR, "flair_hub")
        self.assertIsNone(meta["subset"])
        self.assertEqual({k: v["samples"] for k, v in meta["splits"].items()},
                         {"train": 151974, "val": 38425, "test": 50700})
        self.assertEqual(meta["split_recipe"]["held_out_units"]["val"],
                         ["D024047", "D030", "D032", "D045", "D046", "D059062", "D067", "D077"])
        self.assertEqual(meta["split_recipe"]["excluded"]["names"],
                         sorted(nd.FLAIR_UNREADABLE_IMAGES))
        for name in meta["splits"]:
            with open(os.path.join(_REAL_FLAIR, f"{name}.txt"), encoding="utf-8") as handle:
                self.assertFalse(nd.FLAIR_UNREADABLE_IMAGES & {line.strip() for line in handle})


if __name__ == "__main__":
    unittest.main()
