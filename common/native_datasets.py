# -*- coding: utf-8 -*-
"""Native-class readers for the v4 datasets: FLAIR-HUB, Global-LoveDA, DeepGlobe,
OpenEarthMap (full release, city split plan B) and ISPRS Potsdam (5 cm, cut
into patches).

Each dataset is described once: its native class list, how raw label values
map to model IDs (ignore = 255), how image and label files pair up, and how the
splits are formed.  ``tools/prepare_native_dataset.py`` uses these specs to
build an immutable prepared copy::

    <root>/<group>/images/<name>      hard link to the raw image (Potsdam:
                                      a lossless PNG patch cut from the tile)
    <root>/<group>/labels/<stem>.png  re-encoded index label (0..C-1, 255)
    <root>/<split>.txt                image basenames per split: train/val/test,
                                      OpenEarthMap train/select/test_city/test_indomain
    <root>/native_dataset.json        classes, mapping, split recipe, hashes

Model selection (early stopping, best checkpoint, HPO) may only read the
spec's ``selection_split``; ``check_selection_lists`` enforces that test
splits never reach training.

Training and evaluation then read it through the ordinary
``OpenEarthMapDataset`` (``dataset_kwargs``/``config_values`` give the data
keys), so augmentation, padding and strict label checks are shared with the
OEM pipeline.  Which taxonomy a protocol trains on is not decided here: these
are the datasets' own classes, kept apart from the frozen v3 six-class map.

Only the standard library, NumPy and PIL are used, so preflight needs no torch.
"""
from dataclasses import dataclass, field
import glob
import hashlib
import json
import os
import random

import numpy as np

NATIVE_FORMAT = "native-v1"
IGNORE_INDEX = 255
METADATA_NAME = "native_dataset.json"
LABEL_SUFFIX = ".png"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    name: str        # image basename in the prepared copy (unique)
    group: str       # directory in the prepared copy
    unit: str        # kept together when splitting (department / image)
    official: str    # partition of the release: trainval / train / val / test
    image: str
    label: str
    window: object = None  # (left, top, right, bottom) patch of a large tile, or None


@dataclass(frozen=True)
class NativeSpec:
    dataset_id: str
    title: str
    raw_dir: str                 # directory name under <datasets>/raw
    class_names: tuple
    raw_to_model: dict           # raw id or RGB triple -> model id / IGNORE_INDEX
    label_mode: str              # "index" or "rgb"
    image_ext: str
    rgb_bands: object = None     # bands for 3-channel models (None = image is RGB)
    all_bands: object = None     # every informative band, e.g. RGB + NIR
    alpha_is_nodata: bool = False
    default_fractions: dict = field(default_factory=dict)
    notes: tuple = ()
    split_names: tuple = SPLITS        # list files, in report order
    selection_split: str = "val"       # the only split model selection may read
    expected_counts: object = None     # {split: samples} of the complete release
    expected_list_sha256: object = None  # pinned list hashes of the complete release
    patch_size: object = None          # cut large tiles into non-overlapping patches
    excluded_images: frozenset = frozenset()  # undecodable release files, dropped after the split
    excluded_rule: str = ""

    @property
    def num_classes(self):
        return len(self.class_names)

    @property
    def test_splits(self):
        return tuple(n for n in self.split_names if n not in ("train", self.selection_split))


# ---------------------------------------------------------------- FLAIR-HUB
# Official split_1 (HF data card): 16 test departements, the rest train/val.
FLAIR_TEST_DEPARTMENTS = frozenset((
    "D012", "D015", "D022", "D026", "D036", "D061", "D064", "D068",
    "D069", "D071", "D073", "D075", "D076", "D083", "D084", "D085",
))
FLAIR_TRAINVAL_DEPARTMENTS = frozenset((
    "D004", "D005", "D006", "D007", "D008", "D009", "D010", "D011", "D013",
    "D014", "D016", "D017", "D018", "D020", "D021", "D023", "D024047",
    "D025039", "D029", "D030", "D031", "D032", "D033", "D034", "D035",
    "D037", "D038", "D040", "D041", "D044", "D045", "D046", "D049", "D051",
    "D052", "D054057", "D055", "D056", "D058", "D059062", "D060", "D063",
    "D065", "D066", "D067", "D070", "D072", "D074", "D077", "D078", "D080",
    "D081", "D086", "D091",
))
# Images of the official release that cannot be decoded.  They are dropped
# after the seeded split, so the held-out departements stay as they were.
FLAIR_UNREADABLE_IMAGES = frozenset((
    # TIFF strip byte count 0 (552,960 bytes), found when training read it on
    # 2026-10-28; shipped as such in the official D035-2020_AERIAL_RGBI.zip
    # (SHA256 as published on Hugging Face, unzip CRC passes).
    "D035-2020_AERIAL_RGBI_NA-S1-72_9-2.tif",
))

FLAIR = NativeSpec(
    dataset_id="flair_hub",
    title="FLAIR-HUB AERIAL_RGBI + AERIAL_LABEL-COSIA (split_1)",
    raw_dir="FLAIR-HUB",
    # COSIA codes 0-14 as in the official config_supervision.yaml; 15-18
    # (clear cut, ligneous, mixed, undefined) carry weight 0 in the official
    # training and are not part of the 15-class benchmark, so they are ignored.
    class_names=(
        "building", "greenhouse", "swimming pool", "impervious surface",
        "pervious surface", "bare soil", "water", "snow",
        "herbaceous vegetation", "agricultural land", "plowed land",
        "vineyard", "deciduous", "coniferous", "brushwood",
    ),
    raw_to_model={**{i: i for i in range(15)}, 15: 255, 16: 255, 17: 255, 18: 255},
    label_mode="index",
    image_ext=".tif",
    rgb_bands=(0, 1, 2),
    all_bands=(0, 1, 2, 3),
    default_fractions={"val": 0.2},
    notes=(
        "Images are 4-band uint8 RGBI GeoTIFFs at 0.2 m (PIL reads them as RGBA: "
        "band 3 is near-infrared, not alpha).",
        "The official validation CSV is not published with the HF release; val "
        "is a seeded whole-department subset of the train/val departements "
        "(~20%, like the official 38,175 / 190,400).",
        "Departements with two acquisition years stay in one split.",
        "D035-2020_AERIAL_RGBI_NA-S1-72_9-2.tif (train) cannot be decoded in the "
        "official release; it is excluded after the split.",
    ),
    excluded_images=FLAIR_UNREADABLE_IMAGES,
    excluded_rule="image cannot be decoded in the official release",
)


def _list_flair(src):
    samples = []
    for image_dir in sorted(glob.glob(os.path.join(src, "*_AERIAL_RGBI"))):
        domain = os.path.basename(image_dir)[: -len("_AERIAL_RGBI")]
        department = domain.split("-")[0]
        if department in FLAIR_TEST_DEPARTMENTS:
            official = "test"
        elif department in FLAIR_TRAINVAL_DEPARTMENTS:
            official = "trainval"
        else:
            raise ValueError(f"{domain} 不在 FLAIR-HUB split_1 的省份列表中")
        for image in sorted(glob.glob(os.path.join(image_dir, "*", "*.tif"))):
            label = image.replace("_AERIAL_RGBI", "_AERIAL_LABEL-COSIA")
            samples.append(Sample(os.path.basename(image), domain, department,
                                  official, image, label))
    return samples


# ---------------------------------------------------------------- Global-LoveDA
GLOBAL_LOVEDA = NativeSpec(
    dataset_id="global_loveda",
    title="Global-LoveDA (EarthVL-GLOBAL) Train/Val",
    raw_dir="Global-LoveDA",
    # LoveDA encoding: 0 no-data, 1 background ... 7 agriculture.  Raw ID 8
    # (about 0.2% of pixels) has no LoveDA counterpart and is ignored.
    class_names=("background", "building", "road", "water", "barren",
                 "forest", "agriculture"),
    raw_to_model={0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 255},
    label_mode="index",
    image_ext=".png",
    alpha_is_nodata=True,
    default_fractions={"val": 0.2},
    notes=(
        "Official Val (672) is the test split (Test labels are not public); "
        "val is a seeded image-level subset of Train.",
        "No scene/location metadata ships with the release, so the train/val "
        "split cannot be spatially grouped.",
        "Some images are RGBA with alpha 0/255; RGB models drop alpha, and the "
        "preparation reports labelled pixels under alpha 0.",
        "Raw ID 8 (not in the LoveDA taxonomy) is ignored; its share is recorded.",
    ),
)


def _list_global_loveda(src):
    samples = []
    for official in ("Train", "Val"):
        for image in sorted(glob.glob(os.path.join(src, official, "images_png", "*.png"))):
            label = os.path.join(src, official, "masks_png", os.path.basename(image))
            name = os.path.basename(image)
            samples.append(Sample(name, official.lower(), name, official.lower(),
                                  image, label))
    return samples


# ---------------------------------------------------------------- DeepGlobe
DEEPGLOBE = NativeSpec(
    dataset_id="deepglobe",
    title="DeepGlobe Land Cover 2018 (labelled train set)",
    raw_dir="DeepGlobe_LandCover",
    class_names=("urban", "agriculture", "rangeland", "forest", "water", "barren"),
    raw_to_model={(0, 255, 255): 0, (255, 255, 0): 1, (255, 0, 255): 2,
                  (0, 255, 0): 3, (0, 0, 255): 4, (255, 255, 255): 5,
                  (0, 0, 0): 255},
    label_mode="rgb",
    image_ext=".jpg",
    default_fractions={"val": 0.1, "test": 0.2},
    notes=(
        "Only the 803 train images have masks (valid/test are unlabelled), so "
        "train/val/test are seeded image-level subsets; no coordinates exist "
        "for spatial grouping.",
        "Masks are binarised at 128 per channel as the challenge recommends; "
        "'unknown' (black) is ignored.",
    ),
)


def _list_deepglobe(src):
    samples = []
    for image in sorted(glob.glob(os.path.join(src, "train", "*_sat.jpg"))):
        image_id = os.path.basename(image)[: -len("_sat.jpg")]
        label = os.path.join(src, "train", f"{image_id}_mask.png")
        samples.append(Sample(f"{image_id}.jpg", "train", image_id, "train", image, label))
    return samples


# ---------------------------------------------------------------- OpenEarthMap
# City-level split, plan B of the 2026-10-25 draft (confirmed 2026-10-28).  The
# official test tiles have no labels and official val shares all 97 cities with
# train, so the 3,500 labelled train+val tiles are re-split by city:
#   test_city      15 held-out cities, all tiles (final report only)
#   select         11 cities, all tiles (early stopping, checkpoint and HPO choice)
#   train          official train tiles of the other 71 cities
#   test_indomain  official val tiles of those 71 cities, minus the tiles in
#                  OEM_VAL_TILES_OVERLAPPING_TRAIN (final report, not spatially
#                  independent of train)
# The held-out cities came from a seeded search (seed 20261025, 200,000 draws of
# whole clusters, cities whose nearest tiles are < 50 km apart kept together)
# that balances native classes, continent and GSD.  The selection set is the v3
# dev cities plus sechura (same cluster as piura), khartoum and gorakhpur, which
# add Africa, Asia, Bareland and Agriculture.
OEM_TEST_CITIES = frozenset((
    "mahe", "ngaoundere", "baybay", "chiangmai", "tokyo", "chisinau", "pomorskie",
    "vienna", "zachodniopomorskie", "houston", "little_rock", "tulsa", "adelaide",
    "lima", "rosario",
))
OEM_SELECT_CITIES = frozenset((
    "buenos_aires", "christchurch", "kitsap", "paris", "pedrogao_grande", "piura",
    "rio", "tuscaloosa", "sechura", "khartoum", "gorakhpur",
))
# Official val tiles whose area is more than 1% covered by official train tiles
# (from the tile footprints, 2026-10-25).  Under plan B the three gorakhpur
# tiles belong to the selection set, so 30 are dropped from test_indomain.
OEM_VAL_TILES_OVERLAPPING_TRAIN = frozenset(f"{n}.tif" for n in (
    "daressalaam_10", "daressalaam_20", "daressalaam_31", "daressalaam_85",
    "el_rodeo_18", "gorakhpur_24", "gorakhpur_45", "gorakhpur_64", "jeremie_37",
    "jeremie_38", "joplin_30", "joplin_49", "leilane_estates_40", "les_cayes_23",
    "les_cayes_27", "les_cayes_4", "mexico_city_13", "mexico_city_31",
    "mexico_city_37", "niamey_5", "oklahoma_25", "oklahoma_50", "panama_city_14",
    "panama_city_45", "rotterdam_11", "rotterdam_12", "rotterdam_14", "rotterdam_17",
    "rotterdam_35", "saint_louis_du_sud_19", "santa_rosa_25", "thousand_oaks_25",
    "wallace_45",
))

OPENEARTHMAP = NativeSpec(
    dataset_id="openearthmap",
    title="OpenEarthMap full (xBD images restored) train+val, city split plan B",
    raw_dir="OpenEarthMap",
    # Official encoding: 0 unknown, 1 bareland ... 8 building.  Unknown is not a
    # class and is ignored, as in the OEM benchmark.
    class_names=("bareland", "rangeland", "developed space", "road", "tree",
                 "water", "agriculture land", "building"),
    raw_to_model={0: 255, **{i: i - 1 for i in range(1, 9)}},
    label_mode="index",
    image_ext=".tif",
    split_names=("train", "select", "test_city", "test_indomain"),
    selection_split="select",
    expected_counts={"train": 2124, "select": 483, "test_city": 539, "test_indomain": 324},
    # Lists of the complete release, cross-checked against an independent
    # rebuild from the per-tile metadata of the draft.
    expected_list_sha256={
        "train": "9b442a181c6056b81a6e4c2c5ea1713043370d453a54d413df990fc36f356fb0",
        "select": "1e9f199785d02a6fed987e2dad14683ea9d4d52159dcdefa481463f60183f6ab",
        "test_city": "70df6cb23364f65883992f4148bf1f1ad6981bfd06f09f0d5acbd45f67fc2fa1",
        "test_indomain": "19d9fa898c3b409e71231aba09335135a46b06b24663fd4e28db1762b582cd52",
    },
    notes=(
        "Full release with the xBD RGB images restored (813 of the 3,500 labelled "
        "tiles); the wo_xBD release lacks them and is refused.",
        "Official test tiles (1,500) have no labels and are not used.",
        "Splits are fixed city lists (plan B), not seeded fractions; select is "
        "the only split model selection may read, test_city and test_indomain "
        "are for the final report.",
        "test_indomain shares cities with train: report it as in-domain, not "
        "spatially independent.",
    ),
)


def _list_openearthmap(src):
    samples, missing = [], []
    for official in ("train", "val"):
        with open(os.path.join(src, f"{official}.txt"), "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
        for name in names:
            city = name.rsplit("_", 1)[0]
            image = os.path.join(src, city, "images", name)
            if not os.path.isfile(image):
                missing.append(name)
            samples.append(Sample(name, city, city, official, image,
                                  os.path.join(src, city, "labels", name)))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} 张 OEM 影像缺失，如 {missing[:3]}"
            "（wo_xBD 版本不含 xBD 影像，需要补全后的 full 版本）")
    return samples


def _assign_openearthmap(samples):
    split, excluded = {}, []
    for s in samples:
        if s.unit in OEM_TEST_CITIES:
            split[s.name] = "test_city"
        elif s.unit in OEM_SELECT_CITIES:
            split[s.name] = "select"
        elif s.official == "train":
            split[s.name] = "train"
        elif s.name in OEM_VAL_TILES_OVERLAPPING_TRAIN:
            excluded.append(s.name)
        else:
            split[s.name] = "test_indomain"
    recipe = {
        "seed": None,
        "plan": "OEM city split plan B (draft 2026-10-25, confirmed 2026-10-28)",
        "unit": "city",
        "held_out_units": {"select": sorted(OEM_SELECT_CITIES),
                           "test_city": sorted(OEM_TEST_CITIES)},
        "train_cities": sorted({s.unit for s in samples if split.get(s.name) == "train"}),
        "excluded": {
            "rule": "official val tile of a train city whose area is more than 1% "
                    "covered by official train tiles",
            "names": sorted(excluded),
        },
        "notes": list(OPENEARTHMAP.notes),
    }
    return split, recipe


# ---------------------------------------------------------------- ISPRS Potsdam
# 38 true-orthophoto tiles of 6000² px at 5 cm.  Split as in BANet / UNetFormer
# (Wang et al.): the 14 official test tiles, 2_10 for validation, and the other
# 22 tiles for training; 7_10 is dropped for its erroneous annotation (also
# TreeSegNet).  Tiles are cut into non-overlapping patches, so every pixel of a
# test tile is scored exactly once and a tile never spans two splits.
POTSDAM_TEST_TILES = frozenset((
    "2_13", "2_14", "3_13", "3_14", "4_13", "4_14", "4_15", "5_13", "5_14", "5_15",
    "6_13", "6_14", "6_15", "7_13",
))
POTSDAM_VAL_TILES = frozenset(("2_10",))
POTSDAM_EXCLUDED_TILES = frozenset(("7_10",))

ISPRS_POTSDAM = NativeSpec(
    dataset_id="isprs_potsdam",
    title="ISPRS Potsdam RGB, eroded-boundary labels, 1000² patches",
    raw_dir="ISPRS_Potsdam",
    class_names=("impervious surfaces", "building", "low vegetation", "tree", "car",
                 "clutter"),
    # Official colours; black is the eroded object boundary of the
    # "noBoundary" reference and is ignored.  Tiles 4_12 and 6_7 carry
    # off-palette colours (at most 42 per channel from a palette colour), which
    # the 128-per-channel threshold snaps to the nearest palette colour.
    raw_to_model={(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2,
                  (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5, (0, 0, 0): 255},
    label_mode="rgb",
    image_ext=".png",
    expected_counts={"train": 792, "val": 36, "test": 504},
    patch_size=1000,
    notes=(
        "Only RGB orthophotos (2_Ortho_RGB) and the eroded-boundary reference "
        "(5_Labels_all_noBoundary) are available; boundary pixels are ignored "
        "in training as well as in evaluation.",
        "Split as BANet / UNetFormer: 14 official test tiles, val 2_10, train the "
        "other 22 tiles; 7_10 is excluded for its erroneous annotation.",
        "Each 6000² tile is cut into 6x6 non-overlapping 1000² patches written as "
        "lossless PNG (not hard links).",
        "Clutter is kept as a class; that literature reports mean F1 / mIoU over "
        "the other five classes.",
    ),
)


def _list_potsdam(src):
    from PIL import Image  # reads tile headers only
    Image.MAX_IMAGE_PIXELS = None
    size = get_spec("isprs_potsdam").patch_size
    # The release unpacks RGB tiles either directly or one level deeper.
    images = sorted(glob.glob(os.path.join(src, "2_Ortho_RGB", "*_RGB.tif"))
                    + glob.glob(os.path.join(src, "2_Ortho_RGB", "2_Ortho_RGB", "*_RGB.tif")))
    samples = []
    for image in images:
        tile = os.path.basename(image)[len("top_potsdam_"): -len("_RGB.tif")]
        label = os.path.join(src, "5_Labels_all_noBoundary",
                             f"top_potsdam_{tile}_label_noBoundary.tif")
        with Image.open(image) as handle:
            width, height = handle.size
        if width % size or height % size:
            raise ValueError(f"Potsdam 图块 {tile} 的尺寸 {width}x{height} 不能被 {size} 整除")
        official = "test" if tile in POTSDAM_TEST_TILES else "train"
        for row in range(height // size):
            for col in range(width // size):
                window = (col * size, row * size, (col + 1) * size, (row + 1) * size)
                samples.append(Sample(f"top_potsdam_{tile}_r{row}_c{col}.png", tile, tile,
                                      official, image, label, window))
    return samples


def _assign_potsdam(samples):
    split, excluded = {}, set()
    for s in samples:
        if s.unit in POTSDAM_EXCLUDED_TILES:
            excluded.add(s.unit)
        elif s.unit in POTSDAM_TEST_TILES:
            split[s.name] = "test"
        elif s.unit in POTSDAM_VAL_TILES:
            split[s.name] = "val"
        else:
            split[s.name] = "train"
    recipe = {
        "seed": None,
        "plan": "BANet / UNetFormer split (Wang et al.)",
        "unit": "tile",
        "held_out_units": {"val": sorted(POTSDAM_VAL_TILES), "test": sorted(POTSDAM_TEST_TILES)},
        "train_tiles": sorted({s.unit for s in samples if split.get(s.name) == "train"}),
        "patch": {"size": get_spec("isprs_potsdam").patch_size, "overlap": 0},
        "excluded": {
            "rule": "tile 7_10 has erroneous annotation",
            "tiles": sorted(excluded),
            "names": sorted(s.name for s in samples if s.unit in excluded),
        },
        "notes": list(ISPRS_POTSDAM.notes),
    }
    return split, recipe


SPECS = {spec.dataset_id: spec
         for spec in (FLAIR, GLOBAL_LOVEDA, DEEPGLOBE, OPENEARTHMAP, ISPRS_POTSDAM)}
_LISTERS = {"flair_hub": _list_flair, "global_loveda": _list_global_loveda,
            "deepglobe": _list_deepglobe, "openearthmap": _list_openearthmap,
            "isprs_potsdam": _list_potsdam}
# Datasets split by fixed lists rather than seeded fractions.
_FIXED_ASSIGNERS = {"openearthmap": _assign_openearthmap, "isprs_potsdam": _assign_potsdam}


def get_spec(dataset_id):
    try:
        return SPECS[dataset_id]
    except KeyError:
        raise ValueError(f"未知数据集 {dataset_id!r}（可选: {', '.join(sorted(SPECS))}）") from None


def list_samples(dataset_id, src):
    """Pair every raw image with its label; missing labels or duplicate names fail."""
    samples = _LISTERS[get_spec(dataset_id).dataset_id](src)
    if not samples:
        raise FileNotFoundError(f"{src} 下没有找到 {dataset_id} 的影像")
    missing = [s.label for s in samples if not os.path.isfile(s.label)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} 张影像缺少标签，如 {missing[:3]}")
    names = [s.name for s in samples]
    if len(names) != len(set(names)):
        raise ValueError(f"{dataset_id} 的影像文件名有重复")
    return samples


# ---------------------------------------------------------------- labels
def _rgb_code(rgb):
    return (int(rgb[0] >= 128) << 2) | (int(rgb[1] >= 128) << 1) | int(rgb[2] >= 128)


def _lookup_table(spec):
    if spec.label_mode == "index":
        table = np.full(256, -1, dtype=np.int16)
        for raw, model_id in spec.raw_to_model.items():
            table[raw] = model_id
    else:
        table = np.full(8, -1, dtype=np.int16)
        for rgb, model_id in spec.raw_to_model.items():
            table[_rgb_code(rgb)] = model_id
    return table


def encode_label(spec, raw):
    """Raw label array -> uint8 model IDs (ignore 255); undeclared values fail."""
    raw = np.asarray(raw)
    if spec.label_mode == "index":
        if raw.ndim != 2 or raw.dtype != np.uint8:
            raise ValueError(f"{spec.dataset_id} 标签应为二维 uint8，实际 {raw.shape} {raw.dtype}")
        codes = raw
    else:
        if raw.ndim != 3 or raw.shape[-1] < 3:
            raise ValueError(f"{spec.dataset_id} 标签应为 RGB，实际 {raw.shape}")
        bits = raw[..., :3] >= 128
        codes = (bits[..., 0].astype(np.uint8) << 2) | (bits[..., 1].astype(np.uint8) << 1) \
            | bits[..., 2].astype(np.uint8)
    encoded = _lookup_table(spec)[codes]
    if (encoded < 0).any():
        unknown = np.unique(codes[encoded < 0]).tolist()
        raise ValueError(f"{spec.dataset_id} 标签含未声明的原始值：{unknown[:8]}")
    return encoded.astype(np.uint8)


# ---------------------------------------------------------------- splits
def _closest_prefix(units, sizes, target, rng):
    """Seeded order of whole units; the prefix whose size is closest to target."""
    order = sorted(units)
    rng.shuffle(order)
    best, best_gap, total = 0, target, 0
    for i, unit in enumerate(order, 1):
        total += sizes[unit]
        if abs(total - target) < best_gap:
            best, best_gap = i, abs(total - target)
    return set(order[:best])


def assign_splits(spec, samples, seed=42, fractions=None):
    """Return ({name: split}, recipe) with whole units kept in one split.

    Samples missing from the mapping are excluded from every split (OEM val
    tiles overlapping train, Potsdam 7_10, the spec's ``excluded_images``);
    the recipe lists them.
    """
    if spec.dataset_id in _FIXED_ASSIGNERS:
        if fractions:
            raise ValueError(f"{spec.dataset_id} 使用固定的划分清单，不接受比例参数")
        return _FIXED_ASSIGNERS[spec.dataset_id](samples)
    fractions = dict(spec.default_fractions if fractions is None else fractions)
    if set(fractions) - set(spec.default_fractions):
        raise ValueError(f"{spec.dataset_id} 只支持比例 {sorted(spec.default_fractions)}")
    rng = random.Random(seed)
    split = {}
    # Labelled official holdouts become the test split (FLAIR test, Global-
    # LoveDA Val); val (and DeepGlobe's test) are carved out of the rest.
    official_test = [s for s in samples if s.official in ("test", "val")]
    pool = [s for s in samples if s.official not in ("test", "val")]
    for s in official_test:
        split[s.name] = "test"

    sizes = {}
    for s in pool:
        sizes[s.unit] = sizes.get(s.unit, 0) + 1
    chosen = {}
    remaining = set(sizes)
    for part in ("test", "val"):
        if part not in fractions:
            continue
        target = fractions[part] * len(pool)
        chosen[part] = _closest_prefix(remaining, sizes, target, rng)
        remaining -= chosen[part]
    unit_split = {u: part for part, units in chosen.items() for u in units}
    for s in pool:
        split[s.name] = unit_split.get(s.unit, "train")
    # Dropped only now: the unit sizes above still count them, so excluding a
    # file never moves a whole unit to another split.
    excluded = sorted(name for name in spec.excluded_images if name in split)
    for name in excluded:
        del split[name]

    recipe = {
        "seed": seed,
        "fractions_of_pool": fractions,
        "official_test_source": sorted({s.official for s in official_test}),
        "unit": "departement" if spec.dataset_id == "flair_hub" else "image",
        "held_out_units": {part: sorted(units) for part, units in chosen.items()
                           if spec.dataset_id == "flair_hub"},
        "notes": list(spec.notes),
    }
    if spec.excluded_images:
        recipe["excluded"] = {"rule": spec.excluded_rule, "names": excluded}
    return split, recipe


# ---------------------------------------------------------------- prepared copies
def label_name(image_name):
    return os.path.splitext(image_name)[0] + LABEL_SUFFIX


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_fingerprint(spec):
    payload = {
        "dataset_id": spec.dataset_id, "class_names": list(spec.class_names),
        "raw_to_model": {str(k): v for k, v in spec.raw_to_model.items()},
        "label_mode": spec.label_mode, "ignore_index": IGNORE_INDEX,
    }
    if spec.patch_size is not None:  # absent for whole-image datasets: their hash is unchanged
        payload["patch_size"] = spec.patch_size
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def split_id(meta):
    """Identity of a prepared copy's split: a hash over every list's SHA256."""
    payload = {name: info["list_sha256"] for name, info in meta["splits"].items()}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_metadata(root, dataset_id=None, verify_lists=True):
    """Read and check ``native_dataset.json``; list files must match their hashes.

    A complete (not ``--limit-per-group``) copy of a dataset with pinned lists
    must carry exactly those lists.
    """
    path = os.path.join(root, METADATA_NAME)
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("format") != NATIVE_FORMAT:
        raise ValueError(f"{path} 不是 {NATIVE_FORMAT} 格式")
    if dataset_id is not None and meta.get("dataset_id") != dataset_id:
        raise ValueError(f"{root} 是 {meta.get('dataset_id')}，不是 {dataset_id}")
    spec = get_spec(meta["dataset_id"])
    if meta.get("spec_sha256") != spec_fingerprint(spec):
        raise ValueError(f"{root} 的类别/映射定义与当前代码不一致，请重新准备")
    unknown = set(meta["splits"]) - set(spec.split_names)
    if unknown:
        raise ValueError(f"{root} 含 {spec.dataset_id} 未定义的划分：{sorted(unknown)}")
    if "split_id" in meta and meta["split_id"] != split_id(meta):
        raise ValueError(f"{root} 的 split_id 与列表哈希不一致")
    if spec.expected_list_sha256 and not meta.get("subset"):
        actual = {name: info["list_sha256"] for name, info in meta["splits"].items()}
        if actual != spec.expected_list_sha256:
            raise ValueError(f"{root} 的划分列表与代码中冻结的 {spec.dataset_id} 划分不一致")
    if verify_lists:
        for split_name, info in meta["splits"].items():
            actual = sha256_file(os.path.join(root, f"{split_name}.txt"))
            if actual != info["list_sha256"]:
                raise ValueError(f"{split_name}.txt 已被改动（SHA256 不一致）")
    return meta


def config_values(root, bands="rgb"):
    """Upper-case data keys for a config that trains/evaluates on ``root``.

    ``bands="rgb"`` gives 3-channel input for the RGB models; ``"all"`` adds
    every informative band (FLAIR: NIR), which also needs a 4-value MEAN/STD.
    """
    meta = load_metadata(root)
    spec = get_spec(meta["dataset_id"])
    if bands not in ("rgb", "all"):
        raise ValueError("bands 必须是 'rgb' 或 'all'")
    image_bands = spec.rgb_bands if bands == "rgb" else (spec.all_bands or spec.rgb_bands)
    values = {
        "DATA_ROOT": root,
        "IMAGE_GLOB": f"*/images/*{spec.image_ext}",
        "LABEL_REPLACE": (("images", "labels"),),
        "LABEL_SUFFIX": LABEL_SUFFIX,
        "IMAGE_BANDS": image_bands,
        "NUM_CLASSES": spec.num_classes,
        "CLASS_NAMES": list(spec.class_names),
        "IGNORE_INDEX": IGNORE_INDEX,
        "STRICT_LABEL_SCHEMA": True,
        "SPLIT_ID": split_id(meta),
        "SELECTION_SPLIT": spec.selection_split,
        "TEST_SPLITS": [name for name in spec.test_splits if name in meta["splits"]],
        **{f"{name.upper()}_LIST": os.path.join(root, f"{name}.txt")
           for name in meta["splits"]},
    }
    # VAL_LIST is what training reads for early stopping and checkpoint choice,
    # so it always names the selection split (OEM: select.txt), never a test split.
    if spec.selection_split in meta["splits"]:
        values["VAL_LIST"] = os.path.join(root, f"{spec.selection_split}.txt")
    return values


def _read_names(path):
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def check_selection_lists(root, train_list, val_list):
    """Training on a prepared copy may read only train and the selection split.

    ``train_list`` must be a subset of ``train.txt`` and ``val_list`` (early
    stopping, best checkpoint, HPO) a subset of the selection split; any sample
    of a test split fails.  Roots without native metadata return None.
    """
    root = str(root)
    if not os.path.isfile(os.path.join(root, METADATA_NAME)):
        return None
    meta = load_metadata(root)
    spec = get_spec(meta["dataset_id"])
    tests = {name: _read_names(os.path.join(root, f"{name}.txt"))
             for name in spec.test_splits if name in meta["splits"]}
    for key, path, allowed in (("TRAIN_LIST", train_list, "train"),
                               ("VAL_LIST", val_list, spec.selection_split)):
        names = _read_names(path)
        for test_name, members in tests.items():
            leaked = names & members
            if leaked:
                raise ValueError(
                    f"{key} 含 {test_name} 的样本（{len(leaked)} 张，如 {sorted(leaked)[:3]}）；"
                    "测试集不能进入训练或模型选择")
        extra = names - _read_names(os.path.join(root, f"{allowed}.txt"))
        if extra:
            raise ValueError(f"{key} 只能取 {allowed}.txt 中的样本，多出 {len(extra)} 张，"
                             f"如 {sorted(extra)[:3]}")
    return meta


def dataset_kwargs(root, bands="rgb"):
    """Keyword arguments for ``OpenEarthMapDataset`` on a prepared copy."""
    values = config_values(root, bands)
    return {
        "image_glob": values["IMAGE_GLOB"], "label_replace": values["LABEL_REPLACE"],
        "label_suffix": values["LABEL_SUFFIX"], "image_bands": values["IMAGE_BANDS"],
        "num_classes": values["NUM_CLASSES"], "ignore_index": values["IGNORE_INDEX"],
        "strict_label_schema": True,
    }
