# 五个数据集的读取与准备副本

OpenEarthMap、FLAIR-HUB、Global-LoveDA、DeepGlobe 和 ISPRS Potsdam 都按各自的原生类别读入。
已知的分辨率有 0.05 m（Potsdam）、0.2 m（FLAIR）、约 0.25–0.5 m（OEM）和 0.5 m（DeepGlobe），
Global-LoveDA 的分辨率尚未核实。统一细类协议在训练时才给标签加偏移，准备副本本身不变。

## 结构

| 文件 | 作用 |
|---|---|
| `common/native_datasets.py` | 每个数据集定义一次：原生类别、原始标签 → 模型 ID 的映射（ignore=255）、影像与标签如何配对、怎样划分、哪个划分用于模型选择；另有读取元数据并生成数据键的 `config_values()` / `dataset_kwargs()`，以及检查训练列表的 `check_selection_lists()` |
| `tools/prepare_native_dataset.py` | 按上述定义生成只读的准备副本 |
| `common/dataset.py` | `OpenEarthMapDataset` 读取全部五个副本；`label_suffix` 使 JPEG 影像能配 PNG 标签 |
| `common/train.py` | `validate_training_domain()` 对每个数据集调用 `check_selection_lists()` |

准备副本的布局（读取时都用 `OpenEarthMapDataset`，增强、补边和严格标签检查完全一致）：

```text
<root>/<group>/images/<影像原文件名>    硬链接到 raw/，不复制；Potsdam 是从图块切出的无损 PNG 小块
<root>/<group>/labels/<文件名>.png      重编码后的索引标签：0..C-1，ignore=255
<root>/<划分>.txt                       每个划分的影像文件名：train/val/test；
                                        OpenEarthMap 为 train/select/test_city/test_indomain
<root>/native_dataset.json              类别、映射、划分配方、split_id、逐类像素数、列表 SHA256、生成代码哈希
```

`load_metadata()` 每次读取都会核对：格式版本、类别/映射定义与当前代码一致、只含该数据集定义的
划分、`split_id` 与各列表哈希一致、各列表文件的 SHA256 与元数据一致。OpenEarthMap 的完整副本
还要求列表哈希等于代码中冻结的值。任何一项不符都直接报错。

## 模型选择与测试集

每个数据集有一个 **选择划分**（`selection_split`）：best checkpoint 的选择只能读它。
其余非 train 划分都是测试划分，只在最终报告时使用。

| | 选择划分 | 测试划分 |
|---|---|---|
| FLAIR-HUB、Global-LoveDA、DeepGlobe、ISPRS Potsdam | `val` | `test` |
| OpenEarthMap | `select` | `test_city`、`test_indomain` |

- `config_values()` 中的 `VAL_LIST`（训练读它来选 checkpoint）永远指向选择划分，OEM 即
  `select.txt`；另给出 `SELECTION_SPLIT`、`TEST_SPLITS`、`SPLIT_ID`。
- `check_selection_lists(root, TRAIN_LIST, VAL_LIST)` 要求：
  - `TRAIN_LIST` 只含 `train.txt` 中的样本，`VAL_LIST` 只含选择划分中的样本；
  - 两者都不含任何测试划分的样本。
- `train.py` 在建立任何训练状态之前就做这项检查。

## FLAIR-HUB、Global-LoveDA、DeepGlobe

| | FLAIR-HUB | Global-LoveDA | DeepGlobe |
|---|---|---|---|
| 原始数据 | `raw/FLAIR-HUB`（AERIAL_RGBI + AERIAL_LABEL-COSIA） | `raw/Global-LoveDA` | `raw/DeepGlobe_LandCover` |
| 影像 | 512²，0.2 m，4 波段 uint8 RGBI（PIL 读作 RGBA，第 4 波段是近红外） | 1024² PNG，部分为 RGBA（alpha 只有 0/255） | 2448² JPEG |
| 原生类别 | 15 类：building, greenhouse, swimming pool, impervious surface, pervious surface, bare soil, water, snow, herbaceous vegetation, agricultural land, plowed land, vineyard, deciduous, coniferous, brushwood | 7 类（LoveDA）：background, building, road, water, barren, forest, agriculture | 6 类：urban, agriculture, rangeland, forest, water, barren |
| ignore（255） | COSIA 15–18（clear cut, ligneous, mixed, undefined）：官方训练中权重为 0，不在 15 类基准内 | 0（no-data）；8（LoveDA 体系外，约 0.2% 像素） | unknown（黑色） |
| 标签编码来源 | 官方 `config_supervision.yaml`（FLAIR-HUB 模型卡） | LoveDA 官方编码 | 官方 `class_dict.csv`；按挑战说明逐通道以 128 二值化 |
| test | 官方 split_1 的 16 个测试省（50,700 张） | 官方 Val（672 张；Test 标签未公开） | 从 803 张有标签影像中随机抽 20%（161 张） |
| val | 从训练/验证省中按**整省**抽约 20%（8 个省，38,425 张） | 从官方 Train 中随机抽 20%（548 张） | 随机抽 10%（80 张） |
| train | 其余 50 个省–年份（151,974 张，剔除 1 张无法解码的影像，见下） | 2,190 张 | 562 张 |

种子 42，默认比例。FLAIR 的 val 省份：D024047、D030、D032、D045、D046、D059062、D067、D077。

需要知道的取舍：

- **FLAIR 剔除 1 张无法解码的训练影像** `D035-2020_AERIAL_RGBI_NA-S1-72_9-2.tif`：
  - 文件只有 552,960 字节，TIFF 的 strip byte count 为 0，读不出像素。官方 zip 的 SHA256 与 Hugging Face
    的记录一致，解压时 CRC 也通过，说明官方发布里就是坏的，重新下载也一样。
  - 2026-10-28 统一细类训练读到它时报错中止，由此发现。随后对五个准备副本的全部 500,230 个影像和标签
    逐一完整解码，只有这一个文件有问题。
  - 它在划分**之后**才剔除（`FLAIR_UNREADABLE_IMAGES`），所以各省份的划分不受影响：train 由 151,975 张
    变为 151,974 张，val 和 test 的列表与逐类像素数都不变。`raw/FLAIR-HUB` 中的原文件保留。
  - 准备工具现在会把每张影像完整解码一次，解不开就报错并放弃生成，不会跳过。
- **FLAIR 的 val 不是官方 val。** FLAIR-HUB 的官方 val 列表（`VALID_FLAIR-INC.csv`）没有随 HF
  数据集发布。这里按整省抽取，规模（38,425 张）接近官方的 38,175 / 190,400。
  - 有两个年份的省（D017、D033、D044、D080）两年始终在同一个划分里，避免同一地区同时出现在 train 和 val。
  - 拿到官方 CSV 后，可以改成按官方列表划分。
- **Global-LoveDA 和 DeepGlobe 的划分无法按空间分组。** 发布包里没有场景或坐标信息，只能按影像随机划分。
  相邻影像可能分到不同划分里，报告时需要注明。
- **Global-LoveDA 的 8 号类**（EarthVQA 中对应 playground）不属于 LoveDA 的 7 类体系，按 ignore 处理。
  它占的像素比例会记录在元数据的 `raw_value_pixels` 里。
- **Global-LoveDA 的 alpha=0 区域**：准备时统计这些区域里被标注的像素（`audit`）。全量数据中有
  1,985,821 个（约 0.06%），分布在官方 Train 的 368 张影像里，官方 Val（即 test）中没有。这些位置的
  RGB 是正常影像（只是偏暗），RGB 模型读图时直接丢弃 alpha 通道，所以保持原样。
- **FLAIR 多波段**：`bands="rgb"` 给 3 通道（R,G,B），`bands="all"` 给 4 通道（加近红外），
  后者需要 4 个值的 `MEAN/STD`（用于预处理假设 H4）。

## OpenEarthMap：城市级划分（方案 B）

数据用 `raw/OpenEarthMap`，即补全了 xBD 影像的 full 版本（97 个城市）。

- 只有官方 train 3,000 + val 500 = 3,500 张有标签，其中 813 张的影像来自 xBD。
- 缺少 xBD 影像的 wo_xBD 版本会被拒绝。
- 官方 test 的 1,500 张没有标签，不使用。

官方 val 是在同一批城市里按图块随机抽的，62% 的 val 图块与某个 train 图块相邻，不能当独立测试集。
所以把 3,500 张按城市重新划分。方案 B 拟定于 2026-10-25，2026-10-28 确认。

| 划分 | 内容 | 城市 | 图块 | 用途 |
|---|---|---|---|---|
| `train` | 训练城市的官方 train 图块 | 71 | 2,124 | 训练 |
| `select` | 选择城市的全部图块 | 11 | 483 | 选 checkpoint |
| `test_city` | 留出城市的全部图块 | 15 | 539 | 最终报告：城市级泛化 |
| `test_indomain` | 训练城市的官方 val 图块，剔除与 train 重叠的 30 张 | 69 | 324 | 最终报告：域内，与训练集同城，不是空间独立的 |
| （不使用） | 训练城市中，面积被官方 train 图块覆盖超过 1% 的 val 图块 | — | 30 | — |

- **类别**：8 类，即 bareland, rangeland, developed space, road, tree, water, agriculture land, building
  （原始值 1–8 → 模型 ID 0–7）。原始值 0（unknown）是 ignore，占有标签像素约 0.65%。
- **留出城市**：mahe, ngaoundere, baybay, chiangmai, tokyo, chisinau, pomorskie, vienna,
  zachodniopomorskie, houston, little_rock, tulsa, adelaide, lima, rosario。
  - 用种子 20261025 从城市簇中抽 20 万个组合，按原生类别、大洲和 GSD 的分布偏差选最优；
    最近图块相距不到 50 km 的城市视为同一簇，整体分到同一组。
  - 所有留出城市离任何训练城市都在 100 km 以上。
- **选择城市**：v3 dev 的 8 个城市（buenos_aires, christchurch, kitsap, paris, pedrogao_grande,
  piura, rio, tuscaloosa），加上 sechura（与 piura 同簇）、khartoum（非洲，Bareland 多）、
  gorakhpur（亚洲，Agriculture 多）。
- **固定城市清单，不是按比例抽取。** `--seed` 和 `--val-fraction` 对 OEM 无效，传入比例会报错。
- **冻结**：
  - 完整生成时，四个划分的数量和列表 SHA256 必须等于 `common/native_datasets.py` 中冻结的值，
    否则拒绝生成；
  - 冻结值由两条独立途径得到并互相核对：一是本工具；二是用 2026-10-25 草案的逐图块元数据重建；
  - `split_id = sha256:904bfb68638123dfd1f21dbb065dede9fa15873411580cb4fb5003a5088de209`。

各划分的原生 8 类像素占比（按有标签像素，不含 unknown）：

| 划分 | Bareland | Rangeland | Developed | Road | Tree | Water | Agriculture | Building | xBD 图块 |
|---|---|---|---|---|---|---|---|---|---|
| train | 1.5% | 23.7% | 15.6% | 6.3% | 20.4% | 3.1% | 13.7% | 15.8% | 21.2% |
| select | 1.2% | 21.2% | 17.0% | 8.1% | 19.3% | 2.6% | 15.3% | 15.3% | 27.5% |
| test_city | 1.7% | 19.3% | 17.3% | 7.2% | 19.9% | 4.9% | 14.3% | 15.4% | 28.6% |
| test_indomain | 1.5% | 22.8% | 16.4% | 6.5% | 20.2% | 2.4% | 13.4% | 16.9% | 17.0% |
| 全部 3,500 张 | 1.5% | 22.6% | 16.1% | 6.7% | 20.2% | 3.3% | 13.9% | 15.8% | 23.2% |

## ISPRS Potsdam：0.05 m，按图块固定划分

数据用 `raw/ISPRS_Potsdam`：38 张 6000² 的真正射影像图块，0.05 m，只有 RGB（`2_Ortho_RGB`）。
标签只有腐蚀了边界的版本（`5_Labels_all_noBoundary`），没有 IRRG、DSM，也没有完整边界的标签。

| 划分 | 图块 | 小块（1000²） | 来源 |
|---|---|---|---|
| `train` | 22 张 | 792 | 其余训练图块 |
| `val` | 2_10 | 36 | BANet / UNetFormer 的验证图块 |
| `test` | 2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13 | 504 | 官方 14 张测试图块 |
| （不使用） | 7_10 | 36 | 标注有误，BANet / UNetFormer / TreeSegNet 都去掉了 |

- **划分沿用 BANet / UNetFormer**（Wang et al.），所以 test 结果能与这一系列论文对照。
  - 它们同样用腐蚀边界的标签评估；
  - mean F1 / mIoU 只算 clutter 以外的 5 类。
  - 这里 clutter 仍作为第 6 类保留，评估时算 5 类还是 6 类由协议决定。
- **类别**：6 类，即 impervious surfaces, building, low vegetation, tree, car, clutter。
  黑色（被腐蚀掉的边界，约 7% 像素）是 ignore，所以训练时边界像素也不参与。
- **标签颜色**：
  - 图块 4_12 的标签约 93% 的像素、6_7 的约 18 万像素不是标准色，如 (2,2,214)、(252,255,0)；
  - 这些颜色离最近的标准色最多差 42（每通道），按 128 逐通道二值化后与最近的标准色完全一致；
  - 修正后这两张的类别比例与其余 36 张相符。和 DeepGlobe 一样用 128 二值化。
- **切块**：每张图块切成 6×6 个不重叠的 1000² 小块，存为无损 PNG（不是硬链接），文件名带行列号，
  如 `top_potsdam_2_13_r0_c1.png`。
  - 同一图块的小块总在同一个划分里；
  - test 图块的每个像素恰好评估一次；
  - 小块尺寸与 OEM 图块相当，训练时照常随机裁剪。
- 划分是固定的图块清单；完整生成时三个划分的数量必须等于冻结值。

## 用法

```bash
cd /data/yfk/land_cover_v4
export LAND_COVER_DATASETS=/data/yfk/datasets
PY=/home/yx/miniconda3/envs/yfk-cv/bin/python

# 只看划分与数量（不写文件）
$PY tools/prepare_native_dataset.py --dataset flair_hub --dry-run

# 正式生成（目标目录必须不存在；生成后只读，改划分就生成新版本目录）
$PY tools/prepare_native_dataset.py --dataset flair_hub \
    --dst $LAND_COVER_DATASETS/prepared/FLAIR-HUB_native_v1 --workers 16
$PY tools/prepare_native_dataset.py --dataset global_loveda \
    --dst $LAND_COVER_DATASETS/prepared/Global-LoveDA_native_v1
$PY tools/prepare_native_dataset.py --dataset deepglobe \
    --dst $LAND_COVER_DATASETS/prepared/DeepGlobe_native_v1
$PY tools/prepare_native_dataset.py --dataset openearthmap \
    --dst $LAND_COVER_DATASETS/prepared/OpenEarthMap_native_v1
$PY tools/prepare_native_dataset.py --dataset isprs_potsdam \
    --dst $LAND_COVER_DATASETS/prepared/ISPRS_Potsdam_native_v1 --workers 16
```

可选参数：`--seed`、`--val-fraction`、`--test-fraction`（仅 DeepGlobe）、`--limit-per-group N`
（每组只取前 N 张，用于冒烟测试，会记入元数据，此时不检查冻结的数量和哈希）。生成过程先写隐藏
临时目录，全部样本成功后才原子改名；中途失败会删除临时目录（里面只有硬链接和新标签，不会动 raw/）。

读取（配置或脚本中）：

```python
from common import native_datasets as nd
values = nd.config_values(root)   # DATA_ROOT、IMAGE_GLOB、LABEL_SUFFIX、IMAGE_BANDS、NUM_CLASSES、
                                  # CLASS_NAMES、IGNORE_INDEX、SPLIT_ID、SELECTION_SPLIT、TEST_SPLITS、
                                  # 每个划分的 <名称>_LIST；VAL_LIST = 选择划分
dataset = OpenEarthMapDataset(root, values["VAL_LIST"], training=False, **nd.dataset_kwargs(root))
```

## 已生成的准备副本（2026-10-28）

都在 `/data/yfk/datasets/prepared/`。除 Potsdam 外，影像均为硬链接。

| 目录 | 划分 |
|---|---|
| `OpenEarthMap_native_v1` | train 2,124 / select 483 / test_city 539 / test_indomain 324 |
| `FLAIR-HUB_native_v1` | train 151,974 / val 38,425 / test 50,700（2026-10-29 剔除 1 张无法解码的影像后重新生成） |
| `Global-LoveDA_native_v1` | train 2,190 / val 548 / test 672 |
| `DeepGlobe_native_v1` | train 562 / val 80 / test 161 |
| `ISPRS_Potsdam_native_v1` | train 792 / val 36 / test 504（1000² 小块） |

生成日志在 `/data/yfk/datasets/archives/native_v1_logs/`。

## 验证

- `tests/test_native_datasets.py`（32 个，其中 3 个需要服务器上的 OEM / Potsdam / FLAIR 副本）用合成小样本覆盖：
  - **标签编码**：未声明的原始值报错；DeepGlobe 和 Potsdam 的 128 二值化（含 4_12 那样的非标准色）；
    OEM 的 unknown 设为 ignore。
  - **Potsdam**：
    - 各图块的角色正确，7_10 被剔除；冻结清单自洽：14 + 1 + 1 + 22 = 38 张，每张 36 块；
    - 小块的影像和标签与原图块的对应窗口逐像素一致。
  - **FLAIR 等三个数据集的划分**：FLAIR 整省划分、多年份省不拆开；官方 test 固定；同一种子结果可复现；
    剔除无法解码的影像后，其余样本的划分不变。
  - **OEM 的划分**：
    - 各角色分配正确；与 train 重叠的 val 图块被剔除，但只剔除训练城市的（gorakhpur 属于选择集）；
    - 冻结清单自洽：15 / 11 个城市互不相交，v3 dev 8 城在选择集中，2,124 + 483 + 539 + 324 + 30 = 3,500；
    - 缺少 xBD 影像时拒绝；
    - 完整生成或读取的副本，列表与冻结值不符时拒绝。
  - **测试集不进入训练**：把 `test_city`、`test_indomain` 当 `VAL_LIST`，或把测试样本混进
    `TRAIN_LIST`，`check_selection_lists()` 都会报错；`tests/test_unified_protocol.py` 另外检查
    `train.validate_training_domain()`。
  - **准备工具**：生成后用 `OpenEarthMapDataset` 读回；影像确实是硬链接；FLAIR 4 波段读取；
    目标已存在、dry-run、列表被改动、中途失败时，都会拒绝或不留下残留；影像解不开时整次生成失败。
- **真实数据**（2026-10-28）：五个副本都通过 `load_metadata()`，各划分之间没有重叠。
  - 随机样本能读出正确形状的张量，标签值都在类别范围内。
  - FLAIR 没有省份跨划分。
  - OEM 副本的四份列表与冻结的哈希一致。
  - Potsdam 抽查的小块与原图块逐像素一致。
- **FLAIR 副本重新生成**（2026-10-29）：train 151,974，剔除的影像不在任何列表中，8 个 val 省份不变；
  val、test 的列表哈希和逐类像素数与重新生成前完全相同，train 的像素数正好少了剔除那张的标签。

## 不包含的数据

官方 LoveDA 没有原生 7 类的读取代码；这里用的是 Global-LoveDA。
