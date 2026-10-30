# land_cover_v4

遥感土地覆盖语义分割的正式仓库：**统一细类协议**（创新点 A）下，9 种模型（SegFormer 分 B0、B2 两档，共 10 个配置）在 5 个数据集上联合训练。

一个模型同时用五个数据集训练，输出 19 个统一细类。每个数据集仍读自己的原生标签，真值只要求概率落在对应的细类集合里
（集合交叉熵）。评估时把细类概率按集合求和，再按各数据集的原生类别计算 mIoU。

本仓库由开发仓库 `land_cover_v4dev`（分支 `v4-dev`，提交 `d912606`）精简而来，只保留这一套协议：没有 v3 协议、
没有原生类别协议，也没有 HPO。开发和实验性的内容继续放在 `land_cover_v4dev`。

## 模型

| 系列 | 模型 | 键（`--models`） | 参数量（19 类输出） |
|---|---|---|---|
| CNN 编解码 | U-Net（ResNet-50） | `unet_r50` | 32.52M |
| | UNet++（ResNet-50） | `unetpp_r50` | 35.69M |
| | DeepLabV3+（ResNet-50） | `deeplabv3plus_r50` | 26.7M |
| Transformer | SegFormer-B2 | `segformer_b2_in1k` | 27.36M |
| | SegFormer-B0 | `segformer_b0_in1k` | 3.72M |
| | Swin-T + UperNet | `swin_upernet` | 59.84M |
| | Segmenter-S/16 | `segmenter_s16_in1k` | 25.98M |
| 实时轻量 | MobileNetV3-LRASPP | `mobilenetv3_lraspp_in1k` | 3.22M |
| | PIDNet-S | `pidnet_s_in1k` | 7.72M |
| | DDRNet-23-slim | `ddrnet` | 5.73M |

全部从 ImageNet 权重开始训练。权重来源、键名转换和实现偏离见 [docs/MODELS.md](docs/MODELS.md)。

## 数据集

数据不在仓库里，统一放在 `/data/yfk/datasets/prepared/`，生成后只读。

| 数据集 | 准备副本 | 原生类别 | 模型选择 | test |
|---|---|---|---|---|
| OpenEarthMap | `OpenEarthMap_native_v1` | 8 | `select`（11 个城市，483 张） | `test_city`（留出 15 个城市，539 张）、`test_indomain`（324 张） |
| FLAIR-HUB | `FLAIR-HUB_native_v1` | 15 | `val`（8 个省，38,425 张） | 官方 split_1 的 16 个测试省（50,700 张） |
| Global-LoveDA | `Global-LoveDA_native_v1` | 7 | `val`（548 张） | 官方 Val（672 张） |
| DeepGlobe Land Cover | `DeepGlobe_native_v1` | 6 | `val`（80 张） | 随机切 20%（161 张） |
| ISPRS Potsdam | `ISPRS_Potsdam_native_v1` | 6 | `val`（图块 2_10，36 小块） | 官方 14 张测试图块（504 小块） |

类别、标签编码、划分方案和准备副本格式见 [docs/DATASETS.md](docs/DATASETS.md)。

## 协议

| 项目 | 取值 |
|---|---|
| 细类 | `configs/class_maps/unified_fine_v2.json`，19 类，`status: final` |
| 训练 | crop 512、batch 8、AdamW、AMP；20 万次迭代（200 个虚拟 epoch，每个 1000 次），每个 epoch 每个数据集 1600 个样本 |
| 验证 | 每 2 万次迭代一次，五个选择划分全部评估，FP32 |
| 选模 | 五个数据集原生 mIoU 的平均值 |
| test | 只由 `tools/evaluate_unified.py` 读取，训练时不读 |

细节、映射表和已知局限见 [docs/UNIFIED_FINE_V2.md](docs/UNIFIED_FINE_V2.md)。

## 仓库结构

```text
common/
  train.py              训练入口
  engine.py             损失（含集合交叉熵）、优化器、评估、滑窗
  dataset.py            数据集类、采样器、统一协议的 DataLoader
  native_datasets.py    五个数据集的类别、标签编码、划分与准备副本元数据
  unified_taxonomy.py   读入并检查细类文件
  unified_protocol.py   模型登记表、数据集条目、标签偏移、联合验证
  pretrained.py         权重登记表与严格加载
  models/               9 种模型的工厂与实现
configs/
  config_unified_v4.py  基础配置
  config_unified_*.py   每个配置一个 leaf
  class_maps/           细类文件
tools/
  run_baselines.py      按顺序批量训练
  evaluate_unified.py   test 划分上的最终评估
  prepare_native_dataset.py  生成准备副本
  fetch_pretrained.py   下载并校验权重
tests/                  单元测试
docs/                   详细文档
```

## 环境

| 项目 | 位置 |
|---|---|
| Python 环境 | conda 环境 `yfk-cv`（PyTorch 2.5.1+cu121、timm 1.0.28） |
| 数据 | `$LAND_COVER_DATASETS`，默认仓库同级的 `datasets/`，即 `/data/yfk/datasets` |
| 预训练权重 | `$LAND_COVER_PRETRAINED_DIR`，默认仓库同级的 `pretrained/`，即 `/data/yfk/pretrained` |

## 常用命令

```bash
cd /data/yfk/land_cover_v4
PY=/home/yx/miniconda3/envs/yfk-cv/bin/python

# 全部测试
$PY -m unittest discover -s tests -t .

# 校验预训练权重：已存在且哈希一致就跳过，缺的才下载（下载要走代理）
HTTPS_PROXY=http://127.0.0.1:7897 $PY tools/fetch_pretrained.py --all

# 检查细类文件并打印映射表
$PY -m common.unified_taxonomy

# 看批量训练计划，不写文件
$PY tools/run_baselines.py --run-tag <tag> --seed 42 --dry-run

# 训练：默认全部 10 个配置，由快到慢；一个失败就停止后面的队列
$PY tools/run_baselines.py --run-tag <tag> --seed 42
$PY tools/run_baselines.py --run-tag <tag> --seed 42 --models ddrnet unet_r50

# 训练单个模型
$PY common/train.py --config config_unified_ddrnet --run-tag <tag> --seed 42

# test 划分上的最终评估
$PY tools/evaluate_unified.py --model ddrnet \
  --checkpoint checkpoints/unified_v4/all/baselines/<tag>/ddrnet23_slim/best_model.pth
```

训练产物在 `outputs/`、`checkpoints/`，评估结果在 `evaluations/`，都不进 Git。`--run-tag` 不允许覆盖已有结果。
