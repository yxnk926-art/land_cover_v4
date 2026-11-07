# 十种模型与 ImageNet 预训练

共 10 种模型结构、11 个配置：SegFormer 分 B0 和 B2 两档。

三个系列按设计目标划分，每个系列内放一对只差一个设计的模型，用来回答“这个设计有没有用”。
2026-11-07 按导师要求加入第四个系列 YOLO，目前只有 YOLOv3。
不同系列之间不直接比排名，而是看精度–计算量的效率前沿。

| 系列 | 模型 | 键（`--models`） | `MODEL_NAME` | ImageNet 权重 | 参数量（19 类输出） | 对照关系 |
|---|---|---|---|---|---|---|
| CNN 编解码 | U-Net（ResNet-50） | `unet_r50` | `unet_resnet` | `resnet50.tv_in1k` | 32.52M | 与 UNet++ 只差跳连拓扑 |
| | UNet++（ResNet-50） | `unetpp_r50` | `unetpp_resnet` | `resnet50.tv_in1k` | 35.69M | 同上 |
| | DeepLabV3+（ResNet-50） | `deeplabv3plus_r50` | `deeplabv3plus` | `resnet50.tv_in1k` | 26.7M | |
| Transformer | SegFormer-B2 | `segformer_b2_in1k` | `segformer` | `mit_b2.nvidia_in1k` | 27.36M | 与 B0 共用优化设置 |
| | SegFormer-B0 | `segformer_b0_in1k` | `segformer` | `mit_b0.nvidia_in1k` | 3.72M | 与 B2 只换规模 |
| | Swin-T + UperNet | `swin_upernet` | `swin_upernet` | `swin_tiny_patch4_window7_224.ms_in1k` | 59.84M | |
| | Segmenter-S/16 | `segmenter_s16_in1k` | `segmenter` | `deit_small_patch16_224.fb_in1k` | 25.98M | 与 SegFormer-B2、Swin-T 优化参数相同，只差结构 |
| 实时轻量 | MobileNetV3-LRASPP | `mobilenetv3_lraspp_in1k` | `mobilenetv3_lraspp` | `mobilenetv3_large_100.ra_in1k` | 3.22M | |
| | PIDNet-S | `pidnet_s_in1k` | `pidnet` | `pidnet_s.imagenet` | 7.72M | 与 DDRNet 只差边界分支和融合模块，优化参数相同 |
| | DDRNet-23-slim | `ddrnet` | `ddrnet` | `ddrnet23_slim.imagenet` | 5.73M | 同上 |
| YOLO | YOLOv3（Darknet-53） | `yolov3_in1k` | `yolov3` | `darknet53.imagenet` | 55.59M | 优化参数与三个 R50 配置相同 |

学习率为暂定值（R50 系列与 YOLOv3 AdamW 1e-4；SegFormer、Swin 和 Segmenter 6e-5、poly 1.0；Swin drop path 0.3；
DDRNet 与 PIDNet 1e-3、weight decay 5e-4；MobileNetV3 用基础配置的 5e-4），待关键参数实验校准。
每个模型的设置都写在各自的 `configs/config_unified_<model>.py` 里。

## 预训练权重

权重不进仓库，默认放在仓库同级的 `pretrained/`（如 `/data/yfk/pretrained`），
可用 `LAND_COVER_PRETRAINED_DIR` 覆盖。下载并校验（服务器需经代理）：

```bash
cd /data/yfk/land_cover_v4
HTTPS_PROXY=http://127.0.0.1:7897 /home/yx/miniconda3/bin/conda run -n yfk-cv \
  python tools/fetch_pretrained.py --all
```

9 个文件已在 `/data/yfk/pretrained/`（来源与 SHA256 另记于该目录的 `MANIFEST.tsv`）。

| 名称 | 来源 | SHA256 前 12 位 |
|---|---|---|
| `resnet50.tv_in1k` | torchvision `IMAGENET1K_V1` | `0676ba61b679` |
| `swin_tiny_patch4_window7_224.ms_in1k` | timm（微软 ImageNet-1k 发布） | `4d7d61fb43a9` |
| `ddrnet23_slim.imagenet` | ydhongHIT/DDRNet README（top-1 err 29.8） | `f6d03fa29b80` |
| `mit_b2.nvidia_in1k` | HF `nvidia/mit-b2`（官方 MiT-B2 的 transformers 版） | `4500b5665471` |
| `mit_b0.nvidia_in1k` | HF `nvidia/mit-b0` | `4af8348c2a80` |
| `mobilenetv3_large_100.ra_in1k` | timm（RandAugment 配方，BN eps 1e-5） | `f2ab36b2846d` |
| `pidnet_s.imagenet` | XuJiacong/PIDNet README 的 Google Drive 文件夹 `PIDNet_Models/ImageNet` | `f96e2c96b1ac` |
| `deit_small_patch16_224.fb_in1k` | timm（Facebook DeiT-S ImageNet-1k 发布） | `52fa3d9d746c` |
| `darknet53.imagenet` | 作者为训练 YOLOv3 提供的 `darknet53.conv.74`（pjreddie.com，darknet 二进制格式） | `2495c2690283` |

训练入口在建立输出目录前校验文件与哈希，并把 `PRETRAINED`、`PRETRAINED_SHA256`
写入 resolved config 与续训指纹；续训时不重复加载。加载是双向严格的：每个保留的
checkpoint 张量必须落到同名同形状的参数上，模型侧除声明为新初始化的前缀（解码头、
Swin 的 `out_norms`、DDRNet 的 DAPPM 与分割头、PIDNet 的 D 分支/PAPPM/融合/分割头）外
必须全部覆盖；分类头和可重算的缓冲（Swin 的 `relative_position_index`/`attn_mask`）显式丢弃。

命名与仓库不同的权重带一个 `convert` 函数，在丢弃分类头之后运行。它把每个剩余的键恰好映射一次，
遇到不认识的键就报错，因此发布格式变了也不会被"加载一半"：

- **MiT（HF → `MixVisionTransformer`）**：逐项改名。HF 的 key/value 是两个独立的 Linear，
  按"key 在前、value 在后"拼成仓库的 `kv`，与官方 MiT 的 reshape 顺序一致。
- **MobileNetV3（timm → `MobileNetV3Backbone`）**：timm 按 stage 分组的 15 个瓶颈块，依次展平到仓库的
  `blocks.0–14`；timm 最后的 1×1 ConvBnAct 对应仓库的 `final_conv`。块内按 expand / depthwise / SE /
  project 的顺序对位，第一个块没有 expand。
- **PIDNet（官方 → `PIDNet`）**：
  - P 分支 `layer{3,4,5}_` 改名为 `layer{3,4,5}_p`；
  - PagFM 的 `f_x` / `f_y`（分别作用于 P 和 I 输入）改名为 `f_p` / `f_i`；
  - 官方 stem 卷积带 bias，仓库的没有（后面紧跟 BN）。把 bias 折进 BN 的 `running_mean`：
    评估模式下完全等价，训练模式下 bias 本来就会被批统计量抵消。
- **DeiT-S（timm → Segmenter 的 `VisionTransformer`）**：键名相同；把 14×14 的位置嵌入双线性重采样到 32×32，
  class token 的嵌入保留，做法与官方 Segmenter 相同。

不是 torch 文件的权重带一个 `reader`（路径 → state dict）：

- **Darknet-53（darknet `.weights` → `Darknet53`）**：文件只有一个头（major、minor、revision、seen）
  和按网络顺序排列的数值，没有键名。每个卷积依次存 BN 的 bias、scale、running mean、running var，
  再存卷积核。`read_darknet53` 按 `darknet53.cfg` 的层序给它们命名；文件长度必须与 52 个卷积恰好相符
  （或再加 1×1 分类层，即 `darknet53.weights`，分类层随后丢弃），否则报错。

## 实现要点与偏离

- **ResNet-50**：参数名与 torchvision 一致；`output_stride` 16/8 用空洞替换步长。
- **U-Net / UNet++（R50）**：解码宽度 (256,128,64,32,16)，最后一级回到全分辨率；
  UNet++ 的 X^{3,1} 与 U-Net 的第 3 级块完全相同，单输出头。
- **Swin-T**：不依赖 timm；任意输入尺寸（块内按窗口补齐并按补齐后尺寸生成平移
  掩码），奇数块始终平移窗口，每个输出阶段加新的 LayerNorm。
- **UperNet**：与 mmseg UPerHead 一致（PPM 1/2/3/6、FPN、512 通道），**唯一偏离**：
  PPM 的 1×1 全局池化分支不用 BN（batch 8 时每通道只有 8 个值，与仓库 DeepLabV3+
  ASPP 的处理一致）。辅助 FCN 头只在训练时输出，权重 0.4。
- **DDRNet-23-slim**：复用 `pidnet.py` 的残差块、DAPPM、分割头；主干参数名与作者
  ImageNet checkpoint 一致。
- `build_optimizer` 支持模型声明 `no_weight_decay_keywords()`；Swin 的 12 个相对
  位置偏置表在 `NO_DECAY_BN_BIAS=True` 时不做 weight decay。
- **MobileNetV3**：ImageNet 权重只有完整尾部的版本，所以 leaf 设 `MOBILENET_REDUCED_TAIL=False` 和
  `MOBILENET_BN_EPS=1e-5`；减半尾部加载权重时直接报错。
- **SegFormer 的 LayerNorm eps**：仓库是 1e-6（与官方 MiT 的块内设置相同），HF 是 1e-5。
  用真实 B2 权重比较，各阶段特征的相对 L2 差 ≤ 1.1×10⁻⁴，所以没有改。

## Segmenter-S/16

`segmenter_s16_in1k`（`common/models/segmenter.py`，共 25.98M）由 DeiT-S 编码器（21.98M）和 2 层掩码
解码器（4.00M）组成，结构和超参数照 [rstrudel/segmenter](https://github.com/rstrudel/segmenter)。

- 编码器全程保持 1/16 的单一尺度，每层都对全部 token 做注意力。
- 解码器让可学习的类别嵌入与 patch token 一起过两层 Transformer，再做点积，得到每类的掩码。
- 输出是普通的逐像素 logits，用交叉熵训练，没有匈牙利匹配，所以集合交叉熵、`evaluate` 和滑窗都不用改。
- 1/16 分辨率下，细长地物和小目标是预期的短板。

与论文的差异：

| 项目 | 论文 / 官方代码 | 本仓库 |
|---|---|---|
| 预训练 | ViT-S 用 ImageNet-21k | DeiT-S 用 ImageNet-1k，与其他配置一致 |
| 优化器 | SGD，lr 1e-3，无权重衰减 | AdamW，lr 6e-5，weight decay 0.01，poly 1.0，与 SegFormer-B2、Swin-T 相同（暂定，待 pilot 确认） |
| 位置嵌入 | 按训练 crop 大小存储 | 固定存 32×32（512 px），其他尺寸在前向时重采样 |
| 掩码点积 | 跟随 AMP | 固定单精度，避免半精度压缩类别间很小的差异 |

校验范围：

- 编码器在共享随机权重上与 timm `deit_small_patch16_224` 数值一致。
- 位置嵌入重采样与 timm `resample_abs_pos_embed`（bilinear，无抗锯齿）一致。
- 解码器与官方代码（`rstrudel/segmenter` master，SHA256 前 12 位：`decoder.py` `0e488eb74fac`、`blocks.py`
  `2c3432141eea`、`utils.py` `4185f00f8b83`）在共享随机权重、评估模式下比对过（2026-10-31，在服务器上手动运行）：
  - 参数名和参数量（3,999,782）完全对应；
  - 解码器输出的最大绝对差约 4e-6，补边、解码、上采样、裁剪的整条流程约 7e-6，属于 float32 舍入误差；
  - 位置嵌入重采样完全相同。
- dropout 是随机的，上面的比对不覆盖，它在训练时的位置只靠读代码确认。

合成数据上的单步训练耗时（batch 8，crop 512，AMP，统一协议 19 类）。GPU 当时与正式训练共用，只看相对差异：

| 模型 | 参数量 | ms/iter | 训练显存峰值 |
|---|---|---|---|
| Segmenter-S/16 | 25.98M | 66.6 | 2.83 GiB |
| SegFormer-B2 | 27.36M | 209.8 | 8.76 GiB |
| Swin-T + UperNet | 59.84M | 336.4 | 12.0 GiB |

## YOLOv3（YOLO 系列）

`yolov3_in1k`（`common/models/yolov3.py`）照 Redmon & Farhadi, *YOLOv3: An Incremental Improvement*
（arXiv:1804.02767, 2018）和官方 `cfg/yolov3.cfg`（pjreddie/darknet）逐层实现。2026-11-07 按导师要求
加入，是 YOLO 系列目前唯一的模型。19 类时 55.59M 参数，512×512 输入 47.02 GMACs。代码与开发仓库
`land_cover_v4dev` 的提交 `e81fe1f` 相同。

**为什么选 YOLOv3。** YOLO 各版都是检测器，原论文都不做语义分割，所以要选一个只改输出层就能做
稠密预测的版本：

- YOLOv1（CVPR 2016）用全连接层输出固定的 7×7 网格（步长 64），输入尺寸不能变，也太粗；
- YOLOv2（CVPR 2017）是全卷积，但只有步长 32 一个输出尺度；
- YOLOv3 第一次带自顶向下的多尺度路径，在步长 8 上输出；v4 以后的版本都沿用"骨干 + 颈部 + 多尺度头"
  的结构。它是不必另设计解码器、能按论文复现的最早版本。
- Ultralytics 2026-05 发布的 YOLO26 有官方语义分割模型（`-sem`），但没有论文，可作为这个系列以后的新成员。

**结构**（括号里是 `yolov3.cfg` 的层号）：

- Darknet-53（0–74）：3×3 stem，5 个 stage，各为步长 2 的 3×3 卷积加 1/2/8/8/4 个残差块
  （1×1 减半、3×3、恒等捷径）。每个卷积后接 BN 和 LeakyReLU(0.1)。参数名与 timm `darknet53` 相同。
- 颈部（75–79、83–91、95–103）：步长 32 上 5 个交替的 1×1/3×3 卷积；1×1 降维、最近邻 2 倍上采样、
  与步长 16 的 stage 输出拼接（route 61），再 5 个卷积；同样再与步长 8 的 stage 输出拼接（route 36），
  再 5 个卷积。
- 输出（104–105）：步长 8 网格上的 3×3 卷积和 1×1 线性卷积。

**改成分割的地方**（与检测版的全部差异）：

| 项目 | YOLOv3 检测 | 本仓库 |
|---|---|---|
| 第 105 层 | 每格 3 个 anchor ×（4 个框参数 + objectness + 80 类）= 255 通道 | 每格 `NUM_CLASSES` 个细类分数（即 YOLOv1 图 2 的 class probability map），双线性上采样到输入尺寸，与 DDRNet、PIDNet 的 1/8 输出相同 |
| 步长 32、16 的输出分支（80–82、92–94） | 预测大、中目标的框 | 去掉。它们只服务于框预测；步长 8 分支已经通过 route 83–86、95–98 拿到两个粗尺度的特征。只用 1/8 层做稠密预测也是 YOLOP（Wu et al., MIR 2022）的做法 |
| 类别损失 | 每类独立的 logistic（BCE） | 集合交叉熵：它需要细类上的概率分布 |
| 输入 | [0,1] 的 RGB，letterbox 补 0.5 灰 | 数据管线给的是 (x−mean)/std，模型在第一个卷积前还原为 [0,1]。边长不是 32 的倍数时在右下补均值色（归一化后为 0），输出再裁回 |
| 优化 | SGD、burn-in、分段衰减 | AdamW 1e-4、weight decay 0.01、poly 0.9，与三个 R50 配置相同（暂定，待 pilot 确认） |

保留的 darknet 细节：LeakyReLU 斜率 0.1；BN eps 1e-5（darknet GPU 版的归一化）、momentum 0.01
（darknet 的滑动平均）；最近邻上采样；新卷积按 darknet 初始化（U(±√(2/fan_in))，bias 为 0）。

**预训练权重**是 `darknet53.conv.74`，即作者在 YOLOv3 训练说明里提供的 ImageNet Darknet-53
（52 个卷积，不含分类层）：

- 它和 ImageNet 分类页上的 `darknet53.weights`（论文表 2：top-1 77.2%）不是同一次训练：两者卷积核的
  相关系数约为 0；`darknet53_448.weights` 则由 `darknet53.weights` 微调而来（相关 0.999）。
- 官方 COCO 权重 `yolov3.weights` 的骨干卷积核与 `conv.74` 相关（总体 0.35，stem 0.62），与
  `darknet53.weights` 约为 0。可见官方 YOLOv3 是从 `conv.74` 开始训练的，本仓库也用它。

**校验**（脚本和日志在服务器 `/home/yx/yolo_verify_20261107/`，不进仓库）：

- 骨干与 timm `darknet53` 在共享随机权重上逐级完全一致。timm 的 LeakyReLU 斜率是 0.01，比对时改成
  0.1；timm 自己训练的 `darknet53.c2ns_in1k` 用的是 0.01，与论文不符，所以没有采用。
- 读取器：用同一个读取器载入作者的 `darknet53.weights`（52 个卷积 + 分类层），darknet 自带的
  `data/eagle.jpg` 判为 bald eagle（0.52），`data/dog.jpg` 判为 malamute（0.87）。
- 颈部接线：在同一组模块旁补上检测专用的 80–81、92–93 层和 255 通道的第 105 层，用读取器载入官方
  COCO 权重 `yolov3.weights`（75 个卷积，62,001,757 个数值恰好用完），检测 `data/dog.jpg`：

  | 输入 | dog | bicycle | truck |
  |---|---|---|---|
  | 416×416 | 99% | 99% | 91% |
  | 608×608 | 100% | 99% | 93% |
  | 作者网页 | 99% | 99% | 93% |

- 参数量：`yolov3.cfg` 共 61,949,149 个参数，去掉的 80–81、92–93 层和 255 通道的第 105 层占
  6,359,037 个；本仓库第 105 层每类 257 个参数，19 类时共 55,594,995 个。
- 本仓库与开发仓库的 YOLOv3 统一协议配置有相同的协议指纹，开发仓库训练出的 checkpoint 可以在这里评估。

计算量与速度（512×512，19 类，只计 Conv2d/Linear；同一进程里先后测，GPU 当时空闲）：

| 模型 | 参数量 | MACs | 训练 ms/iter（batch 8，AMP） | 训练显存峰值 |
|---|---|---|---|---|
| YOLOv3 | 55.59M | 47.02 G | 47 | 4.6 GiB |
| DeepLabV3+ R50 | 26.70M | 37.68 G | 50 | 3.9 GiB |

开发仓库里的冒烟（只用选择划分，按正式的学习率曲线跑前几千次迭代，不是成绩）：统一协议 4k 次迭代时
五个数据集的均值 0.482；原生协议 Global-LoveDA 2k 时 0.655（不加载 ImageNet 权重时 0.543），
Potsdam 10k 时 0.718。正式训练 `unified_v2_s42` 于 2026-11-07 在开发仓库启动。

## 验证

- `tests/test_backbones.py`：ResNet-50（含 OS16 空洞）与 torchvision、Swin-T 与
  timm 在相同随机权重下逐阶段数值一致（448×448）；Darknet-53 与 timm `darknet53`（斜率改为 0.1）一致。
- `tests/test_models.py`：各模型任意尺寸输出、训练/评估输出个数、loss 反传、
  参数量（DDRNet 5.73M，与论文 5.7M 一致）、Swin 偏置表的 no-decay 分组；YOLOv3 的层数与参数量
  对得上 `yolov3.cfg`，骨干看到的是 [0,1] 的 RGB，补边裁回后与手动补边逐位相同。
- `tests/test_pretrained_loading.py`：哈希不符、文件缺失、架构不符、缺/多张量均被
  拒绝；本机已有权重时，12 个模型–权重组合全部完整覆盖。
- `tests/test_pretrained_conversion.py`：
  - **MobileNetV3**：timm 随机权重转换后，1/4、1/8 和 960 通道特征与 timm 逐层一致；
  - **MiT**：装有 transformers 时，转换后各阶段输出与 HF `SegformerModel` 一致。
    服务器环境没有装 transformers，曾用临时安装的 4.46.3 跑通一次；
  - **PIDNet**：stem 的 bias 折叠在训练和评估两种模式下都完全等价；用真实权重加载后，
    只有声明为新初始化的模块保持随机值；
  - **darknet 读取器**：随机权重按 darknet 格式写出再读回逐位相同（文件大小与发布的 `conv.74` 一致）；
    旧版 32 位 `seen` 的文件头可读；带分类层的文件读出后丢弃分类层；长度不符时报错；
  - 减半尾部、宽度不符、出现不认识的键时，都会报错。
- `tests/test_unified_protocol.py`：全部 leaf 都写明已登记的权重，输出目录不冲突，对照组的优化参数一致。
