# Changelog

## 2026-11-07：加入 YOLOv3（YOLO 系列）

导师要求在九种模型之外加入 YOLO 系列，从能复现成分割模型的最基础 YOLO 论文开始。YOLOv1 用全连接层输出固定的
7×7 网格，YOLOv2 只有步长 32 的输出，都要另设计解码器；YOLOv3（Redmon & Farhadi, arXiv:1804.02767）是第一个
在步长 8 上输出、带多尺度路径的版本。模型与读取器代码和开发仓库 `land_cover_v4dev` 的提交 `e81fe1f` 相同。

- 新模型 `yolov3`（`common/models/yolov3.py`）和配置 `config_unified_yolov3_in1k`（键 `yolov3_in1k`，55.59M），
  权重 `darknet53.imagenet`。现在是 10 种模型、11 个配置。按官方 `cfg/yolov3.cfg` 逐层实现 Darknet-53 和颈部；
  步长 8 的输出层每格输出细类分数，代替每个 anchor 的框、objectness 和类别；只预测框的步长 32、16 输出分支去掉；
  logits 双线性上采样，用集合交叉熵训练。保留 darknet 的 LeakyReLU 0.1、BN eps 1e-5 和 momentum 0.01、
  最近邻上采样和新层初始化。darknet 权重要求 [0,1] 的 RGB，模型在第一个卷积前还原数据归一化。
- `common/pretrained.py`：`PretrainedSpec.reader` 用于非 torch 格式的权重；`read_darknet_weights` /
  `read_darknet53` 读取 darknet 的 `.weights`（按层序存数值、没有键名，长度不符就报错）。`darknet53.imagenet`
  是 `darknet53.conv.74`，作者为训练 YOLOv3 提供的 ImageNet Darknet-53（SHA256 2495c269…），
  与 `darknet53.weights` 不是同一次训练，官方 COCO 权重由它而来。
- 它进入默认队列，排在 DeepLabV3+ 之前（单步训练 47 对 50 ms）。优化参数与三个 R50 配置相同
  （AdamW 1e-4，weight decay 0.01，poly 0.9），是暂定值。没有新增配置键，其他 10 个配置的指纹不变。

验证结果：

- 测试：205 个通过，1 个跳过（未安装 transformers 时 MiT 与 HF 的数值对照），其中 14 个是新增的；10 个已有配置的协议指纹与上一次提交 `72c0920` 相同。
- 用官方 COCO 权重装进同一组模块检测 darknet 的 `data/dog.jpg`：416 输入 dog 99%、bicycle 99%、truck 91%，
  608 输入 100%、99%、93%；作者网页给出 99%、99%、93%。骨干与 timm `darknet53`（斜率改为 0.1）逐位一致；
  作者的 ImageNet 分类权重经新读取器把 `data/eagle.jpg` 判为 bald eagle。
- 本仓库与开发仓库的 YOLOv3 配置协议指纹相同，开发仓库 `unified_v2_s42` 训练出的 checkpoint 可以在这里评估。
- 开发仓库里的冒烟（选择划分，不读 test）：统一协议 4k 次迭代时均值 0.482。这不是成绩。
- 没有验证：正式训练 2026-11-07 在开发仓库启动，尚未结束；评估工具没有在真实 test 上运行。

## 2026-11-02：PIDNet 在 Potsdam 上的边界目标和按数据集平均的边界感知损失

对应审查项 C1 和 P6。只有 PIDNet 的配置变化，其他配置的指纹不变。已记录的 PIDNet 分数都是修复前的，PIDNet 还没有重训。

- `make_boundary_target(..., bridge_radius)`：把隔开两个类别的忽略带两侧也标为边界。Potsdam 的腐蚀标签里所有类别之间都隔着这样的带（宽度中位数 6 像素），所以它的边界目标一直是空的，真实边界附近的像素还被当成负样本。配置里 `BOUNDARY_BRIDGE_RADIUS = 12`。
- `BOUNDARY_AWARE_PER_DATASET = True`：边界感知损失按数据集平均，避免边界像素多的数据集占主导。
- 新增 `SetCrossEntropyLoss.pixel_losses`。不设新键时损失不变。

每个数据集 160 个真实训练裁剪的结果：Potsdam 边界正样本从 0 变为 5.35%，没有正样本的裁剪从 160 个变为 0 个；OEM +0.01%、FLAIR +1.99%、Global-LoveDA +0.16%、DeepGlobe +1.03%（相对变化）。半径到 10–12 基本饱和。

已验证：191 个测试通过（3 个跳过），其中 16 个是新增的；20 步真实数据训练在 GPU 上数值正常。完整的 1000 次迭代冒烟在开发仓库上运行，`common/engine.py` 与这里相同。
未验证：是否能恢复 PIDNet 在 Potsdam 上的分数，这要重训。

## 2026-11-01: 中断恢复、运行记录和映射勘误

来自 2026-11-01 审查中值得现在做的部分。配方和细类文件都没有改，已跑完模型的续训指纹不变。

- `common/train.py` 和 `tools/run_baselines.py` 加 `--resume`：有 `checkpoint_last.pth` 的接着训练（指纹一致才放行），已跑完的立即结束，没有 checkpoint 的从头开始。checkpoint 改为原子写入。
- `resolved_config.json` 和 checkpoint 记录 `code`：commit、未提交的已跟踪文件、`common/`、`configs/`、`tools/` 的文件哈希；日志每次启动写一行 `[*] code:`；续训时代码不同会警告。
- 新增 `train_history.jsonl`（每个 epoch，含逐数据集 loss）和 `val_history.jsonl`（完整分数和两种混淆矩阵）。
- `tools/evaluate_unified.py` 比较完整协议快照；评估 `checkpoint_last.pth` 要加 `--allow-last`；新增 `--selection`、`--per-image`。
- `common/unified_taxonomy.py` 新增 `identifiability`，配置快照里有 `UNIFIED_IDENTIFIABILITY`（秩，以及一组数据集分不开的细类）。
- 新增 `docs/UNIFIED_FINE_V2_ERRATA.md`，逐格给出映射依据。58 个 `official` 格子里，38 个官方原文明写，13 个只是类名相同，5 个部分成立，2 个与原文矛盾。
- SegFormer-B2 的文档串写明官方解码头有 `lr_mult=10`，这里没有。

试跑和重算（在开发仓库，只用选择划分，不读 test，单种子）：

- SegFormer-B0 解码头 6e-4、编码器 6e-5：2 万次 0.6122，4 万次 0.6315；当前配方 0.6037、0.6278。4 万次差 0.004，低于 0.02 的门槛且在噪声范围内，配方不改。
- SegFormer-B0 单独训练 FLAIR：4 万次 0.6057；8 万次 0.6250（best 0.6262）。统一训练在相同训练量下 FLAIR 是 0.6050。
- 5 个已完成模型的逐图混淆矩阵重算出的 25 个数据集分数，与记录值相差不超过 1e-4。没有强边缘匹配的选择图块，分数高于全部图块（Global-LoveDA 高 0.011–0.016，DeepGlobe 高 0.038–0.083），与"相邻抬高分数"的预期相反；但判据只匹配有纹理的边缘，两个子集不能直接比，这个问题还没有结论。

验证结果：

- 测试：175 个通过，1 个跳过。
- 开发仓库中 36 个配置的续训指纹与 `d3e44f2` 相同；DDRNet、PIDNet 的短跑在新旧代码上损失、分数和权重都相同，本仓库也相同；`unified_v2_s42` 的全部 checkpoint 通过新的评估校验。
- 在开发仓库的 GPU 上用 SIGKILL 中断再续训，最终权重与不中断的运行相同（DDRNet、SegFormer-B0，Potsdam 确定性短跑）。
- 没有验证：完整长度、非确定性运行的续训；评估工具在真实 test 上运行。

没有做：审查的 P4、P5、P6（P6 要和 C1 一起改）。

开发仓库的 `unified_v2_s42` 当天补入 Segmenter-S/16（同一 run-tag 和种子），优化器仍是暂定值。

## 2026-10-31: 加入 Segmenter-S/16

- 新模型 `segmenter`（`common/models/segmenter.py`）和配置 `config_unified_segmenter_s16_in1k`（键
  `segmenter_s16_in1k`，25.98M），权重 `deit_small_patch16_224.fb_in1k`。现在是 9 种模型、10 个配置。
- 它进入默认队列，排在 SegFormer-B0 之后。这个位置是估的：只测过它比 SegFormer-B2 和 Swin-T 快，没有和 B0 比过。
- 它不在开发仓库正在跑的 `unified_v2_s42` 里，要和那轮结果对照，需要单独训练。
- 优化器是暂定值，与论文不同：AdamW，lr 6e-5，weight decay 0.01，poly 1.0，与 SegFormer-B2、Swin-T 相同。
- `SEGMENTER_VARIANT`、`SEGMENTER_DECODER_LAYERS` 进入续训指纹。其他配置没有这两个键，指纹不变。

验证结果：

- 测试：156 个通过，1 个跳过（未安装 transformers 时 MiT 与 HF 的数值对照）。
- 与开发工作区 `land_cover_v4dev_segmenter` 比较：`segmenter.py` 逐字节相同；10 个配置的配置快照（除路径类的键）、
  续训指纹、相同种子下的初始权重、前向输出、损失和优化器分组都相同。其他 9 个配置的指纹与上一次提交相同。
- 真实数据冒烟（五个数据集、100 次迭代，选择划分各取 12 张，不读 test）：权重载入 150 个张量，AMP 训练和含滑窗的
  验证都正常，best checkpoint 通过评估工具的校验并能严格重新载入。这不是成绩。
- 官方代码（`rstrudel/segmenter` master）：在共享随机权重、评估模式下，参数名和参数量完全对应，解码器输出差约 4e-6，
  整条流程差约 7e-6，位置嵌入重采样完全相同。dropout 的位置只靠读代码确认。
- 没有验证：没有做过 pilot 或正式训练；评估工具没有在真实 test 上运行。

## 2026-10-31: 从开发仓库精简而来

- 来源：`land_cover_v4dev`，分支 `v4-dev`，提交 `d912606`。
- 只保留统一细类协议、8 种模型（9 个配置，SegFormer 分 B0、B2）和 5 个数据集。去掉了 v3 协议、原生类别协议、HPO 和细类 v1。
- 未改动的代码：`common/engine.py`、`common/native_datasets.py`、`common/unified_taxonomy.py`、
  `common/early_stopping.py`、`common/numerical_diagnostics.py`、细类文件、`tools/prepare_native_dataset.py`、
  `tools/fetch_pretrained.py`，以及 `swin.py`、`mobilenetv3.py`、`deeplabv3plus.py` 以外的模型文件
  （这三个只改了注释）。
- 改动：
  - `config_v4_recipe.py` 并入 `config_unified_v4.py`，每个 leaf 直接写明自己的模型和优化参数；
  - `common/train.py` 去掉其他协议的分支，并且不再在 import 时解析命令行和读配置；
  - `tools/evaluate_unified.py` 内含原来放在 `tools/evaluate.py`、`tools/evaluate_native.py` 的辅助函数；
  - `tools/run_baselines.py` 只跑统一协议，`--run-tag` 必填，默认队列是全部配置，由快到慢。
- 配置键、续训指纹、权重和 checkpoint 格式都与开发仓库相同，开发仓库训练出的 checkpoint 可以在这里评估。

验证结果：

- 测试：144 个通过，1 个跳过（未安装 transformers 时 MiT 与 HF 的数值对照）。
- 配置（9 个 leaf，与提交 `d912606` 比较）：除路径类的键外，配置快照完全相同；续训指纹、相同种子下的
  初始权重、前向输出、损失和优化器分组逐一相同。
- 短训练（真实数据、相同种子、确定性模式、24 次迭代）：
  - DDRNet 和 PIDNet-S：每次迭代的 loss、最终权重和选择划分分数都完全相同；
  - Swin-T + UperNet：两个仓库的结果不完全相同，但开发仓库自己重复运行三次也互不相同。PyTorch 提示
    `adaptive_avg_pool2d` 的反向传播和 memory-efficient attention 没有确定性实现。两个仓库的差异没有超出
    这个范围（选择划分分数 0.08562–0.08575），所以这项只能说明没有发现异常，不能算逐位相同。
- 正式 checkpoint：开发仓库 `unified_v2_s42` 的 DDRNet `best_model.pth` 通过本仓库评估工具的全部校验，严格载入，
  续训指纹相同（`011acf49ffd5`）。它在 Potsdam、DeepGlobe、Global-LoveDA 选择划分上的 mIoU 与记录值相差不超过
  2.2e-5。
- 命令行：`tools/run_baselines.py` 在真实 Potsdam 数据上启动 DDRNet，加载权重并训练到 1000 次迭代以上，
  中断后没有残留进程。`tools/prepare_native_dataset.py --dry-run` 和 `common.unified_taxonomy` 也都运行过。
- 没有运行：`tools/evaluate_unified.py` 在真实数据上的完整流程。它只读 test 划分，为保持 test 封存只用合成数据测试。
