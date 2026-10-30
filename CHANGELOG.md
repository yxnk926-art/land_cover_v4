# Changelog

## 2026-10-31: 从开发仓库精简而来

- 来源：`land_cover_v4dev`，分支 `v4-dev`，提交 `d912606`。
- 只保留统一细类协议、9 个模型和 5 个数据集。去掉了 v3 协议、原生类别协议、HPO 和细类 v1。
- 未改动的代码：`common/engine.py`、`common/native_datasets.py`、`common/unified_taxonomy.py`、
  `common/early_stopping.py`、`common/numerical_diagnostics.py`、细类文件、`tools/prepare_native_dataset.py`、
  `tools/fetch_pretrained.py`，以及 `swin.py`、`mobilenetv3.py`、`deeplabv3plus.py` 以外的模型文件
  （这三个只改了注释）。
- 改动：
  - `config_v4_recipe.py` 并入 `config_unified_v4.py`，每个 leaf 直接写明自己的模型和优化参数；
  - `common/train.py` 去掉其他协议的分支，并且不再在 import 时解析命令行和读配置；
  - `tools/evaluate_unified.py` 内含原来放在 `tools/evaluate.py`、`tools/evaluate_native.py` 的辅助函数；
  - `tools/run_baselines.py` 只跑统一协议，`--run-tag` 必填，默认队列是全部 9 个模型，由快到慢。
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
