# 统一细类 v2 与联合训练协议

> 状态：细类体系 v2 已于 2026-10-30 定稿（`configs/class_maps/unified_fine_v2.json`，`status: final`），取代 v1；
> 9 个配置的正式训练 `unified_v2_s42` 于同日在开发仓库 `land_cover_v4dev` 启动，后来加入的 Segmenter-S/16 于 2026-11-01 以同一 run-tag 和种子补入这一轮。细类文件为 `final` 时才允许
> `--run-tag` 正式训练和批量启动器，草案只能冒烟、试跑和 dry-run。

## 做法

1. **一个模型，五个数据集一起训练。** 模型输出 19 个统一细类。
2. **标签不改。** 每个数据集仍读自己的原生准备副本。数据集 d 的原生类 c 是一组细类 S_d(c)：
   - 真值为 c 的像素，损失为 `-log Σ_{u∈S_d(c)} p_u`（集合交叉熵），只要求概率落在这组细类里；
   - 训练时给每个数据集的标签加一个偏移量，使一张标签图同时表示“哪个数据集的哪个原生类”。
3. **评估永远按各数据集的原生类别。**
   - 每个原生类的得分是它那组细类的概率之和，取得分最高的原生类作为预测；
   - 再按原生口径计算 mIoU，所以结果可以和各数据集单独训练的基线直接比较。
4. **选模**：五个数据集选择划分上原生 mIoU 的平均值。

细类只是模型内部的输出层，不是评估口径。映射如果定错，会直接表现为某个数据集的原生 mIoU 下降。

## 细类的定义原则

- **按各数据集的官方定义拆分。** 只要有一个数据集把两类地物分进不同的原生类，就拆成两个细类。
  - 例如 FLAIR 把自然裸地算 bare soil，把采石场和填埋场算 pervious surface，所以 `bare_natural` 和 `extraction_dump` 分开；
  - 又如 OEM 把人工草坪（artificial grass areas）算 developed space，天然草皮球场归这里是推断；FLAIR 算 herbaceous vegetation，所以单设 `sports_grass`。
- **每两个细类至少要有一个数据集能区分，否则合并。** 差别只涉及极少像素的细类也合并（见下一节）。
- **一个细类在每个数据集里只属于一个原生类。** 集合交叉熵下，同属两个原生类的细类在这些像素上不受罚，
  会被模型当成万能类（见下一节）。
  - `tests/test_unified_protocol.py` 检查现行细类文件没有重叠；`load_taxonomy()` 本身不拒绝重叠，v1 的文件留在开发仓库。
- **同一数据集中，任何一个原生类的细类集合都不能包含另一个原生类的全部细类。** 否则按集合概率评估时，
  后者永远不会被预测。
- `common/unified_taxonomy.py` 的 `load_taxonomy()` 在每次读入时自动检查第 2、4 条，以及类名与
  `common/native_datasets.py` 是否一致。

## 从 v1 到 v2

**v1 的缺陷。** v1（23 个细类）里 `car` 在 OEM 同属 road 和 developed space，在 Global-LoveDA 同属 road 和 background。
对 `unified_s42` 跑完的 6 个模型的审计（只用选择划分）显示：

- 两个数据集的路面像素有 92–95% 落到 `car`，`road` 细类的使用率是 0.0%，road IoU 都约等于 0；
- 评估时把 `car` 改成只归 road 救不回来：OEM road 升到 0.27，但 developed space 跌到 0.002，mIoU 反而下降。所以必须改训练。

`unified_s42` 在 SegFormer-B2 约 3 万次迭代时停止（UNet++、Swin 没有跑），结果留在
开发仓库的 `outputs|checkpoints/unified_v4/all/baselines/unified_s42/`。

**改动。**

| 改动 | 依据 |
|---|---|
| OEM、Global-LoveDA 的 `car` 只归 road | 取消重叠。停车场上的车在这两个数据集分别标成 developed space、background，现在算少量标签噪声 |
| `track_railway` 并入 `road` | FLAIR 的铁路和砾石路仍是 pervious surface，这些像素改由 `construction_site` 承接 |
| `sports_hard` 并入 `paved_open` | 只有 Potsdam 把硬质运动场算 clutter，而它在 Potsdam 中很少见 |
| `pasture`、`orchard` 并入 `farmland` | FLAIR 的 agricultural land 明确含农用草地；各数据集只有个别格子把它们分开，且多数待核对 |

用 v1 checkpoint 按上面的合并重投影，各模型各数据集的 mIoU 变化都不超过 0.003，
唯一例外是 SegFormer-B0 的 Potsdam（0.766→0.685，这个 checkpoint 依赖 `orchard` 细类）。

**未采纳。**
- 合并 `extraction_dump` 和 `bare_natural`：重投影时 FLAIR pervious surface 0.488→0.262，bare soil 0.281→0.124。
- DeepGlobe 的 road、car 进全部 6 类、FLAIR 的 car 同属 impervious 和 pervious、FLAIR 的 farmland 含 herbaceous：都是新的重叠。
- 单设 `clear_cut`（占 FLAIR 0.60%，需要新的标签副本和评测屏蔽）：暂缓。

## 试跑（2026-10-30）

DDRNet-23-slim，种子和学习率曲线与 v1 相同，只跑 20 万次中的前 4 万次，全量选择划分，test 未碰。
试跑用的候选文件（sha256 `ed987c03…`）与定稿的 v2 只差 `taxonomy_id`、`status`、`description`，细类映射完全相同。

| 指标（v1 → v2） | 2 万次 | 4 万次 |
|---|---|---|
| 五数据集均值 | 0.517 → 0.544 | 0.549 → 0.575 |
| OEM mIoU | 0.396 → 0.483 | 0.404 → 0.481 |
| OEM road | 0.000 → 0.506 | 0.000 → 0.499 |
| Global-LoveDA mIoU | 0.570 → 0.620 | 0.597 → 0.668 |
| Global-LoveDA road | 0.000 → 0.398 | 0.000 → 0.435 |
| FLAIR mIoU | 0.401 → 0.405 | 0.433 → 0.436 |
| DeepGlobe mIoU | 0.571 → 0.560 | 0.616 → 0.621 |
| Potsdam mIoU | 0.646 → 0.652 | 0.695 → 0.670 |

- 4 万次时 OEM 路面像素 70% 落到 `road`，Global-LoveDA 65%；`car` 在 OEM、FLAIR、Global-LoveDA、DeepGlobe 都低于 0.3%，
  只在 Potsdam 用于真实车辆（Potsdam car IoU 0.689）。
- 其他类没有被拖累：4 万次时 OEM 其余 7 类均值 0.462→0.479，Global-LoveDA 其余 6 类 0.696→0.707。
- **待观察**：Potsdam 在 4 万次时 impervious 0.718→0.650、tree 0.604→0.562，降幅超过 v1 相邻验证的典型波动（约 0.02），
  但 2 万次时 impervious 还高 0.039，而且验证集只有一张图（2_10）。要看完整训练才能判断。
- 证据范围：一个模型、一个种子、前 4 万次；单数据集基线尚未训练，还不能比较。

## 映射表

无标记 = JSON 中 `basis` 为 `official`，其中不少只是类名相同或部分成立，逐格结论见[勘误表](UNIFIED_FINE_V2_ERRATA.md)；† = 由官方定义推断；? = 待核对（备选见下一节）。依据见 JSON 的 `sources`
和每个格子的 `note`。下表由 `python -m common.unified_taxonomy` 生成。

| # | 细类 | OEM | FLAIR-HUB | Global-LoveDA | DeepGlobe | Potsdam |
|---|---|---|---|---|---|---|
| 0 | `building` | building | building | building | urban | building |
| 1 | `greenhouse` | building ? | greenhouse | background ? | agriculture † | building ? |
| 2 | `road` | road | impervious surface | road | urban | impervious surfaces |
| 3 | `car` | road † | impervious surface | road † | urban | car |
| 4 | `paved_open` | developed space | impervious surface | background † | urban | impervious surfaces |
| 5 | `sports_grass` | developed space † | herbaceous vegetation | background † | urban ? | low vegetation † |
| 6 | `construction_site` | developed space | pervious surface | background ? | urban † | clutter ? |
| 7 | `extraction_dump` | bareland † | pervious surface | barren ? | barren † | clutter † |
| 8 | `bare_natural` | bareland | bare soil | barren † | barren | clutter ? |
| 9 | `snow_ice` | bareland † | snow | barren † | barren † | clutter † |
| 10 | `water` | water | water | water | water | clutter |
| 11 | `swimming_pool` | water | swimming pool | water ? | urban † | clutter |
| 12 | `herbaceous` | rangeland | herbaceous vegetation | background ? | rangeland | low vegetation |
| 13 | `farmland` | agriculture land | agricultural land | agriculture | agriculture | low vegetation † |
| 14 | `plowed_land` | agriculture land † | plowed land | agriculture † | agriculture † | low vegetation ? |
| 15 | `vineyard` | agriculture land | vineyard | agriculture † | agriculture | low vegetation † |
| 16 | `shrub` | rangeland | brushwood | forest ? | rangeland † | low vegetation ? |
| 17 | `deciduous_tree` | tree | deciduous | forest | forest | tree |
| 18 | `coniferous_tree` | tree | coniferous | forest | forest | tree |

细类定义（英文原文见 JSON 的 `description`）：

| 细类 | 含义 |
|---|---|
| `building` | 有顶建筑：住宅、商业、工业建筑，塔、筒仓、水塔；不含温室 |
| `greenhouse` | 温室、塑料大棚 |
| `road` | 车行道和铁路：街道、公路、非铺装车行道、铁路、机场跑道与滑行道 |
| `car` | 汽车等道路车辆（作为物体） |
| `paved_open` | 道路以外的铺装地面：停车场、人行道、广场、硬化场院、硬质运动场 |
| `sports_grass` | 天然草皮运动场 |
| `construction_site` | 施工场地与人工透水地面：压实土、砾石场地与小路 |
| `extraction_dump` | 采石场、露天矿、垃圾填埋场、土石堆 |
| `bare_natural` | 自然裸地：沙地、海滩、岩石、干河床 |
| `snow_ice` | 冰雪 |
| `water` | 水体：海、河、湖、池塘、水库；不含游泳池 |
| `swimming_pool` | 露天游泳池 |
| `herbaceous` | 非农用草本：草坪、公园、高尔夫球场、天然草地、路边草带 |
| `farmland` | 农用地：耕地作物、水田、菜地、牧场、刈割草地、果园及其他多年生木本作物；不含葡萄园和翻耕地 |
| `plowed_land` | 无可见植被的耕地：翻耕或刚收割 |
| `vineyard` | 葡萄园 |
| `shrub` | 低于约 5 m 的自然木本植被：灌丛、幼树、荒原 |
| `deciduous_tree` | 高于约 5 m 的阔叶树 |
| `coniferous_tree` | 高于约 5 m 的针叶树 |

各数据集的原生类 = 细类集合：

| 数据集 | 原生类 = 细类集合 |
|---|---|
| OEM | **bareland** = {extraction_dump, bare_natural, snow_ice}<br>**rangeland** = {herbaceous, shrub}<br>**developed space** = {paved_open, sports_grass, construction_site}<br>**road** = {road, car}<br>**tree** = {deciduous_tree, coniferous_tree}<br>**water** = {water, swimming_pool}<br>**agriculture land** = {farmland, plowed_land, vineyard}<br>**building** = {building, greenhouse} |
| FLAIR-HUB | **building** = {building}<br>**greenhouse** = {greenhouse}<br>**swimming pool** = {swimming_pool}<br>**impervious surface** = {road, car, paved_open}<br>**pervious surface** = {construction_site, extraction_dump}<br>**bare soil** = {bare_natural}<br>**water** = {water}<br>**snow** = {snow_ice}<br>**herbaceous vegetation** = {sports_grass, herbaceous}<br>**agricultural land** = {farmland}<br>**plowed land** = {plowed_land}<br>**vineyard** = {vineyard}<br>**deciduous** = {deciduous_tree}<br>**coniferous** = {coniferous_tree}<br>**brushwood** = {shrub} |
| Global-LoveDA | **background** = {greenhouse, paved_open, sports_grass, construction_site, herbaceous}<br>**building** = {building}<br>**road** = {road, car}<br>**water** = {water, swimming_pool}<br>**barren** = {extraction_dump, bare_natural, snow_ice}<br>**forest** = {shrub, deciduous_tree, coniferous_tree}<br>**agriculture** = {farmland, plowed_land, vineyard} |
| DeepGlobe | **urban** = {building, road, car, paved_open, sports_grass, construction_site, swimming_pool}<br>**agriculture** = {greenhouse, farmland, plowed_land, vineyard}<br>**rangeland** = {herbaceous, shrub}<br>**forest** = {deciduous_tree, coniferous_tree}<br>**water** = {water}<br>**barren** = {extraction_dump, bare_natural, snow_ice} |
| Potsdam | **impervious surfaces** = {road, paved_open}<br>**building** = {building, greenhouse}<br>**low vegetation** = {sports_grass, herbaceous, farmland, plowed_land, vineyard, shrub}<br>**tree** = {deciduous_tree, coniferous_tree}<br>**car** = {car}<br>**clutter** = {construction_site, extraction_dump, bare_natural, snow_ice, water, swimming_pool} |

## 仍不确定的格子

下面 13 个格子的归属仍缺少官方依据（v1 为 17 个），JSON 里保留了 `alternatives`。
按影响的像素量，前 2 个最值得日后核对：

| 优先 | 细类 | 数据集 | 当前归属 | 备选 | 理由 |
|---|---|---|---|---|---|
| 1 | `herbaceous` | Global-LoveDA | background | agriculture | LoveDA 没有草地类，城区草坪和农区草地都要落到某一类，像素量大 |
| 2 | `shrub` | Global-LoveDA | forest | background | 地理国情普查的林地含灌木林，但 LoveDA 的 forest 是否含灌丛未写明 |
| — | `greenhouse` | OEM / Global-LoveDA / Potsdam | building / background / building | agriculture land / agriculture, building / clutter | 三个数据集的定义都没提温室 |
| — | `construction_site` | Global-LoveDA / Potsdam | background / clutter | barren / impervious surfaces | 定义未提施工场地 |
| — | `extraction_dump` | Global-LoveDA | barren | background | |
| — | `swimming_pool` | Global-LoveDA | water | background | |
| — | `sports_grass` | DeepGlobe | urban | rangeland | 0.5 m 下城区球场多随建成区多边形标注 |
| — | `bare_natural`、`plowed_land`、`shrub` | Potsdam | 见上表 | 见 JSON | Potsdam 的城市场景中很少出现 |

核对方法（任选其一）：

- **抽样看图**：每个格子在对应数据集里找 20–30 处实例，看它被标成哪个原生类；
- **跨数据集混淆**：等单数据集模型训好后，用细类最全的 FLAIR 模型或 OEM 模型预测 Global-LoveDA 的选择划分，
  统计 background、forest 等像素被预测成哪些类（创新点 C 的做法）；
- **联合训练后看 `fine_confusion`**：评估工具会输出“原生真值 × 细类预测”的像素计数。

日后修改：编辑 JSON 中对应格子的 `to`，把 `basis` 改为 `official` 或 `inferred`，删去 `alternatives`，再运行
`python -m common.unified_taxonomy` 确认通过检查。任何改动都会改变细类文件的 SHA256，已训练的 checkpoint 既不能续训，
也会被评估工具拒绝；有了正式结果之后再改，应另存为新版本（`version` 加一），不要原地修改 v2。

## 已知的局限

- **兜底类很大**：
  - Global-LoveDA 的 background 含 5 个细类，DeepGlobe 的 urban 含 7 个；
  - Potsdam 的 clutter 和 low vegetation 各含 6 个。
  - 模型可能把难判断的像素随意分到这些集合里的某个细类（提案 §2.7 的风险一）。
- **6 个细类没有任何数据集单独标注**：`road`、`paved_open`、`sports_grass`、`construction_site`、`extraction_dump`、`herbaceous`。
  - 它们只能靠数据集之间的交集学出来，例如 `road` = OEM 的 road ∩ FLAIR 的 impervious ∩ Potsdam 的 impervious；
  - 这些细类本身的预测不一定可靠，但原生口径的评估不依赖细类本身的身份。
  - 单独标注过的 13 个细类及来源：`car`（Potsdam）、`water`（FLAIR、DeepGlobe）、`building`（FLAIR、Global-LoveDA），
    其余 10 个都来自 FLAIR。
- **合并带来的标签噪声**：停车场上的车（OEM、Global-LoveDA）、FLAIR 的铁路和砾石路、Potsdam 的硬质运动场、
  各数据集里的牧场和果园，都可能与所在集合的原生标签不一致。
- **DeepGlobe 的 urban 多边形很粗**：0.5 m 下，街区内的树和草坪也标成 urban。
  - 这里仍把树、草归到 forest / rangeland，这些像素在训练中相当于标签噪声；
  - 不能把树、草也并进 urban 的集合，否则 urban 会包含 forest 的全部细类，forest 永远不会被预测。
- **各数据集原本就 ignore 的像素仍然 ignore**：
  - FLAIR 的 clear cut、ligneous、mixed、undefined；
  - Global-LoveDA 的 playground（原始值 8）；
  - Potsdam 的腐蚀边界；DeepGlobe 的 unknown；OEM 的 unknown。
- **分辨率不统一**：0.05–0.5 m 混在一起训练，不做 GSD 归一化。
- **SegFormer 解码头的学习率是官方的 1/10**：官方配置 `head` 的 `lr_mult=10`，这里不放大。SegFormer-B0 试跑 4 万次（解码头 6e-4、编码器 6e-5，同种子）的选模均值 0.6315，当前配方同一时刻 0.6278，差 0.004，在噪声范围内，所以配方不改。这只是 4 万次的结果，20 万次时没有对照。
- **映射的依据强弱不一**：58 个 `official` 格子里有 13 个只是类名相同，5 个部分成立，2 个与原文矛盾，见[勘误表](UNIFIED_FINE_V2_ERRATA.md)。
- **五个选择划分的严格度不同**（OEM 按城市、FLAIR 按省、Potsdam 按图块、Global-LoveDA 和 DeepGlobe 按图），均值混合了这些口径：
  - Global-LoveDA、DeepGlobe 的选择图块约一半能和 train 拼接（51%、56%，边缘匹配的启发式，有假阳性）。这不等于分数被抬高：5 个模型在不相邻子集上的分数都高于全部图块（Global-LoveDA 高 0.011–0.016，DeepGlobe 高 0.038–0.083）。但两个子集不能直接比，判据要求图块边缘有纹理，纹理少的图块只会落在不相邻子集。所以相邻是否影响分数没有定论；
  - Potsdam 的选择划分只有图块 2_10，clutter 占 13.99%（train 3.68%），单张图块占选模均值的 1/5；
  - FLAIR 的 `snow` 在选择划分上没有像素，实际算 14 类。
- **各数据集的训练量不等**：每个数据集抽 32 万张次，FLAIR 只相当于 2.1 个 epoch，DeepGlobe 相当于 569 个。
  SegFormer-B0 单独训练 FLAIR，4 万次 0.6057，8 万次 0.6250（最后一次验证，best 0.6262），所以 FLAIR 的训练量可能不够。这只有一个模型、一个种子。所有模型的预算相同，模型之间的比较不受影响，配方没有改。

## 训练配方

配方写在 `configs/config_unified_v4.py`：crop 512、batch 8、AdamW、训练 AMP、评估 FP32、种子 42、不开确定性模式。
各模型的学习率和损失写在各自的 leaf 里，都是暂定值，见 [MODELS.md](MODELS.md)。

| 项目 | 取值 | 说明 |
|---|---|---|
| 训练数据 | 五个数据集的 `train` | 用 `LAND_COVER_UNIFIED_DATASETS=a,b,...` 可只用其中几个（测试、日后的留一数据集实验） |
| 采样 | 每个虚拟 epoch 1000 次迭代 × batch 8 = 8000 个样本，每个数据集固定 1600 个 | 每个数据集有自己的排列流；同一 batch 内混合多个数据集 |
| 训练长度 | 每个数据集 40 个虚拟 epoch，五个数据集共 200 个 = 20 万次迭代 | 每个数据集被抽到的样本数相当于单独训练 4 万次迭代 |
| 验证 | 每 2 万次迭代一次（共 10 次），五个选择划分全部评估 | FLAIR 的选择划分是完整的 38,425 张 |
| 早停 | 无 | |
| 选模 | 五个数据集原生 mIoU 的平均值最高的 checkpoint（`best_model.pth`），另存最后一个 | best checkpoint 同时记录各数据集的选择划分 mIoU |
| 损失 | 集合交叉熵；有辅助头的模型（Swin、PIDNet、DDRNet）辅助头同样用集合交叉熵 | PIDNet 的边界感知 CE 也换成集合交叉熵；不支持 Dice 和类别权重 |

**成本**：一次联合训练的迭代数相当于五次单数据集训练之和，另加 10 次覆盖五个选择划分的验证。

**续训指纹**：细类文件的 SHA256、细类名、训练数据集及其划分、标签偏移、集合矩阵、采样权重都写入指纹。
改了细类表之后，旧 checkpoint 既不能续训，也会被评估工具拒绝。

## 代码结构

| 文件 | 作用 |
|---|---|
| `configs/class_maps/unified_fine_v2.json` | 统一细类 v2：19 个细类及其在每个数据集中的归属、依据和备选 |
| `common/unified_taxonomy.py` | 读入并检查细类文件，生成集合矩阵 M_d；`python -m common.unified_taxonomy` 打印上面的表格 |
| `common/unified_protocol.py` | 模型登记表、单个数据集的配置条目、标签偏移与集合矩阵、联合验证与打印 |
| `configs/config_unified_v4.py` | 基础配置：协议、准备副本表、配方、训练长度 |
| `configs/config_unified_<model>.py` | 10 个 leaf，每个配置一个 |
| `common/engine.py` | `SetCrossEntropyLoss`；`evaluate(..., projection=M_d)` 按集合求和评估并输出 `fine_confusion` |
| `common/dataset.py` | `LabelOffsetDataset`、`BalancedEpochSampler`、`build_unified_dataloaders()` |
| `common/train.py` | 统一细类协议的数据、验证、指纹与 best checkpoint；细类文件不是 `final` 时拒绝 `--run-tag` |
| `tools/evaluate_unified.py` | 在各数据集 test 划分上按原生类别做最终评估 |
| `tools/run_baselines.py` | 按顺序批量训练，默认 10 个配置由快到慢 |

## 命令

```bash
cd /data/yfk/land_cover_v4
export LAND_COVER_DATASETS=/data/yfk/datasets
PY=/home/yx/miniconda3/envs/yfk-cv/bin/python

# 检查细类文件并打印映射表与待核对清单
$PY -m common.unified_taxonomy

# 看批量训练计划
$PY tools/run_baselines.py --run-tag <tag> --seed 42 --dry-run

# 正式训练：单个模型 / 其中几个 / 全部 10 个配置（由快到慢）
$PY common/train.py --config config_unified_unet_r50 --run-tag <tag> --seed 42
$PY tools/run_baselines.py --run-tag <tag> --seed 42 --models ddrnet unet_r50
$PY tools/run_baselines.py --run-tag <tag> --seed 42

# 最终评估：默认评估训练用的全部数据集；--datasets 可选其中几个
$PY tools/evaluate_unified.py --model unet_r50 \
  --checkpoint checkpoints/unified_v4/all/baselines/<tag>/unet_r50/best_model.pth
```

**产物位置**
- 训练：`outputs/unified_v4/all/baselines/<tag>/<MODEL_SLUG>/` 和 `checkpoints/unified_v4/all/baselines/<tag>/<MODEL_SLUG>/`；
  只用部分数据集时，`all` 换成用 `+` 连接的数据集名。
- 评估：`evaluations/unified_v4/all/<MODEL_SLUG>/<checkpoint sha256 前 12 位>/`，包括：
  - `metrics.json`：每个数据集、每个 test 划分的原生逐类 IoU / F1 / 像素数、`miou_present`、`miou_fixed`、OA，
    以及“原生真值 × 细类预测”的 `fine_confusion`；
  - `scores.txt`。
- `MODEL_SLUG`：Swin 是 `swin_t_upernet`，DDRNet 是 `ddrnet23_slim`，其余与模型键相同。
- 开发仓库里的 `unified_v2_s42` checkpoint 可以用本仓库的评估工具评估，只要把 `--checkpoint` 指过去。
  v1 的 checkpoint 与当前细类文件不符，评估工具会拒绝。

## 中断与运行记录

- **续训**：`--run-tag` 默认拒绝写入已有结果的目录。中断后用同样的命令加 `--resume`：
  有 `checkpoint_last.pth` 且续训指纹一致的模型接着训练，已跑完的立即结束，没有 checkpoint 的从头开始。
  随机状态、优化器、AMP 缩放器都会恢复；leaf 自己不能打开 `RESUME`。
- **保存频率**：每次验证（2 万次迭代）保存一次，所以一次中断最多损失这段进度。保存是先写临时文件再改名，保存时断电旧文件也完好。
- **代码版本**：`resolved_config.json` 和 checkpoint 的 `code` 字段记录 git commit、未提交的已跟踪文件，
  以及 `common/`、`configs/`、`tools/` 的文件哈希。工作区未提交时，`core_sha256` 可以和之后的提交对上。
  日志每次启动写一行 `[*] code:`；续训时代码和 checkpoint 不同会警告，但不拒绝。
- **历史文件**（`OUT_DIR` 下，JSON Lines；续训后同一个 epoch 可能出现两次，取最后一条）：
  - `train_history.jsonl`：每个 epoch 的 loss、学习率、耗时，统一协议另有 `dataset_loss`（每 `LOG_INTERVAL` 次迭代抽样一次，只统计主输出）；
  - `val_history.jsonl`：每次验证的完整分数，含各数据集的原生混淆矩阵和 `fine_confusion`。
- **评估工具**：
  - 比较 checkpoint 的完整协议快照，`SEED`、`NUM_WORKERS` 等运行设置除外；
  - 默认只接受 `best_model.pth`，评估 `checkpoint_last.pth` 要加 `--allow-last`；
  - `--selection` 评估选择划分而不是 test；
  - `--per-image` 另存逐图混淆矩阵 `per_image_<dataset>_<split>.npz`，用于同图配对检验和子集重算；
  - `metrics.json` 同时有 `miou_present` 和 `miou_fixed`。
- **去掉数据集时**：`python -m common.unified_taxonomy` 列出去掉每个数据集后的秩和分不开的细类；
  `UNIFIED_IDENTIFIABILITY` 记入配置快照，有分不开的细类时训练日志会写 `[!]`。留一数据集实验要同时报告全部类和可达类。

`unified_v2_s42` 中在这些改动之前跑完的模型没有 `code` 字段和历史文件。

## 验证

- `tests/test_unified_protocol.py`：
  - **细类文件**：已定稿，覆盖五个数据集的全部原生类；没有细类同属两个原生类；
    类名错误、缺数据集、原生类没有细类、集合包含、两个细类无法区分、basis/status 非法、key 重复都会被拒绝。
  - **集合交叉熵**：单位矩阵时等于普通交叉熵；合并集合时等于 `-log(p_a + p_b)`；全部 ignore 时为 0；
    `SegLoss` 深监督和 `PIDNetLoss` 都能用。
  - **按集合评估**：单位矩阵时与普通评估一致；细类 argmax 与原生预测不同时，按集合概率之和取预测。
  - **均衡采样**：每个数据集每个 epoch 的样本数固定，且等于它自己的排列流；续训后结果相同；按权重分配。
  - **配置**（四个合成准备副本）：标签偏移、集合矩阵、各数据集的列表与准备副本一致；全部 leaf 都输出 19 个细类；
    指纹含细类与数据集；测试划分不能进入训练；只有定稿的细类文件允许 `--run-tag`；
    训练 batch 中的标签都落在各自数据集的偏移区间内；选择划分保持原生标签。
  - **工具**：评估工具按原生类别写出全部 test 划分，`fine_confusion` 每行之和等于该类像素数；细类文件不一致时拒绝；
    启动器解析统一细类 leaf，细类文件是草案时只允许 dry-run。
  - **服务器上**：五个真实准备副本能一起载入，标签偏移为 OEM 0、FLAIR 8、Global-LoveDA 23、DeepGlobe 30、Potsdam 36，共 42 个原生类。
- 与开发仓库的一致性见 [CHANGELOG.md](../CHANGELOG.md)。
