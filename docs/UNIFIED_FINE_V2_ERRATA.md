# 统一细类 v2 勘误表

> 2026-11-01 核对结果。`configs/class_maps/unified_fine_v2.json` **不改**：改动会改变文件 SHA256，已训练的 checkpoint 既不能续训，也会被评估工具拒绝。
> 引用映射依据时以本表为准，不能写"映射全部依据官方定义"。

核对方式是逐格对照各数据集的官方定义，再加小样本目视（每个主题 6 处，低分辨率）。目视结果不是定论。
LoveDA 系列没有公开类别对照，涉及它的判断属于推断。

## 依据等级

JSON 的 `basis` 只有 `official`、`inferred`、`uncertain` 三档，其中 58 个 `official` 的实际依据强弱不一，这里再细分：

| 标记 | 等级 | 含义 | 格子数 |
|---|---|---|---:|
| 无 | `official_text` | 官方原文明写 | 38 |
| ⁿ | `name_only` | 只是类名相同，官方没有文字定义（Global-LoveDA 6 格，Potsdam 7 格）| 13 |
| ᵖ | 部分成立 | 官方定义只覆盖这个细类的一部分 | 5 |
| ✗ | 与原文矛盾 | 见下面"与原文矛盾的格子" | 2 |
| † | `inferred` | 由官方定义推断 | 24 |
| ? | `uncertain` | 待核对 | 13 |

评估报告里的 39、13、4、2 是目视之前的统计。目视看到 20 m 以下的散屋不进 DeepGlobe urban，所以 `building` / DeepGlobe 从"原文明写"降为"部分成立"，变成这里的 38、13、5、2。


| 细类 | OEM | FLAIR-HUB | Global-LoveDA | DeepGlobe | Potsdam |
|---|---|---|---|---|---|
| `building` | building | building | building ⁿ | urban ᵖ | building ⁿ |
| `greenhouse` | building ? | greenhouse | background ? | agriculture † | building ? |
| `road` | road | impervious surface ᵖ | road ⁿ | urban ✗ | impervious surfaces ⁿ |
| `car` | road † | impervious surface | road † | urban ✗ | car ⁿ |
| `paved_open` | developed space | impervious surface | background † | urban | impervious surfaces ⁿ |
| `sports_grass` | developed space † | herbaceous vegetation | background † | urban ? | low vegetation † |
| `construction_site` | developed space | pervious surface | background ? | urban † | clutter ? |
| `extraction_dump` | bareland † | pervious surface | barren ? | barren † | clutter † |
| `bare_natural` | bareland | bare soil | barren † | barren | clutter ? |
| `snow_ice` | bareland † | snow | barren † | barren † | clutter † |
| `water` | water | water | water ⁿ | water | clutter |
| `swimming_pool` | water | swimming pool | water ? | urban † | clutter |
| `herbaceous` | rangeland | herbaceous vegetation | background ? | rangeland | low vegetation ⁿ |
| `farmland` | agriculture land | agricultural land ᵖ | agriculture ⁿ | agriculture | low vegetation † |
| `plowed_land` | agriculture land † | plowed land | agriculture † | agriculture † | low vegetation ? |
| `vineyard` | agriculture land | vineyard | agriculture † | agriculture | low vegetation † |
| `shrub` | rangeland | brushwood | forest ? | rangeland † | low vegetation ? |
| `deciduous_tree` | tree | deciduous | forest ⁿ | forest ᵖ | tree ⁿ |
| `coniferous_tree` | tree | coniferous | forest ⁿ | forest ᵖ | tree ⁿ |

## 与原文矛盾的格子

`road`、`car` / DeepGlobe 在 JSON 中是 `urban`、`official`。[DeepGlobe 论文](https://arxiv.org/abs/1805.06561)写的是：

> we intentionally did not annotate roads and bridges because it is already covered in the road extraction challenge.

而 OEM 论文又称 DeepGlobe "buildings and roads are included in the urban class"。两处原文有冲突。少量目视显示农村道路随周边地类，所以这两格最多只在 urban 图斑内成立。道路像素的比例没有测出来。

## 其他有问题的格子

"上限"是这个格子所在原生类占全部训练像素的比例（训练按数据集等量采样，即该类在本数据集的占比 × 20%），不是格子本身的份额。

| 格子 | JSON | 问题 | 上限 |
|---|---|---|---:|
| `herbaceous` / Global-LoveDA | background ? | LoveDA 引用的标准把草地列为一级类，不在 6 个前景类内；4 处目视都在 background，支持现状 | 5.3% |
| `deciduous_tree`、`coniferous_tree` / DeepGlobe | forest | forest 是 ≥20% 树冠且约 ≥20 m 的图斑，城区和农田中的散树不进 forest，图斑内还含空地和 clear cuts | urban 2.2%，forest 2.1% |
| `construction_site`、`extraction_dump` / Global-LoveDA | background ?、barren ? | **依据自相矛盾**：工地因"人工堆掘地"归 background，同属人工堆掘地的采石场和堆放物却归 barren，而标准 §6.3(m) 明确把人工堆掘排除在裸露地表之外。没有抽到工地实例，仍未解决 | 2.6% |
| `road` / FLAIR | impervious surface | 只有铺装路成立；细类描述含砾石路和铁路，FLAIR 把它们标 pervious。同样的铁路在 OEM 是 road | 1.2% 的一部分 |
| `paved_open` / Global-LoveDA | background † | 细类描述含人行道，而标准 §6.3(k) 把人行道并入路面 | 1.1% 的一部分 |
| `farmland` / FLAIR | agricultural land | 细类描述含果园，FLAIR 没有写；果园很可能标 deciduous | 未知 |
| `clutter` 的 5 个非水细类 / Potsdam | 见上表 | 集装箱等物体没有细类；选择图块 2_10 的 clutter 主要是工地，用它外推其他图块要谨慎 | 0.68%（2_10 上占 13.12%） |
| `sports_grass` / OEM、DeepGlobe | developed †、urban ? | OEM 原文是 "artificial grass areas"，天然草皮没有写；这个细类与 `herbaceous` 的区分完全靠这两个非官方格子 | 小 |
| `sports_grass` / Global-LoveDA | background † | 另有 playground（原始值 8，现被 ignore），已标为 playground 的球场不属于 background | 0.05% |
| `swimming_pool` / Global-LoveDA | water ? | 标准把游泳池列入构筑物（0761），更支持 background | 极小 |
| `greenhouse` / Global-LoveDA | background ? | 标准按设施拆分：大棚属构筑物，地膜和简易塑料棚属耕地，0/1 映射表达不了 | 未知 |
| `shrub` / Global-LoveDA | forest ? | 标准的林地含灌木林、稀疏灌丛，支持现状；但目视样点方向相反，保持 uncertain | 2.8% 的一部分 |
| `paved_open` / Potsdam | impervious surfaces ⁿ | ISPRS 明写网球场属 clutter；目视中硬质球场在 22 个 train 图块的 2 个里各占小块一半以上 | 小 |

## 其他事实错误

- JSON note "地理国情普查中园地属种植土地"：GDPJ 03-2013 全文没有"种植土地"一词，园地是与耕地并列的一级类。这条依据要补出处。
- 放牧草地在 FLAIR、OEM 属农业，在 Global-LoveDA 属 background。同样的外观被推向 `farmland` 和 `herbaceous`，这是标签噪声，映射改不掉。
- Potsdam 的 `car` 也含工程机械，这个细类不等于"道路车辆"。

## 引用时可以写和不能写

- 可以写：每格的依据等级；上面这些局限；原生口径指标与单数据集基线的比较。
- 不能写：映射全部依据官方定义；模型学到了 19 个全球语义；细类级精度（除 FLAIR 和 Potsdam car 外没有细类真值）。

## 没有做的改动

- 固定 ε 的软映射、`sports_grass` 并入 `herbaceous`：都要全部配置重训，只能先做探针，目前都没有做。
- 新增 `other_object`：不做。小于 50 m² 的 clutter 连通域约占全部训练像素的 0.19%，而且只有 Potsdam 能分开。
