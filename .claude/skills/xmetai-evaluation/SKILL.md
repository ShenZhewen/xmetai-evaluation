---
name: xmetai-evaluation
description: >
  气象模型评测结果的分析与出图报告。当用户给出评测产物——xmetai_evaluation 的
  evaluation_results 结果目录、scores.csv 长表、或降水 TS 宽表——并希望
  "画图""出报告""分析这次评测""帮我对这个评测结果生成报告"
  "看看强降水表现""看看 RMSE / ACC / 活跃度 / 谱""和基准/别的模型比一比"
  "报告要像论文的实验部分""图下面要有图注""每一节要有总结"时使用。
  覆盖两条链路：降水分类检验（TS 系列）与确定性连续场检验（RMSE / ACC / 预报活跃度 / 纬向谱）。
  **能力身份从产物目录的 manifest.json 自动识别，用户不用记能力名**。
  产出：时效曲线、阈值对比、TS 热力图、性能图、跨时效箱线图、谱曲线，以及一份
  **按论文实验章节组织**的 Markdown 报告——图表就地编号（`图 N` / `表 N`）、
  图下有图注、每节末尾有一段带具体数值的定量+定性小结。
metadata:
  version: 3.1
  author: 沈哲文 (szw)
---

# XMetAI 评测结果：出图与报告

## 这个 skill 干什么

把**评测结果**变成**图表 + 一份能读的报告**。

输入是评测产物（目录或 CSV），输出是 `*.png` 加一份 `REPORT.md`。

## 不干什么

- **不重算指标**。本 skill 只消费已有的评测结果；要重算就去跑 `xmetai_evaluation` 的评测流程，不要在这里另起一套算法。
- **不猜数据**。缺列、缺文件、缺时效时**如实说明**，不要用默认值或估算值把报告"补全"。
- **不编造结论**。报告里的每个数字都必须来自输入表；没有依据的判断不写。

---

## 触发场景

用户说了类似下面的话，就应该用本 skill：

- "这是 FGVP 的 TS 结果，帮我画个图 / 写份报告"
- "分析一下这次降水评测，强降水表现怎么样"
- "对比一下新模型和 FuXi 的 TS"
- "评测跑完了，结果在 `evaluation_results/ts_det_fgvp/` 下面"
- "连续场评测跑完了，看看 RMSE 随时效涨得怎么样"
- "这个 ACC 衰减正常吗""活跃度比接近 1 吗""谱是不是失真了"

---

## 能力路由（先认能力，再选模板）

**不要靠目录名判断这是哪条链路。** 配置名（`weather_field_scores_era5_fuxi`）
与流程名（`weather_field_scores`）本来就不绑定。

权威判据是产物目录里 `manifest.json` 的 `resolved_config.pipeline`；
没有 `resolved_config` 的老产物，退路是 `protocol_id` + `manifest["metrics"]` 的组合。
**这些判断已经在代码里实现了**，直接调（见下面的工作流）：

```bash
python skills/xmetai-evaluation/scripts/generate_report.py <产物目录> --out reports/<名字>
```

它会自己认能力、派发到对应模板。认不出、或者认出来但**报告模板还没做**时，
它会明确报错——**拿错模板比报错危险得多**，所以不要绕过它自己拼。

| 流程 `pipeline` | 覆盖 | 报告模板 |
|---|---|---|
| `weather_ts_det` / `weather_ts_ens` / `fdp_precip_ts` | 站点降水分类检验：TS / POD / FAR / 频率偏差 | ✅ [`assets/templates/ts.md`](./assets/templates/ts.md) |
| `weather_field_scores` | 确定性连续场：RMSE / ACC / 活跃度 / 纬向谱 | ✅ 代码已实现；模板文件待补 |
| 其余 8 条流程 | CRPS、FSS、概率评分…… | ⛔ 还没做，会明确报错 |

> 三条 TS 流程**共用一套模板**：它们写出的产物表结构相同（都是
> `diagnostics/categorical_wide.csv`）。**模板按家族分，不按流程分。**

---

## 输入

两条链路吃两类表，先认清楚是哪一类：

| 链路 | 主表 | 必需列 |
|---|---|---|
| A. 降水分类检验（TS 系列） | `diagnostics/categorical_wide.csv`（宽表） | `lead_h`、`grade`、`TS` |
| B. 确定性连续场（`weather_field_scores`） | `scores.csv`（统一长表） | `lead_h`、`variable`、`metric`、`value` |

### A. 降水分类检验（TS 系列）

#### 数据表契约

必需列：`lead_h`、`grade`、`TS`

常见可选列：`window_h`、`threshold_mm`、`hits`、`misses`、`false_alarms`、`n_pairs`、`POD`、`FAR`、`漏报率`（或 `miss_rate`）、`BIAS`

- 多出来的列（如 `run_id`）忽略即可；
- `漏报率` 缺失时用 `1 − POD` 推导；
- **必需列缺失 → 停下来告诉用户缺什么**，不要继续。

#### 宽表不在怎么办

**不做长表→宽表转换**——转出来的口径和评测流程自己写的那份未必一致，
两份数据混着用比报错危险。宽表缺失时渲染器会明确报错，照着报错里的提示
在评测配置里加上 `writers=["csv_long", "categorical_wide"]` 重跑即可产出。

### B. 确定性连续场（`weather_field_scores`）

#### 数据表契约

直接吃 `scores.csv` 长表，必需列：`lead_h`、`variable`、`metric`、`value`。
`unit` 与 `status` 有则用、没有则兜底。

长表里实际会出现 7 个 `metric`：`rmse`、`acc`、`activity_ratio`、`activity_bias`、
`activity_forecast`、`activity_observation`、`spectrum_power_ratio`。
**变量与指标不是满交叉**——每个变量只算被逐变量路由到的指标
（`z500` 有 ACC，`q700`/`t850`/`msl` 没有；活跃度只有 `z500`/`u*`/`v*`/`ws850` 有）。
所以看到某变量"没有某个指标"，多半是路由没给它，不一定是缺数据。

逐波数谱曲线（`group=k=<波数>`）落在 `diagnostics/scores_detail.csv` 里，
**默认不读**（实测 90 万行，单变量单时效就有 721 个波数）；
只有点名要看曲线时才用 `--spectrum-variable` / `--spectrum-lead` 按需读。

#### 怎么出

```bash
# 一条命令：自动认能力 + 出图 + 写报告
python -m xmetai_evaluation.visualization.report \
  evaluation_results/weather_field_scores_era5_fuxi \
  --out reports/field_fuxi

# 需要看谱曲线时再加两个参数（会读明细表，慢一些）
python -m xmetai_evaluation.visualization.report \
  evaluation_results/weather_field_scores_era5_fuxi \
  --out reports/field_fuxi --spectrum-variable z500 --spectrum-lead 24
```

### 从哪里找输入

产物目录由用户给出。目录里就这几样，不需要猜：

| 文件 | 用途 |
|---|---|
| `manifest.json` | 认能力（`resolved_config.pipeline`）、取 `run_id` 当报告标题 |
| `diagnostics/categorical_wide.csv` | TS 系列的主表 |
| `scores.csv` | 连续场的主表 |
| `diagnostics/scores_detail.csv` | 连续场的逐波数谱明细，**默认不读**（很大） |

**`evaluation_results/` 的布局**：一个模型一次运行一个目录，目录名 = config 的 `name` = `run_id`，
形如 `weather_<流程>_<模型>`（`weather_ts_det_fgvp`、`weather_ts_ens_fuxi`）。
**"对比其他模型"就是看这个目录的同级目录**——每个模型一个，一看就找齐了。

> `manifest.json` 里 `artifacts` 的路径是**生产机的绝对路径**（`/workspace/...`），
> 只能拿它判断"有哪些产物"，**不能拿它拼本机路径**。本机路径按上表约定拼。

---

## 工作流（智能体自己执行）

**这个 skill 只有一条命令。用户不会自己跑 python，命令由你执行。**

**第 1 步 · 跑**

```bash
python skills/xmetai-evaluation/scripts/generate_report.py <产物目录>
```

它自己认能力、选模板、出图、写 `REPORT.md`，最后打印产物清单。
输出目录默认 `reports/<run_id>/`，要换就加 `--out`。

- 用户说了本次改了什么（"我优化了 TP 参数化"）→ 加 `--change "..."`，写进报告头；
- 想换报告标题 → 加 `--model "FGVP-v2"`。默认取 `manifest` 的 `run_id`；
  **不要**用 `forecast_source`（那是 reader 类型如 `"fuxi"`，不是模型名）。

**第 2 步 · 读回报告并转述**

打开生成的 `REPORT.md`，向用户**只转述 3–5 条核心结论**（每条都要带具体数值），
并列出产出的文件。不要整篇复述，也不要在转述里加报告里没有的数字。

**第 3 步 · 失败就直说**

命令报错时**把错误原样转述**，不要自己绕过它去手工拼流程：

| 报错 | 含义 |
|---|---|
| `... 的报告模板还没做` | 这条能力还没有出图与报告代码，如实告诉用户 |
| `... categorical_wide.csv 不存在` | 评测时没写宽表 writer，让用户改配置重跑 |
| `manifest.json 不存在` | 用户给的目录不是评测产物目录 |
| `出图需要 matplotlib` | 环境没装：`pip install -e .[viz]` |

**第 4 步 · 用户要"对比其他模型"时**

加 `--compare "名字=CSV路径"`（可重复）。对比模型吃的是**宽表**，不是产物目录——
它们没有 `manifest.json`，认不出能力，所以名字由你显式给出：

```bash
python skills/xmetai-evaluation/scripts/generate_report.py \
  evaluation_results/weather_ts_det_fgvp --out reports/ts_3models --model FGVP \
  --compare "FuXi=evaluation_results/weather_ts_det_fuxi/diagnostics/categorical_wide.csv" \
  --compare "AIFS=evaluation_results/weather_ts_det_aifs/diagnostics/categorical_wide.csv"
```

给了对比模型，报告会多出"与对比模型的比较"一节，并把它们一起放进**箱线图**与**差值图**。
只对 TS 系列有意义——连续场产物上给 `--compare` 会明确报错。

**用户多半不会给路径**（"我结果在 `evaluation_results/weather_ts_det_fgvp`，我要你对比其他模型的结果"）。
那就**自己去 `evaluation_results/` 找，别反问**，但**先把清单报给用户确认**再跑。五步：

1. **看主模型产物目录的同级目录**：凡是有 `diagnostics/categorical_wide.csv` 的，都是可对比的模型。
   这是全仓唯一的判据，新旧目录都适用。
2. **拿模型名**：读该目录 `manifest.json` 的 `run_id`（`weather_ts_det_aifs` → **AIFS**、
   `weather_ts_ens_fuxi` → **FuXi-ENS**）；没有 manifest 的按目录名同样处理。
   名字取**模型简称**，别把 `run_id` 原样搬进报告。
   **不要用 manifest 的 `forecast_source` / `model_id`**——那是 reader 类型
   （`weather_ts_det_fgvp` 的 `model_id` 写的居然是 `"fuxi"`，但模型是 FGVP）。
3. **跑之前先核对口径**：各目录的 `window_h`、`lead_h` 集合、`grade` 集合要对得上。
   量级少一两档没关系（`weather_ts_ens_fuxi` 就没有 `≥250`），
   但**窗口或时效范围对不上就不能画在一张图里**——那要在报告里点明，或者干脆不比。
4. **把清单报给用户**：目录 → 打算用的名字，一次列清楚，问一句"这几个对吗"，再往下走。
5. 拼命令跑。

> `evaluation_results/` 下每个模型一个目录，每个目录里宽表都在 `diagnostics/categorical_wide.csv`
> （`weather_ts_det_*` 是确定性、`weather_ts_ens_*` 是集合，`_archive/` 是历史结果——
> **归档目录默认不参与对比**）。
> 同一个模型可能有多份运行产物，逐格比一下 `n_pairs` 就知道是不是同一次评测——
> **两次几乎一样的运行别同时放进对比图**，会画出两个重合的箱子。

---

## 输出契约

**B. 确定性连续场**（`weather_field_scores`，`--out` 目录下）：

```
rmse_vs_lead.png                     一个变量一条线；按单位分面（z500 是 m^2/s^2、t2m 是 K…）
acc_vs_lead.png                      ACC 随时间衰减
activity_ratio_vs_lead.png           带 y=1 参考线
spectrum_power_ratio_vs_lead.png     带 y=1 参考线
rmse_heatmap.png                     变量 × 时效，填的是"相对首时效的倍数"（单位无关）
spectrum_curve_<变量>_<时效>h.png     按需，log 纵轴，预报/实况两条线
REPORT.md
```

`REPORT.md` 结构固定：评测对象与数据来源（全部取自 `manifest.json`，路径可溯源）→
执行摘要（规则化诊断，**每条结论都带具体数值**）→ 逐变量总览 → RMSE 随时效 →
活跃度与纬向谱 → 图表索引 → 口径与注意事项。

**A. 降水分类检验**（TS 系列）：

```
reports/<模型>_<批次>/
  ${window}h_TS_vs_lead.png          时效-技巧曲线（按降水等级分线）
  ${window}h_POD_vs_lead.png
  ${window}h_FAR_vs_lead.png
  ${window}h_BIAS_vs_lead.png        BIAS 带 =1 参考线，不截断
  ${window}h_metrics_by_threshold_<lead>h.png
  ${window}h_TS_heatmap.png          等级 × 时效
  ${window}h_performance_diagram_<lead>h.png
  ${window}h_TS_boxplot.png          各模型 TS 跨时效分布（箱线图）
  ${window}h_multi_model_TS_delta_vs_<主模型>.png   多模型对比**画差值**（不画绝对值）
  REPORT.md
```

> 多模型对比为什么画差值：几个模型水平接近时（例如 24h 的 ≥0.1mm 分别是
> 0.529/0.519/0.507/0.504），绝对曲线会重合成一团，谁也看不出差异；
> 差值曲线直接回答"谁在哪一档、哪个时效领先"。**注意纵轴是「对比模型 − 本模型」**，
> 正值表示对比模型领先。

`REPORT.md` 按**论文实验章节**组织，每节都是「引导句 → 图表 → 小结」：

```
评测对象与对比对象（模型 + 来源文件 + 可比性提示）
一、结论摘要           带数值的规则化诊断 + 业务定性
二、逐降水等级表现     表 1 + 图 1（阈值对比）→ 小结
三、TS 随时效变化      图 2（曲线）图 3（热力图）+ 表 2 → 小结
四、误差形态与偏差结构  图 4–7（POD / FAR / BIAS / 性能图）→ 小结
五、跨时效分布         表 3（分位数）+ 图 8（箱线图）→ 小结
六、与对比模型的比较   （可选）表 4 + 图 9（差值）→ 小结
七/六、改进建议
八/七、口径与注意事项
```

**没有"图表索引"节**——图都就地带着图注出现，再列一遍只是重复。

> **"结构固定"是被测试守着的**：[`assets/templates/ts.md`](./assets/templates/ts.md)
> 里放的是一份**填了空格的完整报告**——`{…}` 是数值位置，其余每个字（导语、表头、
> 分隔行、加粗、口径说明）都和最终 `REPORT.md` 逐字一致。
> `tests/unit/visualization/test_ts_report.py` 实跑一遍渲染器，拿真报告去对这份骨架，
> 一共两条独立的检查：
> 1. **逐字扫描**——骨架里所有不含占位符的行必须逐字出现且顺序一致，章节序列逐项比对；
> 2. **编号契约**——`图 N：` / `表 N：` 必须正好是 `1..N`（不跳号不重号）、每节都有
>    `**小结**` 且小结里带数字、产物里出的每一张图都在正文里出现过。
>    这一条是必需的：图注/表题整行都含 `{…}`，第 1 条抓不到它们。
>
> 骨架有三个分支：单模型 / 有基准 / 多模型，分别对应 A / B / C 三块。
> **二 / 三 / 四 / 五这四节三个分支完全一样**，有对比对象时多出的是**第六节**，
> 于是"改进建议 / 口径"从 六/七 挪到 七/八。
> 改渲染器的章节结构必须同步改骨架，否则测试会红。

---

## 硬性规则

1. **数字只能来自输入表**；推算值（如 `漏报率=1−POD`）要在报告里注明是推算的。
2. **等权平均 ≠ 样本加权**。报告里的"全时效平均"是各时效等权平均，必须保留这句说明。
3. **`BIAS` 是频率偏差**（预报事件数/观测事件数），不是平均误差；不要写成"平均误差"。
4. **缺值不是 0**。`TS` 为 `NaN` 表示无有效配对或无事件，与"评分 0.000"不同，解读要区分。
5. **`n_pairs` 太小或无事件的等级**不参与强弱结论（例如某等级命中为 0，只能说"无技巧"，不能说"技巧最差"）。
6. **跨模型比较要看共同口径**：报告里要交代数据来源、时段、时效范围；口径不一致时明确标注"不可直接比较"。
7. **中文字体缺失要说明**：如果出图脚本提示未找到中文字体，转述给用户并建议安装 Noto Sans CJK。
8. **失败要直说**：数据缺列、目录里没有结果表、只有基准没有新模型、能力模板还没做——
   都直接说清楚，不要静默生成空报告，也不要套用另一条链路的模板。
9. **连续场不做绝对分级**。RMSE / ACC / 活跃度 / 谱没有公认的合格线，
   报告里**不写** "ACC ≥ 0.6 算合格"这类判决，也不打 🔴🟡🟢；
   只在数据内部比大小、看形态（衰减幅度、偏离 1 的方向、单调性）。
10. **`n_valid` 的含义不统一**。连续场长表里 `rmse`/`acc`/`activity_*` 行的 `n_valid`
    是格点数（全球 0.25° = 1038240），而 `spectrum_power_ratio` 行是
    **参与平均的二维场个数**（单起报单时效就是 1）。它不是样本量，
    不能拿来判断结果可不可信，更不能写进"样本量"栏。
11. **`unit` 随数据源走**。`*_phys` 布局下 z500 的 RMSE 单位是 `m^2/s^2`（**不除 g**，
    1 m²/s² ≈ 0.102 m），q 是 `g/kg`，无量纲的写 `1`。跨变量的原始值**不可直接比大小**，
    所以 RMSE 图按单位分面、热力图填"相对首时效的倍数"。
12. **TS 报告是"论文实验章节"，不是数据堆**。图下必有图注、表上必有表题，
    编号按在报告里的出现顺序连续；每个分析小节末尾必有一段 `**小结**`，
    小结里的每句话都要带具体数值或能被数据证伪——**写不出结论就如实说
    "数据不足以判断"，不拿"表现良好"这类话凑数**。分析文字由
    `ts_report.py` 的 `analysis_*` 系列函数生成，它们只吃已经算好的统计量，
    **不重新读表、不重新实现指标**（否则表、图、文字会各说各话）。
    这几条由 `tests/unit/visualization/test_ts_report.py` 守着，改渲染器时别绕过。

---

## 常见情形

| 情形 | 处理 |
|---|---|
| 数据里没有 24h 时效 | 不要写死 24h；取数据中最小的时效，并在报告里写明用的是哪个 |
| 只有 6h 窗口的结果 | 阈值口径通常是 `0.1/13/25`，与 24h 的阈值不同，不要混在一起比 |
| 只有单模型、没有基准 | 正常出单模型报告，跳过"与基准对比"一节 |
| 要对比多个模型，但没说对比谁 | 同级目录里有 `diagnostics/categorical_wide.csv` 的都是候选；把「目录 → 名字」清单报给用户确认后再跑（见"第 4 步"）。各模型的降水等级列表可能不同，出图与报告表会按共有等级对齐 |
| 目录里有同一个模型的两份产物 | 逐格比 `n_pairs`：完全一样就是同一次评测的两次运行，别同时放进对比图；`_archive/` 下的默认不比 |
| 集合预报的 TS | 输入的仍是"集合平均场"的 TS（确定性口径）；概率评分（AROC/BSS）是另一张表，不要混进同一张图 |
| 连续场里某变量没有某个指标 | 先看是不是**逐变量路由**没给它（`manifest` 的 `metric_options` 里每个指标带着自己的 `variables` 列表）；真缺数是结果里整行都没有，不是 `value` 为空 |
| 连续场目录的 `scores.csv` 里 `level`/`region`/`threshold`/`window_h` 全空 | 这是 `grid_valid_time` 协议的默认口径（只有 `variable`/`sample_unit`/`unit`），**不是缺数据** |
| 要判断谱是"总量正常但分布失真" | 只用 `spectrum_power_ratio` 看不出来（它是全波数求和后的比值）；用 `--spectrum-variable` / `--spectrum-lead` 出谱曲线看高波数段 |

---

## 参考文档

**报告长什么样**（格式契约，加新能力时照着写）：

- TS 系列：`assets/templates/ts.md`
- 模板目录说明与索引：`assets/templates/README.md`

**数字怎么判读**：

- 指标定义与读法：`references/evaluation-metrics.md`
- 降水评估口径：`references/precipitation-evaluation.md`
- 降水诊断规则（样本量门槛、技巧分级、偏差/误差形态判读、建议触发条件）：`references/diagnostic-rules.md`
- 连续场指标语义与判读：`references/field-evaluation.md`

---

## 维护说明

- **渲染逻辑在主仓库**：`xmetai_evaluation/visualization/`
  （`precipitation_plots.py` + `ts_report.py` 管 TS 链路，
  `field_plots.py` + `field_report.py` 管连续场链路，
  `report.py` 负责"认能力 + 派发"）。改能力改主仓库——那里能被 pytest 覆盖。
  规则与代码的对应关系写在 references 里，**冲突时以代码为准**。
- **`scripts/generate_report.py` 只做入口**（插 `sys.path` → 调 `build_report` → 打印产物）。
  **不要往里加判断逻辑**：一旦它开始自己认能力、自己拼流程，
  就又变成一份会跟主仓库漂移的副本。
- **`assets/templates/` 是格式契约**：一份报告骨架 = **一份完整的报告，只是数值位置
  用 `{…}` 空着**（不是说明文档，也不是模板引擎）。它不参与渲染，但被
  `tests/unit/visualization/test_ts_report.py` 实跑比对守着——
  改了渲染器的章节结构就必须同步改骨架，否则测试红。
- `agents/` 目前是空的。
- **本目录是源，`.claude/skills/` 下的是副本**（Windows 建符号链接要管理员权限，
  所以用复制）。改完这里要重新复制过去，见 `skills/README.md`。
- **维护者**：沈哲文 (szw)
