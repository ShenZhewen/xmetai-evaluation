---
name: xmetai-evaluation
description: >
  气象模型评测结果的分析与出图报告。当用户给出降水分类检验（TS 系列）的结果——
  CSV 宽表、评测输出目录、或 xmetai_evaluation 的 evaluation_results 产物——
  并希望"画图""出报告""分析这次评测""看看强降水表现""和基准/别的模型比一比"时使用。
  产出：时效-技巧曲线、阈值对比、TS 热力图、性能图，以及一份基于真实数值的 Markdown 报告。
metadata:
  version: 2.0
  author: 沈哲文 (szw)
---

# XMetAI 评测结果：出图与报告

## 这个 skill 干什么

把**降水分类检验结果表**变成**图表 + 一份能读的报告**。

输入是一张 TS 结果表（CSV），输出是 `*.png` 加一份 `REPORT.md`。

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

---

## 输入

### 数据表契约

必需列：`lead_h`、`grade`、`TS`

常见可选列：`window_h`、`threshold_mm`、`hits`、`misses`、`false_alarms`、`n_pairs`、`POD`、`FAR`、`漏报率`（或 `miss_rate`）、`BIAS`

- 多出来的列（如 `run_id`）忽略即可；
- `漏报率` 缺失时用 `1 − POD` 推导；
- **必需列缺失 → 停下来告诉用户缺什么**，不要继续。

### 从哪里找输入

按这个优先级定位，找到就停：

1. 用户直接给的 CSV；
2. 用户给的目录下的 `diagnostics/categorical_wide.csv`（宽表，可直接用）；
3. 同目录下的 `scores.csv`（统一长表）——**需要先转宽表**，见下节；
4. 目录下的 `manifest.json` 里有 `input_files` 与 `artifacts`，可用来确认这批结果对应哪次运行。

### 只有长表 `scores.csv` 时怎么办

长表一行一个指标，需要合成为宽表再出图：

- `scores.csv` 里筛 `metric ∈ {ts, pod, far, miss_rate, frequency_bias}`，
  按 `run_id, window_h, lead_h, group, threshold` 透视（指标名转成列 `TS/POD/FAR/漏报率/BIAS`，
  `frequency_bias → BIAS`，`miss_rate → 漏报率`）；
- 列联表计数（`hits/misses/false_alarms/n_pairs`）在 `diagnostics/scores_detail.csv` 里，
  按同一组键连接进来即可；
- 合成后的表列名与上面的契约一致，之后流程完全相同。

也可以让评测流程直接产出宽表（配置里写 `writers=["csv_long","categorical_wide"]`），
避免每次都在这里拼。

---

## 工作流（智能体自己执行）

**第 1 步 · 定输入**
定位结果表；确认是宽表还是长表（长表先按上节转宽表）。

**第 2 步 · 校验**
检查必需列与记录数；把"时效范围、降水等级、样本量"先在心里过一遍——
如果某些等级 `n_pairs` 很小或无事件，报告里要指出，别当成结论。

**第 3 步 · 出图 + 写报告**
调用本仓库既有的可视化能力（**不要自己另写一套绘图逻辑**）：

```python
import pandas as pd
from xmetai_evaluation.visualization import PrecipitationPlotter

plotter = PrecipitationPlotter()
artifacts = plotter.create_report(
    pd.read_csv(csv_path),
    output_dir=out_dir,           # 建议 reports/<模型>_<日期>/
    model_name="FGVP",
    baseline_df=pd.read_csv(baseline_csv) if baseline_csv else None,
    baseline_name="FuXi",
    baselines={"FGVP(旧版)": pd.read_csv(old_csv), "FuXi": pd.read_csv(fuxi_csv)},
    sources={                      # 报告头会写明"评的是谁、和谁比、数据来自哪个文件"
        "FGVP": csv_path,
        "FGVP(旧版)": old_csv,
        "FuXi": fuxi_csv,
    },
    change_description="TP 参数化方案优化（用户口述的改动，写进报告头）",
)
# artifacts: {'TS_vs_lead': Path, ..., 'report': Path}
```

该函数负责：中文字体探测、时效自动选取（不写死 24h）、逐指标独立纵轴、
以及调用 `xmetai_evaluation.visualization.ts_report.build_ts_report` 写出 `REPORT.md`。

**写入报告头的信息**：`model_name`（评测对象）、`baselines`/`baseline_name`（对比对象）、
`sources`（每个对象的来源文件）、`change_description`（本次改动）。
用户如果说了"我优化了 X"，就要把它写进 `change_description`，
让报告开篇就说清"评的是谁、比的是谁、改了什么"。

**第 4 步 · 读回报告并转述**
打开生成的 `REPORT.md`，向用户**只转述 3–5 条核心结论**（每条都要带具体数值），
并列出产出的文件。不要整篇复述，也不要在转述里加报告里没有的数字。

---

## 输出契约

```
reports/<模型>_<批次>/
  ${window}h_TS_vs_lead.png          时效-技巧曲线（按降水等级分线）
  ${window}h_POD_vs_lead.png
  ${window}h_FAR_vs_lead.png
  ${window}h_BIAS_vs_lead.png        BIAS 带 =1 参考线，不截断
  ${window}h_metrics_by_threshold_<lead>h.png
  ${window}h_TS_heatmap.png          等级 × 时效
  ${window}h_performance_diagram_<lead>h.png
  ${window}h_multi_model_TS_delta_vs_<主模型>.png   多模型对比**画差值**（不画绝对值）
  REPORT.md
```

> 多模型对比为什么画差值：几个模型水平接近时（例如 24h 的 ≥0.1mm 分别是
> 0.529/0.519/0.507/0.504），绝对曲线会重合成一团，谁也看不出差异；
> 差值曲线直接回答"谁在哪一档、哪个时效领先"。

`REPORT.md` 结构固定：**评测对象与对比对象**（模型 + 来源文件 + 可比性提示）→
执行摘要（带数值 + 业务定性）→ 逐等级表（TS 带 🟢🟡🔴 分级、计数与比率同口径）→
逐时效 TS 表 →（可选）与对比模型的表 → 图表索引 → **改进建议** → 口径说明。

---

## 硬性规则

1. **数字只能来自输入表**；推算值（如 `漏报率=1−POD`）要在报告里注明是推算的。
2. **等权平均 ≠ 样本加权**。报告里的"全时效平均"是各时效等权平均，必须保留这句说明。
3. **`BIAS` 是频率偏差**（预报事件数/观测事件数），不是平均误差；不要写成"平均误差"。
4. **缺值不是 0**。`TS` 为 `NaN` 表示无有效配对或无事件，与"评分 0.000"不同，解读要区分。
5. **`n_pairs` 太小或无事件的等级**不参与强弱结论（例如某等级命中为 0，只能说"无技巧"，不能说"技巧最差"）。
6. **跨模型比较要看共同口径**：报告里要交代数据来源、时段、时效范围；口径不一致时明确标注"不可直接比较"。
7. **中文字体缺失要说明**：如果出图脚本提示未找到中文字体，转述给用户并建议安装 Noto Sans CJK。
8. **失败要直说**：数据缺列、目录里没有结果表、只有基准没有新模型——都直接说清楚，不要静默生成空报告。

---

## 常见情形

| 情形 | 处理 |
|---|---|
| 数据里没有 24h 时效 | 不要写死 24h；取数据中最小的时效，并在报告里写明用的是哪个 |
| 只有 6h 窗口的结果 | 阈值口径通常是 `0.1/13/25`，与 24h 的阈值不同，不要混在一起比 |
| 只有单模型、没有基准 | 正常出单模型报告，跳过"与基准对比"一节 |
| 要对比多个模型 | 逐个模型读表 → `plot_multi_model_comparison(model_dfs, metric='TS')`；各模型的降水等级列表可能不同，函数会按共有等级对齐 |
| 集合预报的 TS | 输入的仍是"集合平均场"的 TS（确定性口径）；概率评分（AROC/BSS）是另一张表，不要混进同一张图 |

---

## 参考文档

- 指标定义与读法：`references/evaluation-metrics.md`
- 降水评估口径：`references/precipitation-evaluation.md`
- 诊断规则（样本量门槛、技巧分级、偏差/误差形态判读、建议触发条件）：`references/diagnostic-rules.md`

---

## 维护说明

- **可视化与报告能力都在主仓库里**：`xmetai_evaluation/visualization/`
  （`precipitation_plots.py` 出图、`ts_report.py` 写报告）。
  改能力改主仓库，本 skill 只负责"怎么用、什么时候用"。
- `scripts/` 下的 `unified_report.py`、`unified_report_v2.py`、`precipitation_report.py`
  是早期"多模型对比"方案留下的重叠实现（需要 baseline 目录结构，缺基准会直接失败），
  **不要再用**；对应的能力已由 `create_report(..., baseline_df=...)` 覆盖。
- **维护者**：沈哲文 (szw)
