---
name: xmetai-evaluation
description: >
  气象模型评测结果的分析与出图报告。当用户给出评测结果目录——形如
  `outputs/results/weather_ts_single_aifs`、`outputs/results/weather_rmse_single_fgvp`
  这样的目录，里面是 `ts_*.csv` + `*_meta.json`——并希望
  "生成报告""画图""分析这次评测""帮我对比这几个模型""看看强降水表现"
  "看看 RMSE / ACC / 活跃度 / 谱""和基准/别的模型比一比"
  "以 XX 为主模型对比其他模型""我这次改了 XX 模块，看看有没有用"
  "分析出现在模型的问题""给点改进意见"
  "报告要像论文的实验部分""图下面要有图注""每一节要有总结"时使用。
  也覆盖台风路径/强度检验（"台风路径误差""看看台风预报得怎么样""对比几个模型的台风"）——
  结果目录里是 `tc<编号>_<起报时刻>.csv` + `_meta.json`。
  覆盖三条链路：降水分类检验（TS 系列，含集合）、确定性连续场 / RMSE 批次家族、台风路径/强度。
  **能力身份从产物自己认，不靠目录名**；**主模型可以指定，不指定则按目录清单推断**。
  产出：时效曲线、阈值对比、TS 热力图、性能图、跨时效箱线图、谱曲线、多模型差值图，
  以及一份**按论文实验章节组织**的 Markdown 报告——图表就地编号（`图 N` / `表 N`）、
  图下有图注、每节末尾有一段带具体数值的定量+定性小结，
  并给出**有归因支撑**的改进建议（哪些模块有用、哪些没用、代价是什么）。
metadata:
  version: 4.0
  author: 沈哲文 (szw)
---

# XMetAI 评测结果：出图与报告

## 这个 skill 干什么

把**评测结果**变成**图表 + 一份能读的报告**。

输入是评测结果目录（或 CSV），输出是 `*.png` 加一份 `REPORT.md`——
报告不只是"数据堆"，而是**论文实验章节**：每节"引导句 → 图表 → 小结"，
末尾给**有归因支撑**的改进建议。

## 不干什么

- **不重算指标**。本 skill 只消费已有的评测结果；要重算就去跑评测流程，不要在这里另起一套算法。
- **不猜数据**。缺列、缺文件、缺时效时**如实说明**，不要用默认值或估算值把报告"补全"。
- **不编造结论**。报告里的每个数字都必须来自输入表；没有依据的判断不写。
- **不凭目录名认能力**。见「能力路由」。

---

## 触发场景

用户说了类似下面的话，就应该用本 skill：

- "这是 FGVP 的 TS 结果，帮我画个图 / 写份报告"
- "结果在 `outputs/results/weather_ts_single_aifs`，出个报告"
- "这几个目录你都看看，该合的合、该比的比"
- "**以 FGVP 为主模型**，对比一下 FuXi 和 AIFS"
- "我这次把 TP 参数化换了，你看看有没有用 / 哪个模块起作用了"
- "分析一下这次降水评测，强降水表现怎么样，有什么问题"
- "对比一下新模型和 FuXi 的 TS"
- "连续场评测跑完了，看看 RMSE 随时效涨得怎么样"
- "这个 ACC 衰减正常吗""活跃度比接近 1 吗""谱是不是失真了"

---

## 输入：结果目录

### 长什么样

结果目录是**扁平的**——一次评测的产物直接摊在目录根下：

```
outputs/results/<目录名>/
  <name>.csv            结果主表（TS 系列：13 列，见下）
  <name>_meta.json      这次跑批的口径 + 结果摘要
  <capability>.json     可选；把结果和它的产物路径绑在一起
```

`<name>` 由跑批配置的 `name` 决定（`configs/*.py` 里那行
`# name 决定落盘文件名：ts_single_fgvp.csv + single_fgvp_meta.json`）。
**`<name>` 和目录名不保证一致**，别拿目录名拼文件名——**在目录里 glob `*.csv`
比拼名字可靠**。

### `_meta.json` 里有什么

它是**这次跑批的完整口径**，报告头的"评测对象与数据来源"几乎全部取自这里：

| 键 | 含义 | 典型值 |
|---|---|---|
| `root` | 预报源根目录 | `/workspace/data/shenzw/fgvp_output` |
| `var` | 被评变量 | `TP` |
| `step_h` / `init_hour` | 时步 / 起报时刻 | `6.0` / `0.0` |
| `init_dates` | 起报日列表 | 365 个 `YYYYMMDD` |
| `mode` / `n_members` | **确定性还是集合** | `deterministic` / `1` |
| `station_dir` / `station_list` / `n_station_list` | 站点集 | `10285` |
| `tz_shift_h` / `interp` | 时区偏移 / 插值方式 | `8.0` / `bilinear` |
| `tp_scale` | **降水量的单位换算系数** | `1.0` 或 `1000.0` |
| `ts_windows` / `aroc_windows` | 出 TS 的窗口 / 出概率评分的窗口 | `[24.0]` / `[]` |
| `windows.<w>.thresholds` | 该窗口的阈值表 | `[["0.1",0.1],["10",10.0],…]` |
| `windows.<w>.aroc_thresholds` | 概率评分的超越式阈值表 | `["≥0.1",0.1],…` |
| `windows.<w>.leads` | 每个时效的 `skipped` | `{"24":{"skipped":1},…}` |
| `ref_result` / `bss_reference` | 气候概率参考 / BSS 基准的定义 | `null` / `样本气候频率 r(1−r)` |
| `elapsed_s` / `grid` | 耗时 / 网格 | `2541.44` / `721×1440` |

> **`tp_scale` 是单位陷阱**：`1000.0` 表示源数据是 m、乘 1000 换成 mm；
> `1.0` 表示源数据本来就是 mm。**两个值都不影响最终口径（都归到 mm）**，
> 但它在 meta 里留下的差异是**真实的**——说明两个模型的源数据单位不同。
> 看到它别当成"口径不一致所以不能比"。

### 结果主表（TS 系列）

```csv
window_h,lead_h,grade,threshold_mm,hits,misses,false_alarms,n_pairs,TS,POD,FAR,漏报率,BIAS
24,24,≥0.1,0.1,1053429,45439,929900,3510379,0.519246,0.958649,0.468858,0.0413507,1.80488
```

必需列：`lead_h`、`grade`、`TS`。常见可选列：`window_h`、`threshold_mm`、
`hits`、`misses`、`false_alarms`、`n_pairs`、`POD`、`FAR`、`漏报率`（或 `miss_rate`）、`BIAS`。

- 多出来的列（如 `run_id`）忽略即可；
- `漏报率` 缺失时用 `1 − POD` 推导；
- **必需列缺失 → 停下来告诉用户缺什么**，不要继续。

> ⚠️ **`grade` 列的写法在不同批次之间不一致**，实拍：
> 有的批次是 `≥0.1` / `≥10`，有的是裸的 `0.1` / `10`。
> **归类和对齐时必须先归一化**（去掉 `≥`、按 `threshold_mm` 排序），
> 否则同一批数据会被当成两个不同的等级集。
>
> ⚠️ **等级集合也可能不同**：实测 AIFS 有 6 档（到 `≥250`）、FGVP 只有 5 档
> （到 `≥100`），行数 90 vs 75。**按共有等级对齐**，缺的那档在表里记 `—`，
> 不要补 0。

### `reports` 与 `outputs` 的分工

| 目录 | 放什么 |
|---|---|
| `outputs/results/<目录>/` | **评测结果**（输入）——csv + meta.json |
| `outputs/.temp/` | 中间产物（staging、fixture、试跑），**不是交付物** |
| `report/<名字>/` | **生成的报告**（输出）——`REPORT.md` + `*.png` |

---

## 归类：先认这批结果是什么

目录名一律是 `weather_<能力>_<single|ens>_<模型>`，模型名**在后**：

```
weather_ts_single_fgvp      ← 主模型（带 <capability>.json）
weather_ts_single_fuxi      ← 对照模型
weather_ts_single_aifs      ← 对照模型
weather_ts_single_fengqing  ← 对照模型
weather_ts_single_fgvp_tp   ← 带"改了哪个变量"的标记
weather_ts_ens_fuxi         ← 集合
weather_rmse_single_fgvp    ← RMSE 批次家族
```

**但目录名仍然只是线索，不要靠它判断这是哪条链路**——历史遗留的目录未必
守这个格式（存量里存在过 `weather_ts_aifs_single` 这种模型名在前的写法）。
真正判能力的依据，按优先级：

| # | 判据 | 怎么用 |
|---|---|---|
| 1 | `<capability>.json` | 文件名就是能力名（如 `weather_ts_det.json`） |
| 2 | `_meta.json` 的字段组合 | `mode` + `n_members` + `aroc_windows` + `ts_windows`，见下表 |
| 3 | `<name>.csv` 的表头 | 13 列 TS 表 vs 23/24 列长表 `scores.csv` vs 其他 |
| 4 | 目录名 | **只作最后佐证，不作为判据** |

### 归类对照表

| 看到什么 | 归到 | 走哪条路 |
|---|---|---|
| 13 列 TS 表 + `mode: deterministic`（或 `n_members: 1`） | **确定性 TS** | `weather_ts_det` → `weather_ts_single.md` |
| 同一套表 + `mode: ensemble` / `n_members > 1` | **集合 TS** | `weather_ts_ens` → `weather_ts_ens.md` |
| 同上，且 `aroc_windows` 非空、目录里有 `*aroc_bss*.csv` | **概率评分也有** | 集合骨架的第六节终于有数据了（见 `precipitation-evaluation.md` §2.1） |
| `scores.csv`（长表）+ `manifest.json` + `diagnostics/`，`resolved_config.pipeline` 是 `weather_field_scores` | **RMSE 批次家族**（连续场） | 按支线选入口：`_single`→`generate_det_report.py`、`_ens`→`generate_ens_report.py`、`_wave`→`generate_wave_report.py`，见 `rmse-batch-evaluation.md`。**想出一份单模型的连续场报告**走 `generate_report.py`（吃同样形状，只吃一份） |
| `_meta.json` 里有 `babj` / `tcid` / `forecast_type` | **台风路径** | `forecast_type: "det"` → `weather_typhoon_single.md`；`"ens"` → `weather_typhoon_ens.md`（**渲染器都还没做**，见下） |
| 一堆积分清单一（`mni_csv` 那种） | **结构演示/假数据** | 用户没点名就别当结果用 |

### 归类之后：报清单，别闷头跑

一批目录里**经常有几条链路混在一起**（TS 的 + RMSE 的）。
归完类先把清单报给用户：

```
results/ 下 9 个目录：
  TS 确定性   weather_ts_single_fgvp / weather_ts_single_fuxi /
              weather_ts_single_aifs / weather_ts_single_fengqing / weather_ts_single_fgvp_tp
  TS 集合     weather_ts_ens_fuxi（空）
  RMSE 批次   weather_rmse_single_fgvp / _fuxi / _ens_fuxi（都空）
要按哪组出报告？
```

**空目录要点名**，不要静默跳过——用户以为跑了、结果是空的，比报错更糟。

---

## 工作流（智能体自己执行）

**用户不会自己跑 python，命令由你执行。** 六步：

### 第 1 步 · 归类

按上一节认这批目录是什么、有哪几条链路、哪些是空的。**先报清单再动手**。

### 第 2 步 · 定主模型

- **用户点名了**（"以 FGVP 为主"）→ 就用它，不要自作主张换；
- **用户没说** → 按顺序推断，**并把推断结果告诉用户**：
  1. 目录名里带明显主模型标记的（`weather_ts_single_fgvp` 这种最新的一次运行）；
  2. 有 `<capability>.json` 的那个（它被显式标记过）；
  3. 仍然不定 → 反问用户。

主模型决定报告里**所有"本模型"的说法、差值图的纵轴方向、
以及"改进建议"是对谁提的**——选错了整份报告都偏。

### 第 3 步 · 核对口径（跑之前必做）

| 检查 | 不合格怎么办 |
|---|---|
| `window_h` 一致？ | 不一致**不能画进同一张图**，要么分开出报告，要么在报告里点明 |
| `lead_h` 集合对得上？ | 同上 |
| `grade` 集合对得上？ | 按**共有等级**对齐，缺的记 `—`（`≥250` 这种常缺） |
| `grade` 列写法一致？ | 归一化后再说（`≥0.1` vs `0.1`） |
| `tp_scale` / `var` / `interp` / `tz_shift_h` 一致？ | 说明源数据口径不同，**报告里要写进"可比性"声明** |

### 第 4 步 · 跑

```bash
python skills/xmetai-evaluation/scripts/generate_report.py <主模型产物目录> \
  --result report/<报告名> --model <主模型简称> \
  --compare "FuXi=outputs/results/weather_ts_single_fuxi/ts_fuxi_2025.csv" \
  --compare "AIFS=outputs/results/weather_ts_single_aifs/ts_aifs_2025.csv"
```

- **对比模型给的是裸 CSV，不是目录**——`--compare` 本来就吃 CSV：
  `--compare "名字=CSV路径"`，可重复；
- **主模型给的是产物目录**（要有 `manifest.json` 与 `diagnostics/categorical_wide.csv`）。
  ⚠️ **今天的结果目录还喂不进去**，见下面的「现状」；
- 用户说了本次改了什么 → 加 `--change "..."`，它会进报告头，**并且是差异归因的起点**
  （见 `references/model-diff-analysis.md`）；
- 想换报告标题 → 加 `--model "FGVP-v2"`。

> **现状：主模型入口还有一步没接上。**
> `outputs/results/<目录>/` 里是 `ts_*.csv` + `*_meta.json`，
> 而 `generate_report.py` 的主模型入口只认 `manifest.json` +
> `diagnostics/categorical_wide.csv`。**这两样在本仓库里没有任何流程会写出来**
> ——今天那份 `report/weather_ts_single_fgvp_fuxi_aifs/` 是先把结果
> 手工 staging 到 `outputs/.temp/report_staging/` 才跑通的。
> **约定的修法是**：`visualization/report.py` 的 TS 分支加一条回退——
> 找不到 `manifest.json` 时读目录里的 `<name>_meta.json` 认口径、
> `<name>.csv` 当结果表。**该分支尚未实现**；在那之前，
> 主模型要么手工 staging，要么如实告诉用户"这一步还得手工接"，
> **不要假装一键能跑**。

### 第 5 步 · 读回报告并转述

打开生成的 `REPORT.md`，向用户**只转述 3–5 条核心结论**（每条都要带具体数值），
并列出产出的文件。不要整篇复述，也不要在转述里加报告里没有的数字。

**用户给了改动说明时**，还要回答那个他真正关心的问题：
**"改的这个模块有没有用"**——结论要带代价（哪一档变好了、哪一档变差了）。
判断框架见 `references/model-diff-analysis.md`。

### 第 6 步 · 失败就直说

命令报错时**把错误原样转述**，不要自己绕过它去手工拼流程：

| 报错 | 含义 |
|---|---|
| `... 的报告模板还没做` | 这条能力还没有出图与报告代码，如实告诉用户 |
| `... categorical_wide.csv 不存在` | 评测时没写宽表 writer，让用户改配置重跑 |
| `manifest.json 不存在` | 用户给的目录不是产物目录（**当前结果目录必然撞这条**，见第 4 步的现状） |
| `出图需要 matplotlib` | 环境没装：`pip install -e .[viz]` |

---

## 能力路由（先认能力，再选模板）

权威判据是产物里的 `resolved_config.pipeline`（`manifest.json`），
没有 `resolved_config` 的老产物退路是 `protocol_id` + `manifest["metrics"]` 的组合；
**结果目录那种扁平布局**则按「归类」一节的字段判据。
**这些判断已经在代码里实现了**，直接调（见工作流第 4 步）：

```bash
python skills/xmetai-evaluation/scripts/generate_report.py <产物目录> --result reports/<名字>
```

它会自己认能力、派发到对应模板。认不出、或者认出来但**报告模板还没做**时，
它会明确报错——**拿错模板比报错危险得多**，所以不要绕过它自己拼。

| 流程 `pipeline` | 覆盖 | 报告模板 |
|---|---|---|
| `weather_ts_det` / `fdp_precip_ts` | 确定性站点降水分类检验：TS / POD / FAR / 频率偏差 | ✅ [`assets/templates/weather_ts_single.md`](./assets/templates/weather_ts_single.md) |
| `weather_ts_ens` | 集合站点降水分类检验：同上，TS 算在集合平均场上；另加一节概率评分 AROC / BSS | ✅ [`assets/templates/weather_ts_ens.md`](./assets/templates/weather_ts_ens.md) |
| `weather_field_scores` | 确定性连续场：RMSE / ACC / 活跃度 / 纬向谱 | ⚠️ 能力与渲染器都在，但 `assets/templates/` 里的格式契约已删 |
| `typhoon`（`forecast_type: det`） | 台风路径/强度：诊断中心位置 + 中心强度，对比 babj 实况 | ⚠️ 骨架 [`weather_typhoon_single.md`](./assets/templates/weather_typhoon_single.md)，**渲染器还没做** |
| `typhoon`（`forecast_type: ens`） | 同上，集合口径（方案 A/B）+ 离散度 | ⚠️ 骨架 [`weather_typhoon_ens.md`](./assets/templates/weather_typhoon_ens.md)，**渲染器还没做** |
| 其余流程 | CRPS、FSS、概率评分…… | ⛔ 还没做，会明确报错 |

> **RMSE 批次家族（`weather_rmse_single_<模型>` / `weather_rmse_ens_<模型>`）走的是多模型对比入口，
> 不是因为形状不同**：它们的产物就是标准评估产物目录（`scores.csv` + `manifest.json` +
> `diagnostics/`），`pipeline` 也正是上表里的 `weather_field_scores`，
> 所以 `generate_report.py` 同样认得出、同样能出——**只是它一次只吃一份，出单模型报告**。
> 要多模型横向对比就各走各的独立入口——`scripts/` 下
> `generate_det_report.py`（单成员）/ `generate_ens_report.py`（集合）/
> `generate_wave_report.py`（谱检验补充），都是 `--model NAME=目录` 给 N 份（≥2）。
> 骨架分别是 [`weather_rmse_single.md`](./assets/templates/weather_rmse_single.md)（单成员）、
> [`weather_rmse_ens.md`](./assets/templates/weather_rmse_ens.md)（集合）与
> [`weather_rmse_wave.md`](./assets/templates/weather_rmse_wave.md)（纬向 FFT + 球谐带功率谱补充分析）。
> **三份的渲染器都能跑，但入口不能混用**：集合与单成员的产物目录**形状完全一样**，
> 集合产物喂给 `generate_det_report.py` 不会报错，出的却是 `_single` 的章节口径。
> 产物形状、派生指标（相对 RMSE / FA 偏差 / 频谱对数 RMS；集合与谱检验用 **ln**、单成员用 **log10**）
> 与必须一致的口径参数见 `references/rmse-batch-evaluation.md`。

> 三条 TS 流程写的产物表结构相同，走的是**同一个渲染器** `ts_report.py`。
> 骨架分成两份，唯一的原因是集合那条**多一类指标**（AROC / BSS），
> 不是因为流程不同——**模板按家族分、不按流程分**这条原则没变。

---

## 输出契约

**A. 降水分类检验**（TS 系列）：

```
report/<模型>_<批次>/
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
评测对象与对比对象（模型 + 来源文件 + 可比性提示；有改动说明时一并写进报告头）
一、结论摘要           带数值的规则化诊断 + 业务定性
二、逐降水等级表现     表 1 + 图 1（阈值对比）→ 小结
三、TS 随时效变化      图 2（曲线）图 3（热力图）+ 表 2 → 小结
四、误差形态与偏差结构  图 4–7（POD / FAR / BIAS / 性能图）→ 小结
五、跨时效分布         表 3（分位数）+ 图 8（箱线图）→ 小结
六、与对比模型的比较   （可选）表 4 + 图 9（差值）→ 小结
七/六、改进建议        **每条都挂在一条已确认的归因上**，不是通用清单
八/七、口径与注意事项
```

**集合那份（`weather_ts_ens`）在第五、六节之间多一节**「概率评分（AROC / BSS）」，
后面的章节号整体后移一位（A 到八、B/C 到九），图与表也各多一个号。

**TS 系列没有"图表索引"节**——图都就地带着图注出现，再列一遍只是重复。
（连续场反过来：图集中在末尾的「图表」节，正文里不重复出现。）

> **"结构固定"是被测试守着的**：[`assets/templates/weather_ts_single.md`](./assets/templates/weather_ts_single.md)
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

> **"结构固定"原先是被测试守着的**：`tests/unit/visualization/test_field.py` 实跑一遍
> 渲染器，拿真报告去对 `assets/templates/field_scores.md` 那份骨架。
> **骨架已删，这个真值来源没有了**，测试再红也说不清是渲染器变了还是骨架没了。
> 最后那节「图表」**只在出了图时才有**——一张图都没有时它连标题带编号一起消失，
> 「口径与注意事项」顶上来当第六节，不留空档。

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
13. **归因必须有对照**。没有改动说明就只能报差异、不能归因；
    **"改了 X 所以变好了"这句话，得先有"如果 X 有用，哪个数字该往哪动"的预期**。
    判定门槛与代价检查见 `references/model-diff-analysis.md`。
14. **差值 < 0.005 一律视为并列**，不报领先者；名次用**竞赛排名**（并列同名次）。
15. **建议要挂在归因上**。"改进建议"一节里的每一条，都要能指回前面某一节的
    具体数值；没有数字支撑的通用建议（"建议加大模型""建议换注意力机制"）不写。

---

## 常见情形

| 情形 | 处理 |
|---|---|
| 结果目录里 `<name>.csv` 和目录名对不上 | **glob `*.csv`**，别拿目录名拼文件名 |
| 一批目录里 TS 和 RMSE 混在一起 | 先归类、报清单，问用户要按哪组出报告；不要一锅端 |
| 目录是空的 | **点名**，不要静默跳过——用户会以为跑了 |
| 一批目录里同一个模型有两份 | 逐格比 `n_pairs`：完全一样就是同一次评测的两次运行，别同时放进对比图 |
| `grade` 列有的是 `≥0.1`、有的是 `0.1` | 归一化后再对齐；不归一化会被当成两个等级集 |
| 各模型等级集合不同（有的到 `≥250`、有的到 `≥100`） | 按**共有等级**对齐，缺的记 `—`，**不补 0** |
| 用户没指定主模型 | 按目录名标记 → `<capability>.json` → 反问，三步走；推断结果要告诉用户 |
| 用户给了改动说明 | 走归因流程（`references/model-diff-analysis.md`）：预期特征 → 找特征 → 给代价 |
| 数据里没有 24h 时效 | 不要写死 24h；取数据中最小的时效，并在报告里写明用的是哪个 |
| 只有 6h 窗口的结果 | 阈值口径通常是 `0.1/13/25`，与 24h 的阈值不同，不要混在一起比 |
| 只有单模型、没有基准 | 正常出单模型报告，跳过对比章节（箱线图照出，一个量级一个箱体） |
| 只有长表 `scores.csv`、没有宽表 | **不合成**——两份口径混用比报错危险。在评测配置里加上 `writers=["csv_long", "categorical_wide"]` 重跑 |
| 两个目录看着是同一个模型 | 逐格比 `n_pairs`——完全一样就是同一次评测的两次运行，别同时放进对比图（会画出两个重合的箱子） |
| 集合预报 | **两份产物**：一份是**集合平均场**的 TS（确定性口径），一份是概率评分（AROC / BSS）。报告走 `weather_ts_ens.md`，两套数**不要混图**（见 `precipitation-evaluation.md` §3.1） |
| 集合跑法没给 `--ref` | `BS_ref` 整列为空，BSS 算不出来——如实写"无气候基准"，别拿别的窗口顶上 |
| 连续场里某变量没有某个指标 | 先看是不是**逐变量路由**没给它（`manifest` 的 `metric_options` 里每个指标带着自己的 `variables` 列表）；真缺数是结果里整行都没有，不是 `value` 为空 |
| 连续场目录的 `scores.csv` 里 `level`/`region`/`threshold`/`window_h` 全空 | 这是 `grid_valid_time` 协议的默认口径（只有 `variable`/`sample_unit`/`unit`），**不是缺数据** |
| 要判断谱是"总量正常但分布失真" | 只用 `spectrum_power_ratio` 看不出来（它是全波数求和后的比值）；用 `--spectrum-variable` / `--spectrum-lead` 出谱曲线看高波数段 |

---

## 参考文档

**报告长什么样**（格式契约，加新能力时照着写）：

- 降水分类检验：`assets/templates/weather_ts_single.md`（确定性两条）、
  `assets/templates/weather_ts_ens.md`（集合，多一节概率评分）
- RMSE 批次家族：`assets/templates/weather_rmse_single.md`（单成员）、
  `assets/templates/weather_rmse_ens.md`（集合）、`assets/templates/weather_rmse_wave.md`（谱检验补充分析）
- 台风路径/强度：`assets/templates/weather_typhoon_single.md`（单成员）、
  `assets/templates/weather_typhoon_ens.md`（集合，方案 A/B + 离散度）

> `assets/templates/` 现在只剩上面七份。**确定性连续场（`weather_field_scores`）
> 的骨架与模板目录的 `README.md` 都已删除**——能力本身、渲染器、判读规则
> （`references/field-evaluation.md`）都还在，只是格式契约没有真值来源了。

**数字怎么判读**：

- 指标定义与读法：`references/evaluation-metrics.md`
- 降水评估口径：`references/precipitation-evaluation.md`
- 降水诊断规则（样本量门槛、技巧分级、偏差/误差形态判读、建议触发条件）：`references/diagnostic-rules.md`
- 连续场指标语义与判读：`references/field-evaluation.md`
- RMSE 批次家族（`_single` / `_ens` / `_wave`）的归档形状、派生指标与判读：`references/rmse-batch-evaluation.md`
- **模型差异归因**（改动起没起作用、代价是什么、症状词典）：`references/model-diff-analysis.md`

---

## 维护说明

- **渲染逻辑在主仓库**：`visualization/`
  （`precipitation_plots.py` + `ts_report.py` 管 TS 链路，
  `field_plots.py` + `field_report.py` 管连续场链路，
  `det_report.py` 管 RMSE 批次家族，`report.py` 负责"认能力 + 派发"）。
  改能力改主仓库——那里能被 pytest 覆盖。
  规则与代码的对应关系写在 references 里，**冲突时以代码为准**。
- **`scripts/generate_report.py` 只做入口**（插 `sys.path` → 调 `build_report` → 打印产物）。
  **不要往里加判断逻辑**：一旦它开始自己认能力、自己拼流程，
  就又变成一份会跟主仓库漂移的副本。
- **`assets/templates/` 是格式契约**：一份报告骨架 = **一份完整的报告，只是数值位置
  用 `{…}` 空着**（不是说明文档，也不是模板引擎）。它不参与渲染，但被
  `tests/unit/visualization/test_ts_report.py` 实跑比对守着——
  改了渲染器的章节结构就必须同步改骨架，否则测试红。
- **已知缺口**（写在这里免得下次重新发现）：
  1. 主模型入口认不得 `outputs/results/<目录>/` 的扁平布局（见工作流第 4 步）；
  2. **集合报告出不出来，取决于数据而不取决于渲染器**：`generate_ens_report.py`
     要求各模型有 ≥2 天共同日期，否则点名报错。实测三份集合归档
     （FuXi 20250102–20251216 / AIFS 20250702–20250728 / FGVP 20250102–20250103）
     **三方交集为 0**，目前只能出 FuXi↔AIFS 那份（27 天，低于 30 天样本下限，
     报告自己会标「方向性参考」）；
  3. 概率评分（AROC / BSS）取数与绘图那一层还没做，`weather_ts_ens_prob` 标着 `[BLOCKED]`；
  4. 球谐带功率**已经有渲染器**（`wave_report` 的 §5.1–5.3），数据取自长表里
     `spherical_bands` 展开的三行，靠 **`group` 列**区分频带。
     但**现存归档的 `scores.csv` 缺 `group` 这一列**（契约 24 列、实拍 23 列），
     5 个频带的行混在一起认不出来，这三节会走 `SPHERICAL_ABSENT` 标「本批未出」
     ——**重跑评测即可**，不是报告端的问题。见 `references/rmse-batch-evaluation.md` §5.6；
  5. **台风两份骨架的渲染器还没做**（`weather_typhoon_single` / `_ens`）。
     而且 `weather_typhoon_ens.md` 里的列名**全部是从 `core/tc_ref.py` 的集合分支
     读出来的，不是实拍**——第一次真跑之后要拿真实表头回来核一遍。
- `agents/` 目前是空的。
- **本目录是源，`.claude/skills/` 下的是副本**（Windows 建符号链接要管理员权限，
  所以用复制）。改完这里要重新复制过去，见 `skills/README.md`。
- **维护者**：沈哲文 (szw)
