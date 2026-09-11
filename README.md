# xmetai-evaluation

气象模型**离线评测**框架：输入已经推理完成的预报产品 + 观测/再分析/气候态，输出统一的结构化指标与报告。

**只做评测，不做推理**：不加载模型权重、不训练、不调度推理。

```
xmetai-inference → 预报产品文件 → xmetai-evaluation → 评测结果与报告
```

## 能力边界

- 输入预报：NetCDF 格点场（FuXi 确定性 / FuXi 集合 / Fengqing）。
- 输入观测/参考：Diamond 站点观测、CRA40 再分析、CRA 气候态。
- 产品类型：确定性场、集合成员场、概率产品。
- 输出：`scores.csv` 长表、分类/概率宽表、JSON 快照、覆盖率；可选图件。

## 快速开始

```bash
pip install -e .

# 看有哪些评测功能
xmetai-eval --list-pipelines

# 开始评测（配置可以是内置名，也可以是 .py 路径；数据路径由环境变量提供）
xmetai-eval --config weather_ts_ens_fuxi
```

## 目录结构

```text
xmetai-evaluation/
├── pyproject.toml                  # 打包 + CLI 入口 xmetai-eval
├── xmetai_evaluation/
│   ├── cli.py                      # 统一入口：加载配置 → 交给 Runner
│   ├── components.py               # 内置组件注册（按名字查表，不分类型分支）
│   ├── logging_util.py
│   ├── configs/                    # EvalConfig + 任务配置（部分用环境变量覆盖路径）
│   │   ├── base.py                 # EvalConfig 定义 + load_config()
│   │   ├── weather_ts_det_fgvp.py       # FuXi 确定性降水分类检验
│   │   ├── weather_ts_ens_fuxi.py       # FuXi 集合降水：24h TS + 6h 概率两段一趟跑完
│   │   ├── weather_field_scores_fuxi.py # FuXi 确定性连续量检验（RMSE/ACC/FA/谱）
│   │   ├── weather_ens_crps_fuxi.py     # FuXi 集合连续评分（CRPS/Spread-Error）
│   │   └── fdp_field_scores_fengqing.py # 要素场检验（RMSE/Bias/ACC）
│   ├── core/                       # contracts / errors / registry / variables
│   ├── io/                         # gridded / layouts / station_reader / climatology_reader / netcdf_reader / base
│   ├── transforms/                 # interpolation / temporal / regrid
│   ├── metrics/                    # rmse / bias / acc / categorical / probabilistic / ensemble / spatial / specialized
│   ├── pipeline/                   # runner(唯一执行内核) / spec / pipelines / protocols / matcher
│   ├── results/                    # store / table
│   └── visualization/              # precipitation_plots / ts_report
├── tests/                          # unit + integration
├── ref/                            # 参考实现（只读档案，不入库、不参与运行）
├── evaluation_results/             # 评测产物（不入库）
└── reports/                        # 报告图件（不入库）
```

## 执行流程

一条链路，无第二套循环：

```
cli → load_config(EvalConfig) → PipelineSpec(流程模板+数据)
    → Runner(唯一样本循环) → ResultStore 落盘
```

- **流程模板**（`pipeline/pipelines.py`）只声明「怎么算」：协议 + 变换链 + 指标 + 输出视图。
- **配置**（`configs/*.py`）只声明「算什么」：数据在哪、评哪段时间。
- **组件注册**（`components.py`）：新增数据源/指标只加注册项，不动 Runner。

## 已注册组件

| 类型 | 注册名 |
|---|---|
| Reader | `fuxi`、`fuxi_ens`、`fengqing`、`cra`、`station`（`diamond_station` 别名）、`climatology`、`ref_probability` |
| Transform | `grid_to_station`、`time_window_accumulator`、`ensemble_mean` |
| Metric | `rmse`、`bias`、`acc`、`acc_uncentered`、`ts_score`、`ensemble_probability`、`crps`、`spread_error`、`fss`、`activity`、`spectrum`、`zonal_spectrum` |
| Protocol | `station_valid_time`（插值到站点，按有效时刻配对）、`grid_valid_time`（插值到实况网格） |
| Writer | `csv_long`（始终写出）、`coverage`、`details`、`json`、`categorical_wide`、`probability_wide` |

## 评测能力清单

每行是一个可直接跑的评测能力（流程名 `pipeline`），输入/输出文件相对 `output_dir`。

流程与配置按三大业务块统一前缀命名：`fdp_`（业务天气评测，参考 `ref/fdp`）、`weather_`（天气模型验证，参考 `ref/tiqnqi` 两个库）、`clim_`（气候，参考 `ref/qihou`，待落地）。fdp 与 weather 的连续量/集合指标有重叠，属正常——两者是不同业务线，共用同一套 metric 实现。

「评估指标」一列给的是 **`scores.csv` 的 `metric` 列取值**（即过滤长表用的那个字符串），
括号里是该指标在表里变化的坐标轴（阈值 / 时效 / 窗口 / 邻域窗口）。
标注「明细」的是**不进 `scores.csv`** 的诊断量，含义见下面「评估指标说明」。

| 评测能力 | 输入数据 | 输出 CSV | 评估指标（`metric` 列取值） |
|---|---|---|---|
| **确定性降水分类检验**<br>`weather_ts_det` | 确定性格点降水预报（`fuxi`，tp）+ Diamond 站点降水观测 | `scores.csv`、`diagnostics/categorical_wide.csv` | `ts`、`pod`、`far`、`miss_rate`、`frequency_bias`<br>逐 阈值（0.1/10/25/50/100 mm）× 24h 时效 |
| **集合降水分类检验（24h）**<br>`weather_ts_ens` | 集合格点降水预报（`fuxi_ens`，tp）+ Diamond 站点降水观测 | `scores.csv`、`diagnostics/categorical_wide.csv` | `ts`、`pod`、`far`、`miss_rate`、`frequency_bias`（集合平均场）<br>逐 阈值（0.1/10/25/50/100 mm）× 24h 时效；明细 `hits`/`misses`/`false_alarms`/`correct_negatives`/`n_pairs` |
| **集合降水概率评分（6h）**<br>`weather_ts_ens_prob` | 集合格点降水预报（`fuxi_ens`，tp）+ Diamond 站点降水观测 + 气候概率参考 `ref_probability` | `scores.csv`、`diagnostics/probability_wide.csv` | `aroc`、`bs`、`bss`（逐成员概率，不做集合平均）<br>逐 阈值（0.1/4/13/25 mm）× 6h 时效；明细 `BS_ref`/`base_rate`/`n_points` |
| **确定性连续量检验**<br>`weather_field_scores` | 格点场预报（`fuxi`，z500 等）+ 格点实况 + 气候态（ACC/活跃度必需） | `scores.csv` | `rmse`、`acc`、`activity_ratio`、`activity_forecast`、`spectrum_power_ratio`（纬向谱）<br>明细 `OBS_ACTIVITY`/`FC_OBS_BIAS` + 逐波数谱曲线 |
| **集合连续评分**<br>`weather_ens_crps` | 集合格点场预报（`fuxi_ens`）+ 格点实况 | `scores.csv` | `crps`、`spread`、`rmse`（集合平均场）、`spread_error_ratio` |
| **集合场检验**<br>`weather_ens_field_scores` | 集合格点场预报（`fuxi_ens`）+ 格点实况（ERA5）+ 气候态（ACC/活跃度必需） | `scores.csv` | `rmse`、`crps`、`acc`、`activity_ratio`、`activity_forecast`、`spectrum_power_ratio`<br>明细同 `weather_field_scores` |
| **集合连续评分**<br>`fdp_ens_crps` | 集合格点场预报（`fengqing`）+ CRA40 再分析实况 | `scores.csv` | `crps`、`spread`、`rmse`（集合平均场）、`spread_error_ratio` |
| **要素场检验**<br>`fdp_field_scores` | 格点场预报（`fengqing`，z500 等）+ CRA40 实况 + 气候态（可选，ACC 必需） | `scores.csv` | `rmse`、`bias`、`acc` |
| **中国区站点降水检验**<br>`fdp_precip_ts` | 格点降水预报 + Diamond 站点降水观测（中国区，cos 纬度加权） | `scores.csv`、`diagnostics/categorical_wide.csv` | `ts`、`pod`、`far`、`miss_rate`、`frequency_bias`<br>逐 阈值（0.1/13/25 mm）× 6h 时效（UTC 对齐，窗口不要求观测完整） |
| **降水空间检验**<br>`fdp_precip_fss` | 格点降水预报 + 格点降水实况（CRA / CMPAS） | `scores.csv` | `fss`（阈值 13 mm × 邻域窗口 1/3/5/15/31/63，每个组合一行）<br>明细 `window`/`n_points` |
| **活跃度比 / 功率谱**<br>`fdp_activity_spectrum` | 格点场预报（z500）+ 格点实况 + 气候态（活跃度比必需） | `scores.csv` | `activity_ratio`、`activity_forecast`、`spectrum_power_ratio`（二维谱，不减纬向均值）<br>明细 `OBS_ACTIVITY`/`FC_OBS_BIAS` + 逐波数谱曲线 |

内置配置里 `weather_field_scores_fuxi` 与 `fdp_field_scores_fengqing` 另外声明了 `json` writer，
会多写一份 `scores.json`；那是配置的选择，不属于流程模板的产出。

### 评估指标说明

分类检验（降水阈值）：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `ts` | TS（Threat Score，即 CSI）：命中占「命中+漏报+空报」的比例 | `hits / (hits + misses + false_alarms)` | 0~1，越大越好 |
| `pod` | 命中率：实况发生的事件里被预报到的比例 | `hits / (hits + misses)` | 0~1，越大越好；空报多也能拿高分，需与 `far` 同看 |
| `far` | 空报率：预报的事件里实况没发生的比例 | `false_alarms / (hits + false_alarms)` | 0~1，越小越好 |
| `miss_rate` | 漏报率 | `1 − POD` | 0~1，越小越好 |
| `frequency_bias` | 频率偏差：预报事件数 / 实况事件数 | `(hits + false_alarms) / (hits + misses)` | 1 为无偏，>1 空报偏多、<1 漏报偏多 |

概率评分（集合成员，超越式口径 `x ≥ 阈值`）：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `aroc` | 概率排序能力（ROC 曲线下面积） | 以「成员中超过阈值的比例」为预报概率画 ROC，再积分；由概率直方图直接算，不落 ROC 点列 | 0.5 = 无技巧，1 = 完美排序 |
| `bs` | Brier 评分：概率预报的均方误差 | `mean((p − o)²)`，`p` = 成员超越频率，`o ∈ {0,1}` | 0 最好，单位同概率² |
| `bss` | Brier 技巧评分：相对气候基准的技巧 | `1 − BS / BS_ref` | >0 好于气候基准，1 = 完美，<0 不如气候 |
| `bs_ref`（明细） | 参考 BS | 有 `ref_probability` 参考时 `mean((p_clim − o)²)`；缺参考时降级为样本气候频率 `r(1−r)`——**降级后 BSS 与原版不可比** | 同上 |
| `base_rate`（明细） | 事件样本频率 | 该档样本里实况达到阈值的比例 | 用于判断样本是否失衡 |
| `n_points`（明细） | 参与该档评分的配对数 | — | — |

连续量 / 集合评分：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `rmse` | 均方根误差 | `sqrt(Σw(f − o)² / Σw)`，`w` 为区域权重 | 越小越好，单位同变量 |
| `bias` | 平均误差（系统性偏差） | `Σw(f − o) / Σw` | 0 为无偏，>0 预报偏大 |
| `acc` | 距平相关系数：预报距平场与实况距平场的（加权）相关 | 距平 = 场 − 气候态；默认 `centered`（距今平再减域加权均值，经典皮尔逊），FDP/WeatherBench2 的 uncentered 口径用 `acc_uncentered` | −1~1，越大越好；**缺气候态会用零场兜底、结果无意义**（行状态标 `partial`） |
| `crps` | 连续排序概率评分：集合分布与实况的整体差异 | 闭式解，逐点只用有限成员，缺测成员不参与；不做非负截断 | 越小越好 |
| `spread` | 集合离散度 | `sqrt(Σᵢ(mᵢ − m̄)² / (M − 1))` | 与 `rmse` 同量级才有意义 |
| `spread_error_ratio` | 离散度-误差比 | `SPREAD / RMSE(集合平均场)` | ≈1 标定良好，<1 过度自信，>1 欠自信 |
| `activity_ratio` | 活跃度比：预报的距平变化幅度相对实况 | `std(预报距平) / std(实况距平)`，面积加权 | <1 偏平滑（系统性偏弱），>1 偏噪；**缺气候态时无意义** |
| `activity_forecast` | 预报距平标准差 | 同上分子 | 诊断用；实况侧 `OBS_ACTIVITY` 与两者之差 `FC_OBS_BIAS` 在明细表 |
| `fss` | 邻域分数技巧评分：邻域平滑后再比「有/无」 | `1 − Σ(p_f − p_o)² / Σ(p_f² + p_o²)`，`p` 为邻域内超过阈值的格点占比 | 越大越好；窗口越大越接近随机基准，看技巧随尺度衰减 |
| `spectrum_power_ratio` | 总功率比：预报能量相对实况 | 预报功率谱总量 / 实况功率谱总量；逐波数曲线在明细表 | 1 表示总能量不偏；单看总量会掩盖分布失真，需与谱曲线同看 |

几个读表要点：

- **长表只放标量。** 列联表计数（`hits`/`misses`/`false_alarms`/`correct_negatives`/`n_pairs`）、
  `BS_ref`/`base_rate`/`n_points`、`OBS_ACTIVITY`/`FC_OBS_BIAS`、逐波数谱曲线都不占
  `scores.csv` 的列，只进明细表 `diagnostics/scores_detail.csv`（声明 `details` writer 时写出）；
  其中前两组会被 `categorical_wide` / `probability_wide` 各自透视成表头列。
- **同名不同口径靠 `aggregation` 区分。** 例如集合场检验里 `spread_error` 顺带输出的 `rmse`
  是**集合平均场的域加权**口径（`area_weighted`），`rmse` 指标是逐样本平均口径（`mean_over_samples`）。
- **空 `value` 不是 0。** NaN/Inf 一律写空字符串，该档有没有数看 `status` 列
  （`success` / `partial` / `no_valid_data` / `undefined`）。

一条流程模板只有**一个时间窗口**（`station_valid_time` 的 `window_hours` 同时决定观测累积长度和有效时效的筛选），所以集合降水检验按口径拆成了两条：24h 的 `weather_ts_ens` 出 TS 系列，6h 的 `weather_ts_ens_prob` 出概率评分。`weather_ts_ens_fuxi` 配置用 `pipeline=["weather_ts_ens", "weather_ts_ens_prob"]` 一条命令跑完两段，结果合并落同一个 `output_dir`（`scores.csv` 里靠 `window_h` 列区分，两个宽表各取自己那一段）。

> 台风路径/强度检验（`ref/tiqnqi/xmetai_model_verification_xu/run_tc.py`：台风中心诊断 + babj 实况配对 + 路径/强度误差）尚未吸收进框架，属待办 gap。

输出文件口径：

| 文件 | 说明 |
|---|---|
| `scores.csv` | 统一评分长表（始终写出），每行 = 一个 变量×指标×阈值×时效×样本 的评分 |
| `coverage.csv` | 请求/有效样本覆盖率（显式声明 `coverage` writer 时写出） |
| `diagnostics/scores_detail.csv` | 诊断明细：列联表计数、`BS_ref`/`base_rate`/`n_points`、逐波数谱曲线、`OBS_ACTIVITY`/`FC_OBS_BIAS` |
| `diagnostics/categorical_wide.csv` | 分类检验宽表（阈值 × 时效：TS/POD/FAR/漏报率/BIAS + `hits`/`misses`/`false_alarms`/`n_pairs` 计数） |
| `diagnostics/probability_wide.csv` | 概率评分宽表（阈值 × 时效：AROC/BS/BSS + `BS_ref`/`base_rate`/`n_points`） |
| `scores.json` | 评分 JSON 快照 |

当前已接好的内置任务配置（`configs/`）：`weather_ts_det_fgvp`（FGVP 确定性降水）、`weather_ts_ens_fuxi`（FuXi 集合降水，24h TS + 6h 概率两段一趟跑完）、`weather_field_scores_fuxi`（FuXi 确定性连续量）、`weather_ens_crps_fuxi`（FuXi 集合连续评分）、`fdp_field_scores_fengqing`（Fengqing 要素场）。

## 配置

配置是一个 `EvalConfig` 实例（`configs/base.py`），核心字段：

| 字段 | 含义 |
|---|---|
| `pipeline` | 走哪套流程模板（必填）；写成列表则按顺序各跑一段，结果合并落同一个 `output_dir` |
| `forecast_reader` / `observation_reader` | 数据源：`{"type": <reader>, "root_dir": ..., "variable": ...}` |
| `reference_reader` | 参考场：气候态（ACC/活跃度需要）或气候概率（BSS 需要）。配置里给了、但某段的指标用不上时不会构建 |
| `start_date` / `end_date` | 评测时段（`YYYYMMDD` 或 `YYYYMMDDHH`） |
| `output_dir` | 结果输出目录 |
| `transform_options` / `metric_options` | 覆盖模板里的变换/指标参数 |
| `writers` | 覆盖模板的输出视图（默认 `csv_long`） |

最小示例：

```python
from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="my_eval",
    description="FuXi 集合降水分类检验",
    pipeline="weather_ts_ens",
    forecast_reader={"type": "fuxi_ens", "root_dir": "/data/fuxi_ens", "variable": "tp", "step_hours": 6.0},
    observation_reader={"type": "station", "root_dir": "/data/station", "variable": "precipitation"},
    start_date="20250101",
    end_date="20251231",
    output_dir="evaluation_results/my_eval",
)
```

要注意 `transform_options` / `metric_options` / `options` 是**各段共用**的：配置里写 `{"time_window_accumulator": {"window_hours": 6}}` 会把每一段的窗口都改成 6。窗口属于"怎么算"，写在模板里（`pipeline/pipelines.py`）。多段的完整例子见 `configs/weather_ts_ens_fuxi.py`。

内置配置用环境变量覆盖数据路径与时段，例如 `weather_field_scores_fuxi` / `weather_ens_crps_fuxi` 支持 `FUXI_OUTPUT` / `FUXI_ENS_OUTPUT`、`CRA_ROOT`、`CRA_CLI_ROOT`（前者 ACC/活跃度需要）、`START_DATE`、`END_DATE`、`EVAL_OUTPUT`。

## 如何扩展

- **新增模型 / 数据集**：已支持的文件布局 → 写一份数据源配置即可；全新格式 → 在 `io/` 实现 Reader，再到 `components.py` 注册。
- **新增指标**：在 `metrics/` 实现 `Metric`（`accumulate` / `merge` / `finalize`），到 `components.py` 注册。
- **新增流程**：在 `pipeline/pipelines.py` 加一个 `PipelineTemplate`。
- **新增数据源/算法** = 加注册项；**新增评测** = 加配置，代码不动。

## 测试

```bash
pytest
```
