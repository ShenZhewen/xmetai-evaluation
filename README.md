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
│   ├── configs/                    # EvalConfig + 任务配置（环境变量驱动）
│   │   ├── base.py                 # EvalConfig 定义 + load_config()
│   │   ├── weather_ts_det_fgvp.py       # FuXi 确定性降水分类检验
│   │   ├── weather_ts_ens_fuxi.py       # FuXi 集合降水分类 + 概率评分
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

| 评测能力 | 输入数据 | 输出 CSV | 评估指标 |
|---|---|---|---|
| **确定性降水分类检验**<br>`weather_ts_det` | 确定性格点降水预报（`fuxi`，tp）+ Diamond 站点降水观测 | `scores.csv`、`diagnostics/categorical_wide.csv` | TS / POD / FAR / 漏报率 / 频率偏差（BIAS） |
| **集合降水分类 + 概率检验**<br>`weather_ts_ens` | 集合格点降水预报（`fuxi_ens`，tp）+ Diamond 站点降水观测（可选 BSS 外部气候概率 `ref_probability`） | `scores.csv`、`diagnostics/categorical_wide.csv`、`diagnostics/probability_wide.csv` | 集合平均 TS / POD / FAR + 逐成员概率 AROC / BS / BSS |
| **确定性连续量检验**<br>`weather_field_scores` | 格点场预报（`fuxi`，z500 等）+ 格点实况 + 气候态（ACC/活跃度必需） | `scores.csv` | RMSE / ACC / 预报活跃度（FA）+ 纬向功率谱 |
| **集合连续评分**<br>`weather_ens_crps` | 集合格点场预报（`fuxi_ens`）+ 格点实况 | `scores.csv` | CRPS / Spread / 集合平均 RMSE / Spread-Error 比 |
| **集合连续评分**<br>`fdp_ens_crps` | 集合格点场预报（`fengqing`）+ CRA40 再分析实况 | `scores.csv` | CRPS / Spread / 集合平均 RMSE / Spread-Error 比 |
| **要素场检验**<br>`fdp_field_scores` | 格点场预报（`fengqing`，z500 等）+ CRA40 实况 + 气候态（可选，ACC 必需） | `scores.csv`、`scores.json` | RMSE / Bias / ACC |
| **中国区站点降水检验**<br>`fdp_precip_ts` | 格点降水预报 + Diamond 站点降水观测（中国区，cos 纬度加权） | `scores.csv`、`diagnostics/categorical_wide.csv` | TS / 频率偏差（BIAS） |
| **降水空间检验**<br>`fdp_precip_fss` | 格点降水预报 + 格点降水实况（CRA / CMPAS） | `scores.csv` | FSS（多邻域窗口） |
| **活跃度比 / 功率谱**<br>`fdp_activity_spectrum` | 格点场预报（z500）+ 格点实况 + 气候态 | `scores.csv` | 活跃度比（AR / FC / OBS / BIAS）+ 功率谱（谱曲线 + 功率比） |

> 台风路径/强度检验（`ref/tiqnqi/xmetai_model_verification_xu/run_tc.py`：台风中心诊断 + babj 实况配对 + 路径/强度误差）尚未吸收进框架，属待办 gap。

输出文件口径：

| 文件 | 说明 |
|---|---|
| `scores.csv` | 统一评分长表（始终写出），每行 = 一个 变量×指标×阈值×时效×样本 的评分 |
| `coverage.csv` | 请求/有效样本覆盖率（显式声明 `coverage` writer 时写出） |
| `diagnostics/scores_detail.csv` | 列联表计数等诊断明细 |
| `diagnostics/categorical_wide.csv` | 分类检验宽表（阈值 × 时效：TS/POD/FAR/BIAS + hits/misses/false_alarms） |
| `diagnostics/probability_wide.csv` | 概率评分宽表（阈值 × 时效：AROC/BS/BSS + base_rate） |
| `scores.json` | 评分 JSON 快照 |

当前已接好的内置任务配置（`configs/`）：`weather_ts_det_fgvp`（FGVP 确定性降水）、`weather_ts_ens_fuxi`（FuXi 集合降水）、`weather_field_scores_fuxi`（FuXi 确定性连续量）、`weather_ens_crps_fuxi`（FuXi 集合连续评分）、`fdp_field_scores_fengqing`（Fengqing 要素场）。

## 配置

配置是一个 `EvalConfig` 实例（`configs/base.py`），核心字段：

| 字段 | 含义 |
|---|---|
| `pipeline` | 走哪套流程模板（必填） |
| `forecast_reader` / `observation_reader` | 数据源：`{"type": <reader>, "root_dir": ..., "variable": ...}` |
| `reference_reader` | 可选参考场（气候态），ACC/活跃度/功率谱等距平类指标需要 |
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

内置配置用环境变量覆盖数据路径与时段，例如 `weather_ts_ens_fuxi` 支持 `FUXI_ENS_OUTPUT`、`STATION_OBS`、`STATION_LIST`、`BSS_REF`（BSS 外部气候概率目录，仅 `WINDOW_HOURS=6` 生效）、`WINDOW_HOURS`、`LEAD_TIMES`、`START_DATE`、`END_DATE`、`EVAL_OUTPUT`、`WRITERS`；`weather_field_scores_fuxi` / `weather_ens_crps_fuxi` 支持 `FUXI_OUTPUT` / `FUXI_ENS_OUTPUT`、`CRA_ROOT`、`CRA_CLI_ROOT`（前者 ACC/活跃度需要）、`START_DATE`、`END_DATE`、`EVAL_OUTPUT`。

## 如何扩展

- **新增模型 / 数据集**：已支持的文件布局 → 写一份数据源配置即可；全新格式 → 在 `io/` 实现 Reader，再到 `components.py` 注册。
- **新增指标**：在 `metrics/` 实现 `Metric`（`accumulate` / `merge` / `finalize`），到 `components.py` 注册。
- **新增流程**：在 `pipeline/pipelines.py` 加一个 `PipelineTemplate`。
- **新增数据源/算法** = 加注册项；**新增评测** = 加配置，代码不动。

## 测试

```bash
pytest
```
