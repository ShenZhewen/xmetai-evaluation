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

# 列出内置流程
xmetai-eval --list-pipelines

# 跑内置任务（数据路径由环境变量提供，见 configs/*.py）
xmetai-eval --config ts_ens_fuxi

# 或指向自定义配置
xmetai-eval --config /path/to/my_eval.py
# 等价写法：python -m xmetai_evaluation --config ts_ens_fuxi
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
│   │   ├── ts_det_fgvp.py          # FuXi 确定性降水分类检验
│   │   ├── ts_ens_fuxi.py          # FuXi 集合降水分类 + 概率评分
│   │   └── fdp_field_scores_fengqing.py  # 要素场检验（RMSE/Bias/ACC）
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
| Reader | `fuxi`、`fuxi_ens`、`fengqing`、`cra`、`station`（`diamond_station` 别名）、`climatology` |
| Transform | `grid_to_station`、`time_window_accumulator`、`ensemble_mean` |
| Metric | `rmse`、`bias`、`acc`、`ts_score`、`ensemble_probability`、`crps`、`spread_error`、`fss`、`activity`、`spectrum` |
| Protocol | `station_valid_time`（插值到站点，按有效时刻配对）、`grid_valid_time`（插值到实况网格） |
| Writer | `csv_long`（始终写出）、`coverage`、`details`、`json`、`categorical_wide`、`probability_wide` |

## 内置流程

| 流程名 | 协议 | 指标 | 说明 |
|---|---|---|---|
| `ts_det` | station_valid_time | `ts_score` | 确定性降水 TS/POD/FAR/频率偏差 |
| `ts_ens` | station_valid_time | `ts_score` + `ensemble_probability` | 集合降水：集合平均 TS + AROC/BS/BSS |
| `fdp_ens_crps` | grid_valid_time | `crps` + `spread_error` | 集合 CRPS / 离散度-误差比 |
| `fdp_field_scores` | grid_valid_time | `rmse` + `bias` + `acc` | 要素检验（ACC 需气候态参考） |
| `fdp_precip_ts` | station_valid_time | `ts_score` | 中国区站点降水 TS/Bias |
| `fdp_precip_fss` | grid_valid_time | `fss` | 降水邻域分数技巧评分 |
| `fdp_activity_spectrum` | grid_valid_time | `activity` + `spectrum` | Z500 活跃度比 / 功率谱 |

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
    pipeline="ts_ens",
    forecast_reader={"type": "fuxi_ens", "root_dir": "/data/fuxi_ens", "variable": "tp", "step_hours": 6.0},
    observation_reader={"type": "station", "root_dir": "/data/station", "variable": "precipitation"},
    start_date="20250101",
    end_date="20251231",
    output_dir="evaluation_results/my_eval",
)
```

内置配置用环境变量覆盖数据路径与时段，例如 `ts_ens_fuxi` 支持 `FUXI_ENS_OUTPUT`、`STATION_OBS`、`STATION_LIST`、`WINDOW_HOURS`、`LEAD_TIMES`、`START_DATE`、`END_DATE`、`EVAL_OUTPUT`、`WRITERS`。

## 如何扩展

- **新增模型 / 数据集**：已支持的文件布局 → 写一份数据源配置即可；全新格式 → 在 `io/` 实现 Reader，再到 `components.py` 注册。
- **新增指标**：在 `metrics/` 实现 `Metric`（`accumulate` / `merge` / `finalize`），到 `components.py` 注册。
- **新增流程**：在 `pipeline/pipelines.py` 加一个 `PipelineTemplate`。
- **新增数据源/算法** = 加注册项；**新增评测** = 加配置，代码不动。

## 测试

```bash
pytest
```
