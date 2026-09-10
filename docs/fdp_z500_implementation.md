# FDP Z500 连续场评测实现

## 概述

实现了 FDP z500 连续场评测的完整流程，支持 RMSE、Bias、ACC 指标。

## 已实现组件

### 1. 数据读取器

- **FengqingReader** (`xmetai_evaluation/io/fengqing_reader.py`)
  - 读取 Fengqing 模型输出（21成员集合预报）
  - 支持 PLEVELS（z500）和 SURFACE（t2m, msl, u10, v10, tp）变量
  - 自动转换 Z500 单位：m²/s² → m（除以 9.80665）
  - 文件格式：`Fengqing_1.0_GLB_PLEVELS_OP25_6HOR_ENS_FCST_YYYYMMDDHH_LLL.nc`

- **CRAReader** (`xmetai_evaluation/io/cra_reader.py`)
  - 读取 CRA 再分析数据（观测真值）
  - 使用 cfgrib 读取 GRIB/GRIB2 文件
  - 支持按 filter_by_keys 精确提取变量
  - 文件格式：
    - 大气：`ART_ATM_GLB_0P25_6HOR_ANAL_YYYYMMDDHH.grib2`
    - 地面：`CRA40LAND_SURFACE_YYYYMMDDHH_GLB_0P25_HOUR_V1_0_0.grib`

### 2. 数据转换

- **Regrid** (`xmetai_evaluation/transforms/regrid.py`)
  - 空间插值：将预报网格插值到观测网格
  - 集合均值计算
  - 纬度余弦权重计算

### 3. 配对器

- **Matcher** (`xmetai_evaluation/pipeline/matcher.py`)
  - 时间对齐：valid_time = init_time + lead_time
  - 空间对齐：双线性插值
  - 集合处理：计算集合均值（连续场评测使用均值）
  - 生成共同有效掩码和权重

### 4. 评测指标

- **RMSE** (`xmetai_evaluation/metrics/rmse.py`)
  - 均方根误差
  - 支持纬度权重
  
- **Bias** (`xmetai_evaluation/metrics/bias.py`)
  - 平均误差（forecast - observation）
  - 正值表示预报偏高，负值表示预报偏低

- **ACC** (`xmetai_evaluation/metrics/acc.py`)
  - 距平相关系数
  - 需要气候态数据（当前版本可用零场占位）

### 5. 主流程

- **FDPContinuousEvaluator** (`xmetai_evaluation/pipeline/fdp_continuous.py`)
  - 统一调度读取、配对、计算、输出
  - 支持配置文件驱动
  - 输出 JSON 格式结果

## 网格差异处理

- **Fengqing 网格**：721×1440（包含两极，lat=-90...90, lon=0...359.75）
- **CRA 网格**：720×1440（网格中心，lat=89.875...-89.875, lon=0.125...359.875）
- **处理方式**：将 Fengqing 双线性插值到 CRA 网格

## 变量映射

| 标准名 | Fengqing 文件变量 | CRA GRIB shortName | 单位转换 |
|--------|-------------------|--------------------| ---------|
| z500   | Z500              | gh (level=500)     | Fengqing: m²/s² → m |
| t2m    | T2M               | 2t (level=2)       | 无 |
| msl    | MSL               | msl                | 无 |
| u10    | U10               | 10u (level=10)     | 无 |
| v10    | V10               | 10v (level=10)     | 无 |
| tp     | TP                | -                  | 无 |

## 使用方法

### 方法 1：配置文件

```yaml
# configs/fdp_continuous_z500.yaml
name: fdp_continuous_z500
description: FDP continuous field evaluation for z500

forecast:
  reader: fengqing
  root: "D:/气象研究/春燕_weather_test_bash/fengqing"
  variables: [z500]
  init_times: ["2026-08-19T00:00:00"]
  lead_times: [24, 48, 72, 96, 120]

observation:
  reader: cra
  root: "D:/气象研究/春燕_weather_test_bash/cra_root"
  variables: [z500]

metrics:
  - name: rmse
    params: {}
  - name: bias
    params: {}
  - name: acc
    params:
      climatology_path: null

output:
  path: "D:/xmetai-evalation/results/fdp_z500_{timestamp}.json"
```

```python
from pathlib import Path
from xmetai_evaluation.pipeline.fdp_continuous import run_evaluation

# 运行评测
result_bundle = run_evaluation(Path("configs/fdp_continuous_z500.yaml"))
```

### 方法 2：编程式

```python
from datetime import datetime
from xmetai_evaluation.pipeline.fdp_continuous import FDPContinuousEvaluator

config = {
    "name": "fdp_z500_test",
    "forecast": {
        "reader": "fengqing",
        "root": "path/to/fengqing",
        "variables": ["z500"],
        "init_times": ["2026-08-19T00:00:00"],
        "lead_times": [24, 48, 72],
    },
    "observation": {
        "reader": "cra",
        "root": "path/to/cra",
        "variables": ["z500"],
    },
    "metrics": [
        {"name": "rmse", "params": {}},
        {"name": "bias", "params": {}},
    ],
    "output": {
        "path": "results/fdp_z500.json",
    },
}

evaluator = FDPContinuousEvaluator(config)
result_bundle = evaluator.run()
```

## 输出格式

```json
{
  "config": { ... },
  "results": [
    {
      "metric_name": "rmse",
      "metric_version": "1.0.0",
      "value": 42.3,
      "status": "success",
      "n_requested": 1036800,
      "n_valid": 1036800,
      "weights_sum": 518400.0,
      "aggregation": "mean_over_samples",
      "warnings": []
    },
    {
      "metric_name": "bias",
      "metric_version": "1.0.0",
      "value": -1.2,
      "status": "success",
      "n_requested": 1036800,
      "n_valid": 1036800,
      "weights_sum": 518400.0,
      "aggregation": "mean_over_samples",
      "warnings": []
    }
  ]
}
```

## 测试

运行单元测试：

```bash
cd D:\xmetai-evalation
pytest tests/integration/test_fdp_z500.py -v
```

注意：部分测试需要真实数据文件，已标记为 skip。

## 技术要点

### 1. 单位转换
- Fengqing Z500 从 m²/s² 转为 m：除以标准重力加速度 9.80665
- CRA gh 单位为 gpm（geopotential meters），即米，无需转换

### 2. 集合处理
- 连续场评测使用集合均值：`forecast.mean(dim="member")`
- 21 个成员平均后与观测对比

### 3. 权重
- 使用纬度余弦权重：`weights = cos(lat * π / 180)`
- 全球评分时考虑纬度圈面积差异

### 4. 时间对齐
- Fengqing: (init_time, lead_time, member, lat, lon)
- CRA: (valid_time, lat, lon)
- 对齐: valid_time = init_time + lead_time

### 5. 空间对齐
- 使用 xarray.interp(method='linear') 双线性插值
- 保留 NaN（不外插）

## 依赖项

新增依赖（需要安装）：
- `cfgrib`: CRA GRIB 文件读取
- `eccodes`: cfgrib 后端
- `pyyaml`: YAML 配置文件解析

```bash
conda install -c conda-forge cfgrib eccodes pyyaml
```

## 待完成事项

1. **气候态数据**
   - 当前 ACC 可运行但使用零场占位
   - 需要提供月气候态数据文件

2. **对比验证**
   - 用原始 `multi_model_verifier_fix.py` 运行相同数据
   - 对比 RMSE/Bias/ACC 数值（相对误差 < 0.1%）

3. **扩展到多变量**
   - 当前实现支持框架
   - 需要确认 SURFACE 文件是否包含 T2M/MSL/U10/V10

4. **性能优化**
   - 大规模数据时的内存管理
   - 并行化处理多个起报时间

## 文件清单

```
xmetai_evaluation/
├── io/
│   ├── fengqing_reader.py         # Fengqing 读取器 ✓
│   └── cra_reader.py              # CRA 读取器 ✓
├── transforms/
│   └── regrid.py                  # 网格插值和集合处理 ✓
├── metrics/
│   ├── bias.py                    # Bias 指标 ✓
│   └── acc.py                     # ACC 指标 ✓
├── pipeline/
│   ├── matcher.py                 # 配对器 ✓
│   └── fdp_continuous.py          # 主流程 ✓
├── configs/
│   └── fdp_continuous_z500.yaml   # z500 配置 ✓
└── tests/
    └── integration/
        └── test_fdp_z500.py       # 集成测试 ✓
```

## 与现有代码的兼容性

- 遵循 README.md 定义的数据契约
- 使用现有的 `DataRequest`, `DataBundle`, `EvaluationBatch` 等核心契约
- 遵循 Metric 接口的 accumulate/merge/finalize 模式
- 与现有 RMSE、Registry、Executor 等组件兼容

## 下一步

1. 在真实数据上运行评测
2. 对比原始参考代码的结果
3. 如果数值一致，验收通过
4. 如果有差异，调试并记录差异原因
