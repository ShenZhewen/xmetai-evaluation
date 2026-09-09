# 使用新框架运行 FuXi 降水分类评测

## 已实现的功能

✅ **Change 7**: FuXi Reader + Diamond Station Reader  
✅ **Change 9**: 网格插值 + 时间累积  
✅ **Change 10**: TS 分类指标  
✅ **101 个测试全部通过**

---

## 快速开始示例

### 方式 1：Python API（推荐用于调试）

```python
from datetime import datetime
from pathlib import Path
import xarray as xr

# 导入新框架模块
from xmetai_evaluation.io.fuxi_reader import FuXiReader, FuXiCatalog
from xmetai_evaluation.io.station_reader import DiamondStationReader, DiamondStationCatalog
from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator
from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator
from xmetai_evaluation.metrics.categorical import TSScore
from xmetai_evaluation.core.contracts import DataRequest, EvaluationBatch

# ==================== 1. 读取预报数据 ====================
print("读取 FuXi 预报...")
fuxi_reader, fuxi_catalog = FuXiReader.with_catalog(
    root_dir=Path("/workspace/data/shenzw/fuxi_single_output"),
    source_id="fuxi",
    step_hours=6.0,
)

# 请求数据
forecast_request = DataRequest(
    source_id="fuxi",
    variables=["tp"],  # 小写
    init_times=[
        datetime(2025, 1, 1, 0),
        datetime(2025, 1, 2, 0),
        # 添加更多起报时间...
    ],
)

# 发现文件
forecast_index = fuxi_catalog.discover(forecast_request)
print(f"发现 {len(forecast_index.available[0])} 个起报时间")

# 读取数据
forecast_bundle = fuxi_reader.read(forecast_request, forecast_index)
print(f"预报数据形状: {forecast_bundle.payload['tp'].shape}")
# 输出：(n_init, n_lead, 721, 1440)

# ==================== 2. 读取站点观测 ====================
print("\n读取站点观测...")
station_reader, station_catalog = DiamondStationReader.with_catalog(
    station_dir=Path("/workspace/data/worm/r0/2025"),
    source_id="diamond_obs",
    station_whitelist=None,  # 或提供站点列表
)

obs_request = DataRequest(
    source_id="diamond_obs",
    variables=["precipitation"],
)

obs_index = station_catalog.discover(obs_request)
print(f"发现 {len(obs_index.available)} 个站点文件")

obs_bundle = station_reader.read(obs_request, obs_index)
print(f"观测数据形状: {obs_bundle.payload['precipitation'].shape}")
# 输出：(n_time, n_station)

# ==================== 3. 插值到站点 ====================
print("\n插值网格到站点...")
interpolator = GridToStationInterpolator(method="bilinear")

# 提取预报降水场（选择一个起报时间和时效）
forecast_field = forecast_bundle.payload['tp'].isel(init_time=0, lead_time=0)  # 第一个起报，第一个时效
station_lats = obs_bundle.payload.coords['lat'].values
station_lons = obs_bundle.payload.coords['lon'].values

forecast_at_stations = interpolator.transform(
    forecast_field, 
    station_lats, 
    station_lons
)
print(f"插值后形状: {forecast_at_stations.shape}")  # (n_station,)

# ==================== 4. 时间累积（24h 窗口）====================
print("\n时间累积...")
# 对预报做 24h 累积
accumulator = TimeWindowAccumulator(window_hours=24.0, time_dim="lead_time")
forecast_24h = accumulator.transform(forecast_bundle.payload['tp'].isel(init_time=0))
print(f"24h 累积后: {forecast_24h.shape}")

# 对观测也需要做累积（这里需要额外的时间对齐逻辑，简化示例）

# ==================== 5. 计算 TS 指标 ====================
print("\n计算 TS 指标...")

# 构建 EvaluationBatch（需要对齐的预报和观测）
# 这里简化：假设已经对齐
valid_mask = xr.DataArray(
    ~forecast_at_stations.isnull(),
    dims=forecast_at_stations.dims,
    coords=forecast_at_stations.coords,
)

batch = EvaluationBatch(
    forecast=forecast_at_stations,
    observation=obs_bundle.payload['precipitation'].isel(time=0),  # 示例
    sample_keys=[{"station": i} for i in range(len(station_lats))],
    valid_mask=valid_mask,
    alignment={"method": "bilinear_interp", "time_matched": True},
)

# 计算 TS
ts_metric = TSScore(thresholds=[
    ("≥0.1", 0.1),
    ("≥10", 10.0),
    ("≥25", 25.0),
    ("≥50", 50.0),
])

result = ts_metric.compute(batch)

# 输出结果
print(f"\n状态: {result.status}")
print(f"有效样本数: {result.n_valid}")

for threshold_name, metrics in result.value.items():
    print(f"\n{threshold_name}:")
    print(f"  TS:   {metrics['TS']:.4f}")
    print(f"  POD:  {metrics['POD']:.4f}")
    print(f"  FAR:  {metrics['FAR']:.4f}")
    print(f"  BIAS: {metrics['BIAS']:.4f}")
    print(f"  Hits: {metrics['hits']}, Misses: {metrics['misses']}, False Alarms: {metrics['false_alarms']}")
```

---

## 方式 2：批量处理脚本

创建文件 `run_fuxi_ts_eval.py`:

```python
#!/usr/bin/env python
"""
FuXi 降水 TS 评测完整流程

用法：
    python run_fuxi_ts_eval.py --forecast-root /path/to/fuxi_output \\
                               --station-dir /path/to/stations \\
                               --init-dates 20250101 20250102 \\
                               --output results/fuxi_ts.csv
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

from xmetai_evaluation.io.fuxi_reader import FuXiReader, FuXiCatalog
from xmetai_evaluation.io.station_reader import DiamondStationReader, DiamondStationCatalog
from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator
from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator
from xmetai_evaluation.metrics.categorical import TSScore
from xmetai_evaluation.core.contracts import DataRequest, EvaluationBatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-root", required=True, help="FuXi 输出根目录")
    parser.add_argument("--station-dir", required=True, help="站点观测目录")
    parser.add_argument("--init-dates", nargs="+", required=True, help="起报日期列表 YYYYMMDD")
    parser.add_argument("--window", type=float, default=24.0, help="累积窗口（小时）")
    parser.add_argument("--output", required=True, help="输出 CSV 文件")
    args = parser.parse_args()

    # 解析起报日期
    init_times = [datetime.strptime(d, "%Y%m%d") for d in args.init_dates]

    # 初始化 Reader
    fuxi_reader, fuxi_catalog = FuXiReader.with_catalog(
        root_dir=Path(args.forecast_root),
        source_id="fuxi",
    )
    station_reader, station_catalog = DiamondStationReader.with_catalog(
        station_dir=Path(args.station_dir),
        source_id="diamond_obs",
    )

    # 读取数据
    print("读取预报数据...")
    forecast_request = DataRequest(source_id="fuxi", variables=["tp"], init_times=init_times)
    forecast_index = fuxi_catalog.discover(forecast_request)
    forecast_bundle = fuxi_reader.read(forecast_request, forecast_index)

    print("读取站点观测...")
    obs_request = DataRequest(source_id="diamond_obs", variables=["precipitation"])
    obs_index = station_catalog.discover(obs_request)
    obs_bundle = station_reader.read(obs_request, obs_index)

    # Transform + 计算指标
    interpolator = GridToStationInterpolator(method="bilinear")
    accumulator = TimeWindowAccumulator(window_hours=args.window, time_dim="lead_time")
    ts_metric = TSScore(thresholds=[
        ("≥0.1", 0.1), ("≥10", 10.0), ("≥25", 25.0), ("≥50", 50.0),
    ])

    # 循环处理每个起报时间和时效
    all_results = []
    for init_idx in range(len(init_times)):
        forecast_24h = accumulator.transform(forecast_bundle.payload['tp'].isel(init_time=init_idx))
        
        for lead_idx in range(len(forecast_24h.lead_time)):
            # 插值 + 对齐 + 计算
            # （这里需要完整的时间匹配逻辑）
            # ...
            
            # 累积结果
            pass  # 实际实现

    # 输出 CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
```

---

## 与原代码的对比

| 功能 | 原代码 (`run_categorical.py`) | 新框架 |
|------|-------------------------------|--------|
| **数据读取** | `ClassDataset` 硬编码逻辑 | `FuXiReader` + `DiamondStationReader`（可扩展） |
| **插值** | `interp_to_stations` 函数 | `GridToStationInterpolator`（可配置方法） |
| **累积** | `accumulate_window` 函数 | `TimeWindowAccumulator`（通用） |
| **指标计算** | `TSContingency` 类 | `TSScore` Metric（支持分块合并） |
| **并行** | `ProcessPoolExecutor` 手动管理 | Pipeline 自动管理（未来扩展） |
| **缓存** | 无 | Pipeline 自动缓存 |
| **错误处理** | 简单 try-catch | 分类错误 + 上下文追踪 |
| **可扩展性** | 修改源码 | 注册新组件 |

---

## 下一步工作

### 立即可做（完善 Change 7/9/10）

1. **时间对齐逻辑**  
   需要实现预报有效时刻与观测时刻的匹配：
   ```python
   # xmetai_evaluation/transforms/alignment.py
   class TemporalAligner:
       def align_forecast_observation(forecast, observation, ...):
           # 根据 init_time + lead_time 计算 valid_time
           # 匹配观测时间
           pass
   ```

2. **完整的端到端脚本**  
   基于上面的示例，编写完整的 `run_fuxi_ts_eval.py`

3. **集成测试**  
   用服务器上的真实数据测试完整流程

### 后续扩展（Change 11+）

4. **集合预报支持** - AROC/BS/BSS 指标
5. **更多指标** - ACC, MAE, Bias, FSS...
6. **并行执行** - Pipeline 并行处理多个起报时间
7. **报告生成** - 自动生成 CSV/HTML 报告

---

## 测试状态

```bash
# 当前测试覆盖
$ python -m pytest tests/unit/ -v
===================== 101 passed, 1 warning in 1.68s ======================

# 测试分布
- Core (41): contracts, errors, registry, config
- Metrics (19): RMSE, TS categorical
- Pipeline (18): executor, state machine
- IO (14): NetCDF reader
- Transforms (9): interpolation, accumulation
```

---

## 如何在服务器上运行

### 步骤 1：安装依赖

```bash
cd /workspace/szwCode/xmetai-evalation
pip install -e .
pip install scipy  # 插值需要
```

### 步骤 2：运行示例

```bash
# 交互式测试
python
>>> from xmetai_evaluation.io.fuxi_reader import FuXiReader, FuXiCatalog
>>> # 按照上面的示例代码运行
```

### 步骤 3：对比结果

运行原代码和新框架，对比输出的 TS 指标是否一致。

---

**新框架已准备就绪，可以开始在服务器上测试了！** 🚀
