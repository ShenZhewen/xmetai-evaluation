# FDP 连续场评测实现计划

## 目标

实现 FDP 连续场评测（RMSE / Bias / ACC），支持变量：z500 / t2m / msl / u10 / v10

**验证标准**：用我们的框架代码和原始参考代码分别评测同一批数据，输出指标数值一致即为成功。

## 当前数据契约（已确认）

### Fengqing 预报

**文件结构**：
```
fengqing/YYYYMMDD/
    Fengqing_1.0_GLB_PLEVELS_OP25_6HOR_ENS_FCST_YYYYMMDDHH_LLL.nc
    Fengqing_1.0_GLB_SURFACE_OP25_6HOR_ENS_FCST_YYYYMMDDHH_LLL.nc
```

**PLEVELS 文件**：
- 变量：Z500
- 维度：(member=21, level=1, time=1, dtime=1, lat=721, lon=1440)
- Z500 单位：m²/s²（需要除以 9.80665 转为 m）
- 网格：lat=-90...90, lon=0...359.75

**SURFACE 文件（当前样例）**：
- 变量：TP
- 维度：(member=21, level=1, time=1, dtime=1, lat=721, lon=1440)
- 单位：mm

**注意**：当前样例 SURFACE 文件只有 TP，没有 T2M/MSL/U10/V10。先实现 z500 评测闭环，为后续表面变量预留接口。

### CRA 实况

**文件结构**：
```
cra_root/YYYYMMDD/
    ART_ATM_GLB_0P25_6HOR_ANAL_YYYYMMDDHH.grib2  （大气变量）
    CRA40LAND_SURFACE_YYYYMMDDHH_GLB_0P25_HOUR_V1_0_0.grib  （地面变量）
```

**变量映射**（已确认）：
- z500: shortName='gh', typeOfLevel='isobaricInhPa', level=500, units='gpm'（geopotential meters，即米）
- t2m: shortName='2t', typeOfLevel='heightAboveGround', level=2, units='K'
- msl: shortName='msl', typeOfLevel='meanSea', units='Pa'
- u10: shortName='10u', typeOfLevel='heightAboveGround', level=10, units='m s**-1'
- v10: shortName='10v', typeOfLevel='heightAboveGround', level=10, units='m s**-1'

**网格**：
- 维度：(lat=720, lon=1440)
- 网格：lat=89.875...-89.875, lon=0.125...359.875
- 分辨率：0.25°

**网格差异**：
- Fengqing: 721×1440（包含两极）
- CRA: 720×1440（网格中心）
- 需要将 Fengqing 插值到 CRA 网格

## 实现文件清单

### 1. Fengqing Reader 和 Catalog
**文件**：`xmetai_evaluation/io/fengqing_reader.py`

**功能**：
- `FengqingCatalog`: 扫描 `root/YYYYMMDD/` 目录，解析文件名，生成 DataIndex
- `FengqingReader`: 读取 NetCDF，处理集合维度，转换单位，输出 DataBundle

**关键逻辑**：
- 文件名解析正则：`Fengqing_1.0_GLB_(PLEVELS|SURFACE)_OP25_6HOR_ENS_FCST_(\d{10})_(\d{3}).nc`
- init_time 从文件名提取（YYYYMMDDHH）
- lead_time 从文件名提取（小时）
- 读取时合并 member 维度，重构为 (init_time, lead_time, member, lat, lon)
- Z500 单位转换：value / 9.80665
- 变量标准化映射：Z500 → z500
- 支持任意日期目录（允许部分日期缺失）

### 2. CRA Reader 和 Catalog
**文件**：`xmetai_evaluation/io/cra_reader.py`

**功能**：
- `CRACatalog`: 扫描 `cra_root/YYYYMMDD/` 目录，按变量分类文件（ATM/LAND）
- `CRAReader`: 使用 cfgrib 按 filter_by_keys 读取指定变量

**关键逻辑**：
- 每个变量需要独立的 filter_by_keys
- z500: `{'shortName': 'gh', 'typeOfLevel': 'isobaricInhPa', 'level': 500}`
- t2m: `{'shortName': '2t'}`
- msl: `{'shortName': 'msl'}`
- u10: `{'shortName': '10u'}`
- v10: `{'shortName': '10v'}`
- 读取后统一坐标名：latitude→lat, longitude→lon
- 输出 DataBundle，kind=DataKind.OBSERVATION

### 3. 空间插值 Transform
**文件**：`xmetai_evaluation/transforms/grid_regrid.py`

**功能**：
- 将 Fengqing 721×1440 网格插值到 CRA 720×1440 网格
- 使用 xarray.interp(method='linear')
- 处理经度范围一致性

### 4. Matcher
**文件**：`xmetai_evaluation/pipeline/matcher.py`

**功能**：
- 对齐 Fengqing 和 CRA 的 valid_time
- Fengqing valid_time = init_time + lead_time
- CRA valid_time 从文件名或 time 坐标读取
- 调用空间插值将预报网格对齐到实况网格
- 计算集合均值（连续场评测使用均值）
- 生成 EvaluationBatch，包含：
  - forecast: 集合均值后的预报
  - observation: 实况
  - valid_mask: 有效数据掩码
  - weights: 纬度余弦权重
  - sample_keys: 记录 init_time, lead_time, valid_time

### 5. Bias Metric
**文件**：`xmetai_evaluation/metrics/bias.py`

**功能**：
- 实现 Bias = mean(forecast - observation)
- 参考 RMSE 实现 accumulate / merge / finalize 模式
- 支持纬度权重

### 6. ACC Metric
**文件**：`xmetai_evaluation/metrics/acc.py`

**功能**：
- 实现 ACC（Anomaly Correlation Coefficient）
- ACC = corr(forecast_anomaly, obs_anomaly)
- anomaly = field - climatology
- 需要气候态数据（从原始脚本中读取或另外提供）
- 按 FDP 参考实现，ACC 只对 z500 计算

### 7. 评测主流程
**文件**：`xmetai_evaluation/pipeline/fdp_continuous.py`

**功能**：
- 编排完整评测流程：
  1. FengqingCatalog.discover() → 预报文件索引
  2. CRACatalog.discover() → 实况文件索引
  3. FengqingReader.read() → 预报 DataBundle
  4. CRAReader.read() → 实况 DataBundle
  5. Matcher.match() → EvaluationBatch
  6. 对每个变量：
     - RMSE.compute()
     - Bias.compute()
     - ACC.compute()（仅 z500）
  7. 输出结果到 JSON

### 8. 配置文件
**文件**：`configs/fdp_continuous_z500.yaml`

```yaml
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
      climatology_path: "path/to/climatology.nc"

output:
  path: "results/fdp_z500_{timestamp}.json"
```

## 实现顺序

### 第1步：FengqingReader 和 CRAReader（只读 z500）
- 实现 `fengqing_reader.py`
- 实现 `cra_reader.py`
- 单元测试：读取样例文件，验证维度、单位、坐标

### 第2步：空间插值 Transform
- 实现 `grid_regrid.py`
- 单元测试：插值前后网格点数正确

### 第3步：Matcher
- 实现 `matcher.py`
- 单元测试：valid_time 对齐、集合均值、权重生成

### 第4步：Bias 和 ACC Metric
- 实现 `bias.py`
- 实现 `acc.py`（暂时跳过气候态，先用零场测试）
- 单元测试：accumulate/merge/finalize 一致性

### 第5步：评测主流程
- 实现 `fdp_continuous.py`
- 集成测试：端到端运行一次 z500 评测

### 第6步：对比验证
- 用我们的框架代码运行评测
- 用原始 `multi_model_verifier_fix.py` 运行评测
- 对比 RMSE / Bias / ACC 数值
- 误差容忍度：相对误差 < 0.1%

### 第7步：扩展到五变量（如果数据可用）
- 检查是否有 T2M/MSL/U10/V10 数据
- 如果有，扩展 Reader 和配置
- 如果没有，文档说明当前只验证了 z500

## 关键技术点

### 集合预报处理
连续场评测使用集合均值：
```python
forecast_mean = forecast.mean(dim="member")
```

### 纬度权重
全球评分使用余弦纬度权重：
```python
weights = np.cos(np.deg2rad(lat))
```

### Z500 单位转换
Fengqing Z500 从 m²/s² 转为 m：
```python
z500_m = z500_m2s2 / 9.80665
```

### ACC 气候态
原始脚本从外部文件加载月气候态：
```python
# 从 /mnt/petrelfs/xxx/gh_500hPa_climatology_cra.nc 读取
# 按月份选择气候态场
```
如果气候态数据不可用，ACC 暂时跳过或使用零场占位。

### 缺失数据处理
- 允许部分日期缺失（Catalog 返回 ambiguous=False 但 files 列表不完整）
- 允许部分 lead 缺失
- valid_mask 标记缺失点为 False

## 验证标准

### 成功标准
用我们的框架和原始参考代码分别评测同一数据（如 2026-08-19 起报，24h 时效），输出的 RMSE / Bias / ACC 数值一致（相对误差 < 0.1%）。

### 输出示例
```json
{
  "variable": "z500",
  "init_time": "2026-08-19T00:00:00",
  "lead_time": 24,
  "metrics": {
    "rmse": {
      "value": 42.3,
      "unit": "m",
      "n_valid": 1036800,
      "status": "SUCCESS"
    },
    "bias": {
      "value": -1.2,
      "unit": "m",
      "n_valid": 1036800,
      "status": "SUCCESS"
    },
    "acc": {
      "value": 0.987,
      "unit": "dimensionless",
      "n_valid": 1036800,
      "status": "SUCCESS"
    }
  }
}
```

## 风险和缓解

### 风险1：表面变量数据不可用
**缓解**：先完成 z500 闭环，为后续预留接口，不阻塞当前实现。

### 风险2：气候态数据缺失
**缓解**：ACC 暂时跳过或使用零场占位，文档说明需要外部气候态数据。

### 风险3：网格插值精度
**缓解**：使用 xarray 内置双线性插值，与原始脚本保持一致。

### 风险4：时间对齐复杂
**缓解**：Matcher 显式记录每个样本的 init_time / lead_time / valid_time，便于调试。

## 时间估算

- 第1步：2小时（Reader）
- 第2步：1小时（插值）
- 第3步：1.5小时（Matcher）
- 第4步：1.5小时（Metric）
- 第5步：1小时（主流程）
- 第6步：1小时（对比验证）

**总计**：约 8 小时

## 下一步行动

用户确认计划后，开始实现第1步：FengqingReader 和 CRAReader。
