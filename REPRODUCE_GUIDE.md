# 复现验证指南

## 目标

用新框架复现原代码 `run_categorical.py` 的结果，验证框架正确性。

---

## 步骤 1：准备环境（服务器上）

```bash
# 1. 进入新框架目录
cd /workspace/szwCode/xmetai-evalation

# 2. 安装框架
pip install -e .

# 3. 安装额外依赖
pip install scipy pandas

# 4. 验证安装
python -c "from xmetai_evaluation.io.fuxi_reader import FuXiReader; print('✓ 安装成功')"
```

---

## 步骤 2：小样本测试（快速验证）

**目的：** 1个起报，24h时效，验证数值计算正确

```bash
# 运行小样本测试
python reproduce_ts_eval.py --mode small

# 预计耗时：1-2 分钟
```

**预期输出：**
```
============================================================
FuXi 降水 TS 评测复现脚本
============================================================
模式: small
...
【小样本模式】：1个起报，24h时效

步骤 1：初始化数据读取器...
  ✓ Reader 初始化完成

步骤 2：读取预报数据...
  ✓ 发现 1 个起报时间的数据
  ✓ 预报数据形状: (1, 10, 721, 1440)

步骤 3：读取站点观测...
  ✓ 发现 XXX 个站点文件
  ✓ 观测数据形状: (XXX, YYY)

步骤 4：初始化转换器和指标...
  ✓ 转换器和指标初始化完成（6 个阈值）

步骤 5：开始计算...
============================================================
处理起报: 2025-01-02 00:00
  ✓ 24h累积完成
  处理 1 个时效: [24.0]
    时效 24.0h: ✓ 完成（XXX 个有效站点）

============================================================
步骤 6：合并所有状态...
  成功处理: 1 个 (起报×时效)
  ✓ 状态合并完成
  总有效样本数: XXX

步骤 7：输出结果...
============================================================

≥0.1:
  TS:   0.XXXXXX
  POD:  0.XXXXXX
  FAR:  0.XXXXXX
  BIAS: X.XXXXXX
  命中: XXX, 漏报: XXX, 空报: XXX

... (其他阈值)

✓ 结果已保存到: /workspace/szwCode/xmetai-evalation/results_reproduce.csv
```

**如果出错：**
- 检查路径是否正确
- 查看错误信息，告诉我具体问题

---

## 步骤 3：对比结果

```bash
# 查看新框架结果
cat /workspace/szwCode/xmetai-evalation/results_reproduce.csv

# 查看原代码结果（找对应的行）
# 原代码结果在：/workspace/szwCode/xmetai-evaluate/ref/tiqnqi/xmetai_model_verification/results2/en/ts_*.csv
head -20 /workspace/szwCode/xmetai-evaluate/ref/tiqnqi/xmetai_model_verification/results2/en/ts_*.csv | grep "24,24"

# 对比关键指标：
# - TS (Threat Score)
# - POD (Probability of Detection)
# - FAR (False Alarm Ratio)
# - BIAS

# 允许的误差：±0.01（由于插值和数值精度差异）
```

**预期：** 新框架和原代码的 TS/POD/FAR/BIAS 应该**非常接近**（±0.01 以内）

---

## 步骤 4：单起报完整测试（如果小样本通过）

```bash
# 测试一个完整起报的所有时效
python reproduce_ts_eval.py --mode single --init-date 20250102

# 预计耗时：5-10 分钟（取决于时效数量）
```

这会处理该起报的所有时效（24h, 48h, 72h, ...），输出综合结果。

---

## 步骤 5：全量测试（谨慎）

```bash
# 只有在前两步都通过后再运行
python reproduce_ts_eval.py --mode full

# 预计耗时：可能需要较长时间（取决于起报数量）
```

---

## 🔍 调试技巧

### 如果小样本测试失败

**1. 检查数据读取**
```python
# 在服务器上交互式测试
python
>>> from xmetai_evaluation.io.fuxi_reader import FuXiReader, FuXiCatalog
>>> from pathlib import Path
>>> from datetime import datetime
>>> from xmetai_evaluation.core.contracts import DataRequest

>>> reader, catalog = FuXiReader.with_catalog(Path("/workspace/data/shenzw/fuxi_single_output"))
>>> req = DataRequest(source_id="fuxi", variables=["tp"], init_times=[datetime(2025,1,2,0)])
>>> index = catalog.discover(req)
>>> print(f"发现文件: {len(index.available[0])}")
>>> bundle = reader.read(req, index)
>>> print(f"数据形状: {bundle.payload['tp'].shape}")
>>> print(f"数据范围: {bundle.payload['tp'].min().values} ~ {bundle.payload['tp'].max().values}")
```

**2. 检查站点观测**
```python
>>> from xmetai_evaluation.io.station_reader import DiamondStationReader, DiamondStationCatalog
>>> reader_s, catalog_s = DiamondStationReader.with_catalog(Path("/workspace/data/worm/r0/2025"))
>>> req_s = DataRequest(source_id="obs", variables=["precipitation"])
>>> index_s = catalog_s.discover(req_s)
>>> print(f"发现站点文件: {len(index_s.available)}")
>>> bundle_s = reader_s.read(req_s, index_s)
>>> print(f"站点数: {len(bundle_s.payload.coords['station'])}")
>>> print(f"时间范围: {bundle_s.payload.coords['time'].values[0]} ~ {bundle_s.payload.coords['time'].values[-1]}")
```

**3. 检查插值**
```python
>>> from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator
>>> import numpy as np

>>> interp = GridToStationInterpolator(method="bilinear")
>>> # 测试插值一个场
>>> test_field = bundle.payload['tp'].isel(init_time=0, lead_time=0)
>>> station_lats = bundle_s.payload.coords['lat'].values
>>> station_lons = bundle_s.payload.coords['lon'].values
>>> result = interp.transform(test_field, station_lats, station_lons)
>>> print(f"插值后形状: {result.shape}")
>>> print(f"有效站点数: {np.isfinite(result.values).sum()}")
```

---

## ⚠️ 已知的潜在问题

### 1. 时间对齐

**问题：** 观测文件是逐小时的，预报是累积24h的，需要确保时间匹配正确。

**当前脚本的简化假设：** 
- 假设观测文件已经是累积值（实际可能需要手动累积24小时）
- 用 `nearest` 方法匹配时间（可能需要更精确的逻辑）

**如果结果不一致，可能需要：**
```python
# 手动累积观测的24小时
obs_24h = obs_data.sel(time=slice(start_time, end_time)).sum(dim='time')
```

### 2. 站点白名单

**原代码：** 可能使用了站点白名单（`--station-list`）

**当前脚本：** 使用所有站点

**如果需要白名单：**
```python
# 修改脚本，添加
station_whitelist = [45004, 45005, ...]  # 从原代码的站点列表文件读取
station_reader, station_catalog = DiamondStationReader.with_catalog(
    station_dir=Path(args.station_dir),
    source_id="diamond_obs",
    station_whitelist=station_whitelist,  # 添加这个参数
)
```

### 3. 单位转换

**预报：** FuXi 输出的 TP 单位可能需要检查（mm 还是 m？）

**观测：** Diamond 文件的降水单位（通常是 mm）

**如果单位不一致：**
```python
# 在插值后转换
if forecast_unit == "m":
    forecast_interp = forecast_interp * 1000  # m -> mm
```

---

## 📊 成功标准

**小样本测试通过 = 新框架可用：**
- TS 误差 < 0.01
- POD 误差 < 0.01  
- FAR 误差 < 0.01
- BIAS 误差 < 0.05

**全量测试通过 = 完全复现：**
- 所有起报×时效的结果与原代码一致

---

## 🚀 现在开始

```bash
# 在服务器上运行
cd /workspace/szwCode/xmetai-evalation
python reproduce_ts_eval.py --mode small
```

**把运行结果（完整的终端输出）发给我，我帮你分析！**
