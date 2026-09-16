# eval_pro: 气象模型离线评测框架

气象模型**离线评测**：输入已经推理完成的预报产品 + 观测/再分析/气候态，输出结构化指标 CSV。

**只做评测，不做推理**：不加载模型权重、不训练、不调度推理。

```
xmetai-inference（推理框架）→ 预报产品目录 → eval_pro → outputs/results/ 下的长表 CSV
```

## 目录结构

```text
eval_pro/
├── runner.py            # 唯一入口：加载 config → capability 或 batch
├── core/                # 核心实现（runner 之外的所有 py 都在这里）
│   ├── api.py               # capability 注册与参数适配
│   ├── batch_adapter.py     # batch config → 参考实现 CLI 参数
│   ├── run_batch_rmse.py     # 参考实现入口（RMSE/ACC/FA/谱 批量评测）
│   ├── runlog.py             # fd 级运行日志（控制台 + logs/ 双写）
│   ├── categorical_ref.py    # 降水分类检验参考实现
│   └── tc_ref.py             # 台风路径/强度检验参考实现
├── vfc/                 # 验证核心模块（拷自 xmetai_model_verification_xu）
├── configs/            # 评测配置（Python dict，字面量默认值）
├── outputs/results/    # 评测产物
├── outputs/.temp/      # 断点续跑缓存（按 config 指纹分目录）
├── ref_result/         # 参考结果（只读对照，不参与运行）
├── skills/             # 报告生成技能（见 skills/xmetai-evaluation/）
└── visualization/      # 报告图件
```

配置文件按 `weather_<流程>_<single/ens>_<模型>` 命名，输出目录与 config 名一致。

## 快速开始

```bash
# capability 类配置（降水 TS / 台风）
python runner.py -config configs/weather_ts_single_fgvp.py

# batch 类配置（RMSE/ACC/FA/谱，自 scripts/*.sh 迁移）
python runner.py --config configs/weather_rmse_single_fengqing.py
```

## 评测能力清单

| 能力 | 类型 | 输入 | 输出 | 对应 config |
|---|---|---|---|---|
| **确定性降水分类检验** `weather_ts_det` | capability | 格点降水预报 + Diamond 站点观测 | `ts_<name>.csv`：TS/POD/FAR/漏报率/BIAS × 阈值(0.1/10/25/50/100/250mm) × 24h 时效 | `weather_ts_single_fgvp` |
| **集合降水分类检验** `weather_ts_ens` | capability | 集合平均降水场 + 站点观测 | 同上（集合平均场口径） | `weather_ts_ens_fuxi` |
| **集合降水概率评分** `weather_ts_ens_prob` | capability | 集合降水 + 站点观测 + 气候概率 | AROC/BS/BSS × 6h | **[BLOCKED]**（缺 `vfc.reader_categorical` 等模块，`weather_ts_prob_ens_fuxi` 仅供参考） |
| **确定性连续量检验**（batch） | batch | 预报根目录 + era5 zarr + 气候态 | RMSE/ACC/活跃度/谱长表 | `weather_rmse_single_{fuxi,fgvp,fengqing}` |
| **集合连续量检验**（batch） | batch | 同上（集合） | 同上 + CRPS/离散度 | `weather_rmse_ens_fuxi` |
| **台风路径/强度检验** `typhoon` | capability | 预报场 + babj 实况 | `tc<编号>_<起报>.csv`：路径/强度误差 | `weather_typhoon_single_fuxi` |

## 评估指标说明

分类检验（降水阈值，超越式口径 `x ≥ 阈值`）：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `ts` | TS（Threat Score，即 CSI）：命中占「命中+漏报+空报」的比例 | `hits / (hits + misses + false_alarms)` | 0~1，越大越好 |
| `pod` | 命中率：实况发生的事件里被预报到的比例 | `hits / (hits + misses)` | 越大越好；空报多也能拿高分，需与 `far` 同看 |
| `far` | 空报率：预报的事件里实况没发生的比例 | `false_alarms / (hits + false_alarms)` | 越小越好 |
| `miss_rate` | 漏报率 | `1 − POD` | 越小越好 |
| `frequency_bias` | 频率偏差：预报事件数 / 实况事件数 | `(hits + false_alarms) / (hits + misses)` | 1 为无偏，>1 空报偏多、<1 漏报偏多 |

连续量 / 集合评分（batch 类）：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `rmse` | 均方根误差 | 逐起报算 RMSE，再跨起报平均（**不是**把 MSE 平均后开根） | 越小越好 |
| `acc` | 距平相关系数 | 距平 = 场 − 气候态，域加权 | −1~1，越大越好；**缺气候态则无意义** |
| `fa` | 活跃度比：预报距平变化幅度相对实况 | `std(预报距平) / std(实况距平)`，面积加权 | <1 偏平滑（系统性偏弱），>1 偏噪 |
| `spectrum` | 功率谱 / 纬向谱 | 预报与实况逐波数功率对比；总功率比 = 能量是否偏 | 1 表示总能量不偏；单看总量会掩盖分布失真，需与谱曲线同看 |
| `crps` | 连续排序概率评分（集合） | 闭式解，逐点只用有限成员 | 越小越好 |
| `spread` | 集合离散度（集合） | 成员标准差 | 与 `rmse` 同量级才有意义 |
| `spread_error_ratio` | 离散度-误差比（集合） | `SPREAD / RMSE(集合平均场)` | ≈1 标定良好，<1 过度自信，>1 欠自信 |

读表要点：

- **跨起报平均在输出层做，不在指标层**：对逐起报 RMSE 再平均 ≠ 把 MSE 平均后开根，参考实现也是前者。
- **ts CSV 里的 `grade` 列写 `≥0.1`/`≥250`**（超越式），阈值档位 0.1/10/25/50/100/250mm 对齐参考结果 `ts_fgvp_2025.csv`。
- **列联表计数（hits/misses/false_alarms/n_pairs）直接在 ts CSV 里**，不单独透视。

### 单位换算要点（踩过的坑）

| 模型 | 预报 q 单位 | config 写法 |
|---|---|---|
| FuXi / FGVP | g/kg | `"pred_q_scale": 0.001`（→ kg/kg 对齐 era5 目标） |
| **FengQing** | **kg/kg** | **不写 `pred_q_scale`**（默认 1.0；官方 upper_mean.npy q 块 mean≈0.0018 实锤）。抄 fgvp 的 0.001 会把 q 缩小 1000 倍，q 的 RMSE 大到离谱且只有 q 异常 |

## 配置

两类配置都是 Python dict、字面量默认值（不用环境变量）。

**capability 类**（降水 TS / 台风）：

```python
# configs/weather_ts_single_fgvp.py
CONFIG = {
    "capability": "weather_ts_det",
    "pred": "/workspace/data/shenzw/fgvp_output",
    "station_dir": "/workspace/data/worm/r0/2025",
    "start_date": "20250101", "end_date": "20251216",  # 20251217 起预报数据缺失，不评
    "windows": [24.0], "lead_step": 6.0, "tz_shift": 8.0,
    "interp": "bilinear", "tp_scale": 1.0, "workers": 8,
    # name 决定落盘文件名：ts_single_fgvp.csv + single_fgvp_meta.json
    "name": "single_fgvp",
    "output_dir": ".../outputs/results/weather_ts_single_fgvp",
}
```

**batch 类**（RMSE/ACC/FA/谱，自 scripts/*.sh 迁移，语义与原脚本一致）：

```python
# configs/weather_rmse_single_fengqing.py（节选）
CONFIG = {
    "type": "batch",
    "label": "fengqing",
    "output_name": "weather_rmse_single_fengqing",
    "pred_root": "/workspace/data/shenzw/fengqing_output",
    "target_zarr": ["…sfc….zarr", "…pl….zarr"],
    "outdir_root": "/workspace/_XMETAI_test_results/single_fengqing",
    # 20251217 起预报数据缺失/无效（q700 全 NaN、单日内存暴涨），所有 config 统一止于 20251216
    "periods": [("20250101", "20250630"), ("20250701", "20251216")],
    "metrics": ["rmse", "spectrum", "acc", "fa"],
    "variables": ["z500", "q700", ...],
    "var_metrics": {"z500": ["rmse", "spectrum", "acc", "fa"], ...},  # 逐变量过滤
    "climo": "/workspace/data/worm/era5_clim_phys_14.nc",
    "n_workers": 48,
    "worker_fallback": [48, 4, 2],
    "resume": True,
    "resume_cache": True,
    "summarize_mode": "--summarize-det",
    "env_overrides": {"VFC_DATES_PER_CHILD": "1", "OMP_NUM_THREADS": "1", ...},
}
```

batch 配置语义：

| 字段 | 含义 |
|---|---|
| `periods` | 评测时段（过滤到 pred_root 里实际存在的 YYYYMMDD 目录） |
| `var_metrics` | 逐变量 × 指标过滤（`metrics` 是总集） |
| `pred_q_scale` | 预报 q 缩放（见上表；FuXi/FGVP=0.001，FengQing 不写） |
| `n_workers` / `worker_fallback` | 并行进程数 / 失败降档重试序列 |
| `resume` | 跳过已有完整输出的日期 |
| `summarize_mode` | 汇总口径：`--summarize-det` / `--summarize-ens` |
| `env_overrides` | 子进程启动前注入的环境变量（BLAS 线程控制等） |

## 批处理输出与断点续跑

- 逐日期结果写到 `outputs/.temp/<output_name>/<fingerprint>/<YYYYMMDD>/`（rmse/acc 等逐日 CSV + `<date>_meta.json`），**持续保留**，`resume: True` 断点续跑就靠它。
- 改配置（日期/指标/变量）会产生新 fingerprint 目录，旧缓存不自动清理。想让新 periods 复用旧缓存：跑一次拿到新 fingerprint 路径（日志 `Using persistent cache:` 行），把旧缓存目录改名成新路径即可——但**必须删掉不再评测的日期目录**（如 20251217-20251228），因为 `--summarize-det` 扫的是缓存根下所有 YYYYMMDD 目录、不看 periods，留着会把坏数据混进 mean 文件。
- 汇总文件在缓存根：`summary.csv`、`mean_*.csv`、`det_summary_overall.csv`（或 ens）。
- 最终长表发布到 `outputs/results/<output_name>/<output_name>.csv`。

并行策略：起报日期按 `VFC_DATES_PER_CHILD`（=1）切块，短命 worker 每进程只跑一块，内存不跨日期累积。首次启动有 20 分钟左右的冷启动静默（48 个 worker 同时加载库/气候态），期间日志无输出是**正常现象**，等第一波 chunk 完成后进度会突然密集出现。

**worker 数怎么定**（2026-09-15 480G/48核 全量日志实锤）：普通日期单 worker 常驻 <10G，48 并发吞吐最高（~4 日期/分钟）；OOM 只发生在坏数据日期上（单日 >40G，12 并发即崩）。所以 `n_workers=48` 起跑、fallback `[48, 4, 2]`——中间档（如 36/24/12）撞上坏日期一样崩，每档白死 ~10 分钟，不值得放阶梯里。换内存规格的机器按 `可用内存 × 0.7 ÷ 单 worker 峰值` 估算并发数。

## 运行日志

- `runner.py` 把全部输出（含并行 worker 和 `run_batch_rmse.py` 子进程）同时写到控制台和 `logs/runner_<config名>_<北京时间>.log`。
- 接管的是文件描述符而不是 `sys.stdout`，所以 `logging` / 子进程 / 未捕获的 traceback 都会进日志；每个 chunk 都是不带缓冲的 `os.write`，进程被 `kill -9` 掉日志也是完整的。
- 同一秒内重跑同一个 config 会写成 `runner_<config名>_<时间>_2.log`，不会混进同一个文件。

## 如何扩展

- **新增模型评一批 RMSE/TS**：抄最接近的 config 改 `pred_root` 和单位（注意 q 换算表），代码不动。
- **新增指标**：在 `vfc/metrics/` 实现，参考实现在 `core/run_batch_rmse.py` / `vfc/regr_ens.py` 的指标分发处接入。
- **新增评测能力**：`core/api.py` 加适配函数，`runner.py` 的 `CAPABILITIES` 注册。

## 已知限制

1. **无本地执行**：数据在远程服务器，本仓库只维护代码与配置。
2. **概率评分 BLOCKED**：`weather_ts_ens_prob` 依赖的 `vfc.reader_categorical` / `station_categorical` / `metric_categorical` 在参考源里缺失，没有源码不造实现。
3. **无输出清理**：逐日结果永久保留（resume 需要）。
4. **batch 迁移不全**：原 run_all.sh 的 6 个模型只迁了 4 个（single_fuxi / ens_fuxi / single_fgvp / single_fengqing；single_aifs、ensemble_aifs 未迁）。
5. **评测时段止于 20251216**：20251217 起预报数据缺失/无效（q700 全 NaN、评测时单日内存暴涨 10 倍），所有 RMSE/TS config 统一排除该窗口。跨模型对比时 H2 口径为 `20250701-20251216`。
