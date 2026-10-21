# -*- coding: utf-8 -*-
"""确定性降水分类检验（AIFS 模型 × Diamond 站点）。

流程：``weather_ts_det``（站点有效时刻配对 + 24h 累积 + 网格到站点插值 + TS 系列）。
本文件只声明"数据从哪来、评哪段时间、写到哪"，算法口径全在流程模板里
（``pipeline/pipelines.py`` 的 ``weather_ts_det``）。

数据：
    预报  {root}/YYYYMMDD/001.nc…（tp，**落盘单位 m**；布局见 ``io/layouts.py``）
    观测  Diamond 站点降水（北京时文本，逐小时；协议按 +8 对齐）
    参考  无（TS 系列不吃气候态）

单位：AIFS 的 tp 落盘是 **米**，而 ``FUXI_LAYOUT`` 声明的是 mm（FGVP/FuXi 的落盘
就是 mm）。所以必须写 ``"scales": {"tp": 1000.0}``——不乘的话 ≥0.1mm 这一档几乎
全是漏报、TS 掉到 1e-3 量级（实测 24h ≥0.1 的 TS 0.0007，乘后 0.52，与参考评测
``ts_multi_aifs`` 一致）。scales 在布局换算之后再乘一次，见
``io/gridded.py`` 的 ``GriddedReader``。

规模：lead 到 360h、24h 累积窗 → 采样点是 24 的倍数（15 个）；
``chunk_days=1`` × ``lead_chunk_days=1`` 切出 15 个时效窗。

先冒烟再正式：``limit`` 改成 2 之类跑一遍，再改回 ``None``。``chunk_id``
不含 ``limit``，``resume`` 打开时冒烟算过的块会被原样复用，不会白算。
"""
from xmetai_evaluation.configs.base import EvalConfig

#: TS 阈值。**这是对模板的真覆盖**：``pipelines.TS_THRESHOLDS`` 只有 5 档
#: （0.1 / 10 / 25 / 50 / 100），这里补了 ≥250mm 的极端量级，与参考评测一致。
TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]

# 路径一律字面量，换机器/换目录直接改这里。
_AIFS_ROOT = "/workspace/data/fanpy/aifs-single-v10-2025-output"
_STATION_ROOT = "/workspace/data/worm/r0/2025"
_STATION_LIST = "/workspace/data/worm/r0/zd_sta_10285.dat"
_OUTPUT_DIR = "/workspace/data/shenzw/evaluation_results/weather_ts_single_aifs"

cfg = EvalConfig(
    name="weather_ts_single_aifs",
    description="AIFS 确定性降水 TS 评估（与站点观测对比）",
    pipeline="weather_ts_det",

    # ── 数据源 ──────────────────────────────────────────────────────
    forecast_reader={
        "type": "fuxi",                 # AIFS 落盘用的是 fuxi 的目录布局
        "root_dir": _AIFS_ROOT,
        "variable": "tp",
        "step_hours": 6.0,              # 布局步长：lead_from="index"，001.nc → +6h
        # tp 落盘 m、布局声明 mm：×1000 换成毫米。其余模型（FGVP/FuXi）是 mm，别照抄。
        "scales": {"tp": 1000.0},
        # 时效列表（小时）。**不写这一行 = 从目录索引推**：取目录里的最大时效
        # lead_max，按 step_hours 从 0 铺起（0, 6, 12, …, lead_max）。本流程只有
        # 24 的倍数那 15 个是采样点，多推出来的时刻读进来没人用，不影响结果。
        #
        # ⚠ 要写就必须**按 6h 密排**。24h 累积窗是 4 个连续 6h 步加出来的
        # （见 transforms/temporal.py），只写采样点会让每个窗凑不齐步数，
        # **一个样本都出不来**：
        #     ✗ [24, 48, 72, …, 360]         窗里只有 1 步，累积窗永远不完整
        #     ✓ [6, 12, 18, 24, 30, …, 360]  与推导结果同集合
        "lead_times": None,
    },
    observation_reader={
        "type": "station",              # 别名 diamond_station
        "root_dir": _STATION_ROOT,
        "variable": "precipitation",    # 标准名 tp；这个源只出降水
        # 与参考评测一致：仅用 zd_sta_10285.dat 的站点；改成 None 则用全部站点
        "station_list": _STATION_LIST,
    },

    # ── 口径覆盖 ────────────────────────────────────────────────────
    # 这三项是**替换**模板自带的那一份、不是追加：漏写就会丢掉模板的对应输出。
    transform_options={"time_window_accumulator": {"window_hours": 24}},
    metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
    writers=["csv_long", "categorical_wide"],   # 漏掉 categorical_wide，报告就没主表

    # 协议口径（station_valid_time）：8 = 北京时，Diamond 观测自带的口径，也是缺省。
    options={
        "local_utc_offset_hours": 8,
        # 分纬度带评估（闭区间）：{带名: {"lat_min": …, "lat_max": …}}，
        # 不加这个键就只有全球行。
        # "regions": {
        #     "南方": {"lat_min": 0, "lat_max": 25},
        #     "长江中下游": {"lat_min": 25, "lat_max": 32},
        #     "华北": {"lat_min": 32, "lat_max": 40},
        #     "东北": {"lat_min": 40, "lat_max": 90},
        # },
    },

    # ── 评测时段（periods 与 start_date/end_date **二选一**，同时给会报配置错）──
    # 单段用 periods 包一层即可；多段写多个元组，结果汇成一套统计量、空档不切块。
    # ⚠ 观测跨度取「最早起报 → 最晚起报」（protocols.py 的 station_observation_span），
    #   空档期的观测会被照读一遍（本配置 resident，拿 IO 换省事）。
    periods=[("20250102", "20251215")],
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=_OUTPUT_DIR,
    log_level="INFO",

    # ── 并发与数据加载：execution ────────────────────────────────────
    # 可用键只有 mode / n_workers / chunk_days / lead_chunk_days / loads / resume，
    # 写错键名会直接报配置错。
    #
    # 本流程是**轻指标**（逐站列联表，无 FFT、无成员维、无邻域），推导缺省就是
    # threads。**不要照抄 weather_rmse_single_aifs 的 processes + 18 进程**
    # ——那份是谱/活跃度的重指标，靠多进程绕 GIL；这里换多进程只会多出"每个子
    # 进程一份站点观测"的开销，而 threads 下 resident 全局只有 1 份（loader 有锁）。
    execution={
        "mode": "threads",      # 轻指标 → threads（auto 也会推导成它，这里写死）
        "n_workers": 24,        # 缺省 4；numpy/读盘释放 GIL，典型加速 2~3×，再加收益递减
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效），正好框住 1 个 24h 采样点
        # 站点观测的推导缺省就是 resident（整个 run 读一次）。本配置没有参考源，
        # 故不写 reference。
        "loads": {"observation": "resident"},
        "resume": True,         # 全年段建议开：中断后接着算
    },
)
