# -*- coding: utf-8 -*-
"""确定性降水分类检验（FGVP 模型 × Diamond 站点）。

流程：``weather_ts_det``（站点有效时刻配对 + 24h 累积 + 网格到站点插值 + TS 系列）。
本文件只声明"数据从哪来、评哪段时间、写到哪"，算法口径全在流程模板里
（``pipeline/pipelines.py`` 的 ``weather_ts_det``）。

数据：
    预报  {root}/YYYYMMDD/001.nc…（tp，mm，逐 6h；布局见 ``io/layouts.py``）
    观测  Diamond 站点降水（北京时文本，逐小时；协议按 +8 对齐）
    参考  无（TS 系列不吃气候态）

规模：lead 到 360h、24h 累积窗 → 采样点是 24 的倍数（15 个）；
``chunk_days=1`` × ``lead_chunk_days=1`` 切出 15 个时效窗，348 个起报
→ 348 × 15 = 5220 块。

先冒烟再正式：``limit`` 改成 2 之类跑一遍，再改回 ``None``。``chunk_id``
不含 ``limit``，``resume`` 打开时冒烟算过的块会被原样复用，不会白算。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

#: TS 阈值。**这是对模板的真覆盖**：``pipelines.TS_THRESHOLDS`` 只有 5 档
#: （0.1 / 10 / 25 / 50 / 100），这里补了 ≥250mm 的极端量级，与参考评测一致。
TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]

cfg = EvalConfig(
    name="weather_ts_det_fgvp",
    description="FGVP 确定性降水 TS 评估（与站点观测对比）",
    pipeline="weather_ts_det",

    # ── 数据源 ──────────────────────────────────────────────────────
    # 路径是字面量，换机器/换目录直接改这里。
    forecast_reader={
        "type": "fuxi",                 # FGVP 落盘用的是 fuxi 的目录布局
        "root_dir": os.environ.get("FGVP_OUTPUT", "/workspace/data/shenzw/fgvp_output"),
        "variable": "tp",
        "step_hours": 6.0,              # 布局步长：lead_from="index"，001.nc → +6h
        # 时效列表（小时）。**不写这一行 = 从目录索引推**：取目录里的最大时效
        # lead_max，按 step_hours 从 0 铺起（0, 6, 12, …, lead_max）。本流程只有
        # 24 的倍数那 15 个是采样点，多推出来的时刻读进来没人用，不影响结果，
        # 所以缺省就是对的。
        #
        # ⚠ 要写就必须**按 6h 密排**。24h 累积窗是 4 个连续 6h 步加出来的
        # （见 transforms/temporal.py），预热时效只从这张列表里取（plan.py 的
        # _chunk_work）。只写采样点会让每个窗凑不齐步数，**一个样本都出不来**：
        #     ✗ [24, 48, 72, …, 360]         窗里只有 1 步，累积窗永远不完整
        #     ✓ [6, 12, 18, 24, 30, …, 360]  与推导结果同集合
        # 只想缩短范围就写稠密前缀，省读盘也省内存：
        #     "lead_times": [6, 12, 18, 24, 30, 36, …, 240],   # 只评到 240h
        #
        # 换逐日数据：只把 step_hours 改成 24.0 就够了（列表不写，推导自动对：
        # 15 个日文件 × 24h → 0, 24, …, 360）。反过来，**只改列表不改
        # step_hours 是静默错**——框架仍按 6h 去切日文件，采样点从 15 个掉到
        # 3 个，24h 累积量变成 4 个日文件相加（实际 96h），全程不报一个错。
        "lead_times": None,
    },
    observation_reader={
        "type": "station",              # 别名 diamond_station
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",    # 标准名 tp；这个源只出降水
        # 与参考评测一致：仅用 zd_sta_10285.dat 的站点；改成 None 则用全部站点
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ),
    },

    # ── 口径覆盖 ────────────────────────────────────────────────────
    # 这三项是**替换**模板自带的那一份、不是追加：漏写就会丢掉模板的对应输出。
    # 下面窗口与 writers 两行目前与模板值相同（24h / ["csv_long",
    # "categorical_wide"]），写出来是为了好改——改了它就和模板口径不同了。
    # 窗口的正本在模板里（"怎么算"归模板，"算什么"归配置）。
    transform_options={"time_window_accumulator": {"window_hours": 24}},
    metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
    writers=["csv_long", "categorical_wide"],   # 漏掉 categorical_wide，报告就没主表

    # 协议口径（station_valid_time）：8 = 北京时，Diamond 观测自带的口径，
    # 也是协议缺省。另可配 region（区域筛选）、weights（"cos_lat"）。
    options={"local_utc_offset_hours": 8},

    start_date=os.environ.get("START_DATE", "20250101"),
    # 止于 15 号、不是 31 号：**站点观测档案到 2025-12-31 23:00 就没了**，预报比它长。
    # 起报越靠后，长时效的验证时刻越容易越过这条线，那种块整块评不出样本（日志里是
    # 「观测窗口不完整」+「没有任何样本可评」），只会白占一个失败块。15 号的最长时效
    # 落在 12-30，留一天余量。观测补齐后可改回来，`.states/` 里的旧块会原样复用。
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_ts_det_fgvp",
    ),
    log_level="INFO",

    # ── 并发与数据加载：execution ────────────────────────────────────
    # 可用键只有 mode / n_workers / chunk_days / lead_chunk_days / loads / resume，
    # 写错键名会直接报配置错。全部字面量，不写的键用推导值。
    #
    # 本流程是**轻指标**（逐站列联表，无 FFT、无成员维、无邻域），推导缺省就是
    # threads。**不要照抄 weather_field_scores_era5_fuxi 的 processes + 24 进程**
    # ——那份是谱/活跃度的重指标，靠多进程绕 GIL；这里换多进程只会多出"每个子
    # 进程一份站点观测"的开销，而 threads 下 resident 全局只有 1 份（loader 有锁）。
    execution={
        "mode": "threads",      # 轻指标 → threads（auto 也会推导成它，这里写死）
        "n_workers": 4,         # 缺省 4；numpy/读盘释放 GIL，典型加速 2~3×，再加收益递减
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效），正好框住 1 个 24h 采样点
        # 站点观测的推导缺省就是 resident（整个 run 读一次：Diamond 是逐时文本，
        # 按块现读等于把全年观测读 5220 遍）。本配置没有参考源，故不写 reference。
        "loads": {"observation": "resident"},
        "resume": True,         # 全年段建议开：中断后接着算
    },
)
