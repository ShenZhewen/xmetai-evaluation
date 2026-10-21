# -*- coding: utf-8 -*-
"""确定性降水分类检验（FGVP 模型 × Diamond 站点），流程 weather_ts_det。"""

import os

from xmetai_evaluation.configs.base import EvalConfig

# TS 阈值；比模板多 ≥250mm 一档。
TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]

cfg = EvalConfig(
    name="weather_ts_single_fgvp",
    description="FGVP 确定性降水 TS 评估（与站点观测对比）",
    pipeline="weather_ts_det",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi",                 # FGVP 落盘用的是 fuxi 的目录布局
        "root_dir": os.environ.get("FGVP_OUTPUT", "/workspace/data/shenzw/fgvp_output"),
        "variable": "tp",
        "step_hours": 6.0,              # 布局步长：lead_from="index"，001.nc → +6h
        # None = 从目录索引推导；显式写须按 6h 密排，否则 24h 累积窗凑不齐。
        "lead_times": None,
    },
    observation_reader={
        "type": "station",              # 别名 diamond_station
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",    # 标准名 tp；这个源只出降水
        # 与参考评测一致：仅用 zd_sta_10285.dat 的站点；None = 全部站点。
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ),
    },

    # ── 口径覆盖 ──
    # 这三项替换模板自带值，不是追加。
    transform_options={"time_window_accumulator": {"window_hours": 24}},
    metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
    writers=["csv_long", "categorical_wide"],   # 漏掉 categorical_wide，报告就没主表

    # ── 协议 ──
    # 8 = 北京时（Diamond 观测口径）。
    options={
        "local_utc_offset_hours": 8,

        # 纬度带（闭区间，单位度；站点按站点纬度整站归带）。
        "regions": {
            "南方": {"lat_min": 0, "lat_max": 25},
            "长江中下游": {"lat_min": 25, "lat_max": 32},
            "华北": {"lat_min": 32, "lat_max": 40},
            "东北": {"lat_min": 40, "lat_max": 90},
        },
    },

    # ── 时段 ──
    # 两段共 8 个起报日；段间空档不切块、不出样本。
    periods=[("20250102", "20250105"), ("20250702", "20250705")],
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/evaluation_results/weather_ts_single_fgvp",
    ),
    log_level="INFO",

    # ── 执行 ──
    execution={
        "mode": "threads",      # 轻指标 → threads（auto 也会推导成它，这里写死）
        "n_workers": 24,         # 缺省 4；numpy/读盘释放 GIL，典型加速 2~3×，再加收益递减
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效），正好框住 1 个 24h 采样点
        "loads": {"observation": "resident"},
        "resume": True,         # 全年段建议开：中断后接着算
    },
)
