# -*- coding: utf-8 -*-
"""集合降水分类检验（FuXi 集合 × Diamond 站点）：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS。"""

import os

from xmetai_evaluation.configs.base import EvalConfig

#: BSS 外部气候概率目录（ref/MMDDHH.000）；负责人给定的固定输入，必填。
BSS_REF_DEFAULT = os.environ.get("BSS_REF", "/workspace/data/worm/r0/ref")

cfg = EvalConfig(
    name="weather_ts_ens_fgvp",
    description="FuXi 集合降水评估：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS",
    # 两段按顺序执行，结果合并落盘。
    pipeline=["weather_ts_ens", "weather_ts_ens_prob"],

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fgvp_ens_merged"
        ),
        "variable": "tp",
        "step_hours": 6.0,      # 布局步长：lead_from="index"，001.nc → +6h
        # None = 从目录索引推导；显式写须按 6h 密排，否则累积窗凑不齐。
        "lead_times": None,
    },
    observation_reader={
        "type": "station",
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ),
    },
    # BSS 气候概率参考（必填）。
    reference_reader={"type": "ref_probability", "root_dir": BSS_REF_DEFAULT},

    # ── 时段 ──
    # 五段起报日共 228 天；段间空档不切块、不出样本。
    periods=[
        ("20250102", "20250411"),   # 100 天
        ("20250504", "20250518"),   #  15 天
        ("20250702", "20250721"),   #  20 天（7/22 残缺，单独剔在外面）
        ("20250723", "20251007"),   #  77 天
        ("20251104", "20251119"),   #  16 天
    ],
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT", "/workspace/data/shenzw/evaluation_results/weather_ts_ens_fgvp"
    ),
    log_level="INFO",

    # ── 协议 ──
    options={
        "local_utc_offset_hours": 8,   # 北京时（Diamond 观测口径）
        # 分带：填 {"南方": {"lat_min": 0, "lat_max": 25}, ...} 即启用，闭区间。
        "regions": {},
    },

    # ── 执行 ──
    # n_workers 按 nproc 调整，单进程约 2 GB。
    execution={
        "mode": "processes",    # 两段都走进程：threads 会把读盘全排在 hdf5_guard 的锁上
        "n_workers": 24,
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {"observation": "resident"},
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
