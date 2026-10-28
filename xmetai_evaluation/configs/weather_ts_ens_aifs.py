# -*- coding: utf-8 -*-
"""集合降水分类检验（AIFS 集合 × Diamond 站点）：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS。

与 ``weather_ts_ens_fgvp`` 只差一处实质配置：AIFS 落盘的 ``tp`` 是**自起报累计**，
所以 ``forecast_reader`` 里必须写 ``"since_init_variables": ["tp"]``——Reader 读入后
沿 ``lead_time`` 差分回逐 6h 间隔，累积窗才能用。不写这一项时 24h 窗会把
1..k 步再加一遍：≥10mm 的 BIAS 从 +24h 2.63 一路涨到 +360h 10.98（实测旧归档）。

单位：AIFS 集合归档的 tp 是 **mm**，不用写 ``scales``。别照抄
``weather_ts_single_aifs.py``——那份确定性归档落盘是米，要乘 1000。

时段：2025-07-02 ~ 2025-12-15 共 167 个 00Z 起报，与 ``降水/weather_ts_ens_aifs``
旧归档、FuXi / Tongyun 同期子集完全同一批日期，可直接配对比较。

输出目录特意用了新名字：旧目录的 ``.states/`` 是错误口径算出来的，而块名不含
口径，复用会让旧块静默生效。要覆盖旧目录，必须先删掉它的 ``.states/``。
"""

from xmetai_evaluation.configs.base import EvalConfig

# 路径一律字面量，换机器/换目录直接改这里。
_AIFS_ENS_ROOT = "/workspace/data/fanpy/aifs_ens_output"
_STATION_ROOT = "/workspace/data/worm/r0/2025"
_STATION_LIST = "/workspace/data/worm/r0/zd_sta_10285.dat"
_BSS_REF = "/workspace/data/worm/r0/ref"
_OUTPUT_DIR = "/workspace/data/shenzw/evaluation_results/weather_ts_ens_aifs_since_init"

cfg = EvalConfig(
    name="weather_ts_ens_aifs",
    description="AIFS 集合降水评估：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS",
    # 两段按顺序执行，结果合并落盘。
    pipeline=["weather_ts_ens", "weather_ts_ens_prob"],

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": _AIFS_ENS_ROOT,
        "variable": "tp",
        "step_hours": 6.0,      # 布局步长：lead_from="index"，001.nc → +6h
        # 目录里是 001.nc ~ 060.nc；文件变量名大写（TP）也无妨，布局大小写不敏感。
        "lead_times": None,
        # AIFS 的 tp 是 C(k)=Σ_{j≤k} I(j)（自起报累计），读入后沿 lead 差分。
        # 缺了这一项，24h 窗和 6h 概率检验都会把累计量当间隔量，偏差随时效线性增长。
        "since_init_variables": ["tp"],
    },
    observation_reader={
        "type": "station",
        "root_dir": _STATION_ROOT,
        "variable": "precipitation",
        "station_list": _STATION_LIST,
    },
    # BSS 气候概率参考（必填）。
    reference_reader={"type": "ref_probability", "root_dir": _BSS_REF},

    # ── 时段 ──
    # 与旧 AIFS 归档、FuXi/Tongyun 同期子集同一批 167 个 00Z 起报。
    periods=[("20250702", "20251215")],
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=_OUTPUT_DIR,
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
