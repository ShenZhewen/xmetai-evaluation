# -*- coding: utf-8 -*-
"""台风路径与强度检验（FuXi 确定性 × BABJ 报文）：链式诊断预报中心，与 BABJ 分析场配对。"""
import os

from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="weather_typhoon_single_fuxi",
    description="台风路径与强度检验（FuXi 确定性 × BABJ 报文）",
    pipeline="weather_typhoon_det",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_phys",            # 带 _phys = xu 复刻口径，要素表全、msl 报 Pa
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
        ),
        # 定中心要 msl，定强度要 u10m/v10m；缺风速只留空强度项。
        "variables": ["msl", "u10m", "v10m"],
        "step_hours": 6.0,              # 布局步长：lead_from="index"，001.nc → +6h
        # None = 目录索引推导；链式诊断不能写稀疏集。
        "lead_times": None,
    },
    observation_reader={
        "type": "babj",
        "root_dir": os.environ.get("BABJ_DIR", "/workspace/data/LiuJs/babj"),
    },

    # ── 协议 ──
    # 搜索框单位经纬度；early_hours 内用大框。
    options={
        "center_half_deg": 3.0,
        "early_half_deg": 4.0,
        "early_hours": 12.0,
        # 强度框取当前诊断中心附近最大 10m 风速。
        "intensity_half_deg": 5.0,
        "wind_variables": ["u10m", "v10m"],
        # 种子：起报后第一条不早于 init + 6h 的实况。
        "seed_min_offset_hours": 6.0,
    },

    # ── 时段 ──
    # 2025 全年起报日；场次还要经 BABJ 报文筛选。
    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251231"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/data/shenzw/evaluation_results/weather_typhoon_single_fuxi",
    ),
    log_level="INFO",

    writers=["csv_long", "typhoon_cases"],

    # ── 执行 ──
    # n_workers 按内存调整，单块约 0.75 GB。
    execution={
        "mode": "processes",
        "n_workers": 8,         # 8 × 0.75G ≈ 6G 工作集；内存够可以往上加
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 0,   # 不切时效（链式诊断的硬要求，改成非 0 会报错）
        "loads": {"observation": "slice"},
        "resume": True,         # 中断后接着算
    },
)
