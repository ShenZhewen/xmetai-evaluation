# -*- coding: utf-8 -*-
"""台风路径与强度检验（FGVP 集合 × BABJ 报文）：逐成员链式诊断中心，两种集合口径输出误差曲线。"""
from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="weather_typhoon_ens_fgvp",
    description="台风路径与强度检验（FGVP 集合 × BABJ 报文）",
    pipeline="weather_typhoon_ens",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": "/workspace/data/shenzw/fgvp_ens_merged",
        # 定中心要 msl，定强度要 u10m/v10m；缺风速只留空强度项。
        "variables": ["msl", "u10m", "v10m"],
        "step_hours": 6.0,              # 布局步长：lead_from="index"，001.nc → +6h
        # None = 目录索引推导；链式诊断不能写稀疏集。
        "lead_times": None,
    },
    observation_reader={
        "type": "babj",
        "root_dir": "/workspace/data/LiuJs/babj",
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
        # 种子：起报后第一条不早于 init + 6h 的实况，全体成员共用。
        "seed_min_offset_hours": 6.0,
    },

    # ── 时段 ──
    # 三段起报日共 113 个；段间空档不切块、不出样本。
    periods=[
        ("20250702", "20250721"),   # 20 天
        ("20250723", "20251007"),   # 77 天
        ("20251104", "20251119"),   # 16 天
    ],

    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir="/workspace/data/shenzw/evaluation_results/weather_typhoon_ens_fgvp",
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
