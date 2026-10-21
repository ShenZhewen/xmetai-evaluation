# -*- coding: utf-8 -*-
"""确定性连续量检验（FengQing × ERA5，复用 fuxi_phys 布局）：RMSE / 谱 / ACC / 活跃度。
z500 报 m²/s²（不除 g）；q 落盘 kg/kg，scales ×1000 后按 g/kg 报。
"""

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 要素表 ──
VARS = [
    "z500", "q700", "t700", "t2m", "t850", "msl",
    "u850", "v850", "u10m", "v10m", "ws10m", "ws850",
]

# ── 逐变量指标表 ──
VAR_METRICS = {
    "z500": ["rmse", "zonal_spectrum", "spherical_bands", "acc", "activity"],
    "q700": ["rmse"],
    "t700": ["rmse", "zonal_spectrum", "spherical_bands"],
    "t2m": ["rmse", "zonal_spectrum", "spherical_bands"],
    "t850": ["rmse"],
    "msl": ["rmse", "zonal_spectrum", "spherical_bands"],
    "u850": ["rmse", "activity", "zonal_spectrum", "spherical_bands"],
    "v850": ["rmse", "activity", "zonal_spectrum", "spherical_bands"],
    "u10m": ["rmse", "activity"],
    "v10m": ["rmse", "activity"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# centered=False（uncentered / FDP 口径，对齐 xu）
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# max_wavenumber=720（全球 0.25° Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}

_FENGQING_ROOT = "/workspace/data/shenzw/fengqing_output"
_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
_OUTPUT_DIR = "/workspace/data/shenzw/evaluation_results/weather_rmse_single_fengqing"

cfg = EvalConfig(
    name="weather_rmse_single_fengqing",
    description="确定性连续量检验（FengQing × ERA5）：10 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_phys",
        "source_id": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": VARS,
        "step_hours": 6.0,      # lead_from="index"：文件名 001.nc -> +6h
        # q 落盘 kg/kg，×1000 到报告单位 g/kg（实况侧已换算）
        "scales": {"q700": 1000.0},
    },
    observation_reader={
        "type": "era5_zarr",
        "stores": {
            "pl": f"{_ERA5_STORE_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
            "sfc": f"{_ERA5_STORE_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        },
        "variables": VARS,
    },
    reference_reader={
        "type": "daily_climatology",
        "root_dir": "/workspace/data/worm/era5_clim_phys_14.nc",
        "smooth_days": 15,
    },

    # ── 时段 ──
    start_date="20250102",
    end_date="20251215",
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=_OUTPUT_DIR,
    metric_options=METRIC_OPTIONS,
    log_level="INFO",
    writers=["csv_long", "spectrum"],

    # ── 协议参数 ──
    options={
        # sample_by="init_lead"：逐 (起报, 时效) 一个样本
        "sample_by": "init_lead",
    },

    # ── 执行 ──
    execution={
        "mode": "processes",
        "n_workers": 18,        # 沿用归档实测值
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "window:16",
            "reference": "resident",
        },
        "resume": True,
    },
)
