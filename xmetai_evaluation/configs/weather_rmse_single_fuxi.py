# -*- coding: utf-8 -*-
"""确定性连续量检验（FuXi × ERA5）：RMSE / 谱 / ACC / 活跃度。
z500 报 m²/s²（不除 g）、q 报 g/kg、tp 报 mm；ws850 由 sqrt(u²+v²) 现合成，ws10m 取 ERA5 10m 风速。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 要素表 ──
# ws850 现合成、ws10m 实况直取。
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

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

cfg = EvalConfig(
    name="weather_rmse_single_fuxi",
    description="确定性连续量检验（FuXi × ERA5）：12 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
        ),
        "variables": VARS,
        "step_hours": 6.0,
    },
    observation_reader={
        "type": "era5_zarr",
        "stores": {
            "pl": os.environ.get(
                "ERA5_PL_STORE",
                f"{_ERA5_STORE_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
            ),
            "sfc": os.environ.get(
                "ERA5_SFC_STORE",
                f"{_ERA5_STORE_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
            ),
        },
        "variables": VARS,
    },
    # smooth_days=15：±7.5 天中心滑动（跨年首尾相接）
    reference_reader={
        "type": "daily_climatology",
        "root_dir": os.environ.get("ERA5_CLIMO", "/workspace/data/worm/era5_clim_phys_14.nc"),
        "smooth_days": 15,  # ±7.5 天中心滑动、跨年首尾相接，抹平单日气候态噪声
    },

    # ── 时段 ──
    # 20250102 .. 20251215（348 天）
    start_date=os.environ.get("START_DATE", "20250102"),
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,  # 限起报数：冒烟改 2；None = 不限。改回 None 重跑时 resume
                 # 会复用冒烟算过的块，不白算

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_rmse_single_fuxi",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",  # DEBUG / INFO / WARNING / ERROR

    # ── 输出视图 ──
    # writers：csv_long 主表，spectrum 逐波数谱曲线。
    writers=["csv_long", "spectrum"],

    # ── 协议参数 ──
    options={
        # sample_by="init_lead"：逐 (起报, 时效) 一个样本
        "sample_by": "init_lead",
        # 三分带：热带 ±20°，南/北中高纬到 ±90°（闭区间，拼满全球）
        "regions": {
            "tropics": {"lat_min": -20, "lat_max": 20},
            "nh_extratropics": {"lat_min": 20, "lat_max": 90},
            "sh_extratropics": {"lat_min": -90, "lat_max": -20},
        },
    },

    # ── 执行 ──
    execution={
        "mode": "processes",
        "n_workers": 24,       # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "window:16",
            "reference": "resident",     # 日序气候态整 run 驻留（实测 67G，fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
