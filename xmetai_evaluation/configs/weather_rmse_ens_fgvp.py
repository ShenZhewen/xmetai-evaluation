# -*- coding: utf-8 -*-
"""集合场检验（FuXi 集合 × ERA5）：确定性误差、CRPS / spread 与谱。
z500 报 m²/s²（不除 g）、msl 报 Pa、风报 m/s；ws200 由 sqrt(u²+v²) 现合成。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 要素表 ──
VARS = ["z500", "msl", "u200", "v200", "ws200"]

# ── 逐变量指标表 ──
# crps 只路由给 z500；spread_error 每个变量都点名（对齐老档案）。
VAR_METRICS = {
    "z500": ["rmse", "crps", "spread_error", "acc", "activity",
             "zonal_spectrum", "spherical_bands"],
    "msl": ["rmse", "spread_error", "acc"],
    "u200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
    "v200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
    "ws200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# centered=False（uncentered / FDP 口径，对齐 xu）
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# max_wavenumber=720（全球 0.25° Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}
# spherical_bands 用默认分带（老仓 (1,4)/(5,20)/(21,40)/(41,64)/(65,128)），对拍别动。
# ddof=0（除 N，对齐 ref_result 的 spread_*.csv）
METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

cfg = EvalConfig(
    name="weather_rmse_ens_fgvp",
    description="集合场检验（FuXi 集合 × ERA5）：z500/msl/u200/v200/ws200 的 RMSE / CRPS / ACC / 活跃度 / 谱 / 球谐带功率",
    pipeline="weather_ens_field_scores",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fgvp_output"
        ),
        "variables": VARS,
        "step_hours": 6.0,
        # lead_times=None：不限制，从目录索引推算
        "lead_times": None,
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
    periods=[
        ("20250102", "20251215")
    ],
    limit=None,  # 限起报数：冒烟改 2；None = 不限。改回 None 重跑时 resume
                 # 会复用冒烟算过的块，不白算

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/data/shenzw/evaluation_results/weather_rmse_single_fgvp",
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
        # regions 为空：只出全球行
        "regions": {
        },
    },

    # ── 执行 ──
    execution={
        "mode": "processes",
        "n_workers": 18,       # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "window:16",
            "reference": "resident",     # 日序气候态整 run 驻留（z500 实测 5.6G，fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
