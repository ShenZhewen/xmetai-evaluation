# -*- coding: utf-8 -*-
"""确定性连续量检验（风清单卡 × ERA5）：对标 xu ``scripts/single_fengqing.sh``。

与 ``weather_field_scores_era5_fuxi.py`` 同流程、同单位口径，差别只有两处数据事实：
风清单卡输出里没有 q2m，且地面风变量名是 ``U10``/``V10``（由布局翻译成
标准的 ``u10m``/``v10m``，与 ERA5 zarr 对齐）。

    xu --vars z500 q700 t700 t2m t850 msl u200 v200 u850 v850 u10m v10m ws10m ws850 ws200
    xu --var-metrics ...（transcribed below；fa -> activity、spectrum -> zonal_spectrum）

单位口径（与 xu 报告层一致，便于逐格对拍）：
    z500 报 m²/s²（不除 g）、q 报 g/kg、tp 报 mm。

``ws10m`` / ``ws850`` / ``ws200`` 由 u/v 分量按 ``sqrt(u²+v²)`` 现合成。

内存：``grid_valid_time`` 会一次读入评测时段内全部有效时刻，默认给的是
冒烟级小区间；正式跑直接改下面的 ``start_date`` / ``end_date`` / ``limit``。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# xu: --vars z500 q700 t700 t2m t850 msl u200 v200 u850 v850 u10m v10m ws10m ws850 ws200
VARS = [
    "z500", "q700", "t700", "t2m", "t850", "msl",
    "u200", "v200", "u850", "v850", "u10m", "v10m",
    "ws10m", "ws850", "ws200",
]

# xu: --var-metrics（逐条转写；fa -> activity、spectrum -> zonal_spectrum）
VAR_METRICS = {
    "z500": ["rmse", "zonal_spectrum", "acc", "activity"],
    "q700": ["rmse"],
    "t700": ["rmse", "zonal_spectrum"],
    "t2m": ["rmse", "zonal_spectrum"],
    "t850": ["rmse"],
    "msl": ["rmse", "zonal_spectrum"],
    "u200": ["rmse", "activity", "zonal_spectrum"],
    "v200": ["rmse", "activity", "zonal_spectrum"],
    "u850": ["rmse", "activity", "zonal_spectrum"],
    "v850": ["rmse", "activity", "zonal_spectrum"],
    "u10m": ["rmse", "activity"],
    "v10m": ["rmse", "activity"],
    "ws10m": ["activity", "rmse"],
    "ws850": ["activity", "rmse", "zonal_spectrum"],
    "ws200": ["rmse", "zonal_spectrum"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# xu 的 ACC 是 uncentered（FDP / WeatherBench2 口径），不是经典皮尔逊
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# xu 的谱取到 720 波（全球 0.25° 的 Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

cfg = EvalConfig(
    name="weather_field_scores_era5_fengqing",
    description="确定性连续量检验（风清单卡 × ERA5）：15 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",

    forecast_reader={
        "type": "fengqing_phys",
        "root_dir": os.environ.get(
            "FENGQING_OUTPUT", "/workspace/data/shenzw/fengqing_output"
        ),
        "variables": VARS,
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
    reference_reader={
        "type": "daily_climatology",
        "root_dir": os.environ.get("ERA5_CLIMO", "/workspace/data/worm/era5_clim_phys_14.nc"),
        "window": 15,
    },

    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20250102"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_field_scores_era5_fengqing",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",
)
