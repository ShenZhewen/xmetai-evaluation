# -*- coding: utf-8 -*-
"""集合场检验（FuXi 集合 × ERA5）：对标 xu ``scripts/ensemble_fuxi.sh``。

流程：``weather_ens_field_scores``（确定性误差 + 集合分布质量同跑）。
xu 这条脚本的五种指标横跨"确定性"与"概率"两类，所以既不能套
``weather_field_scores``（无 crps），也不能套 ``weather_ens_crps``（无谱/ACC）。

    xu --metrics rmse crps acc fa spectrum  ->  rmse / crps / acc / activity / zonal_spectrum
    xu --vars z500 msl u200 v200 ws200      ->  VARS
    xu --var-metrics z500:rmse,crps,acc,fa,spectrum ...  ->  VAR_METRICS

注意 crps 只路由给 z500（xu 也只对它算）：CRPS 直接吃原始成员，不走集合均值。

单位口径（与 xu 报告层一致，便于逐格对拍）：
    z500 报 m²/s²（不除 g）、msl 报 Pa、风报 m/s。
``ws200`` 由 ``sqrt(u200²+v200²)`` 现合成（预报、实况、气候态三侧都合成）。

内存：``grid_valid_time`` 会一次读入评测时段内全部有效时刻；集合还多一份成员场。
默认给的是冒烟级小区间，正式跑直接改下面的 ``start_date`` / ``end_date`` / ``limit``。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# xu: --vars z500 msl u200 v200 ws200
VARS = ["z500", "msl", "u200", "v200", "ws200"]

# xu: --var-metrics（逐条转写；fa -> activity、spectrum -> zonal_spectrum）
VAR_METRICS = {
    "z500": ["rmse", "crps", "acc", "activity", "zonal_spectrum"],
    "msl": ["rmse", "acc"],
    "u200": ["rmse", "activity", "zonal_spectrum"],
    "v200": ["rmse", "activity", "zonal_spectrum"],
    "ws200": ["rmse", "activity", "zonal_spectrum"],
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
    name="weather_ens_field_scores_era5_fuxi",
    description="集合场检验（FuXi 集合 × ERA5）：z500/msl/u200/v200/ws200 的 RMSE / CRPS / ACC / 活跃度 / 谱",
    pipeline="weather_ens_field_scores",

    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fuxi_ens_output"
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
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_ens_field_scores_era5_fuxi",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",
)
