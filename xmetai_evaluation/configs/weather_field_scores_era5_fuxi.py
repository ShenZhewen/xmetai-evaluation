# -*- coding: utf-8 -*-
"""确定性连续量检验（FuXi × ERA5）：对标 xu ``scripts/single_fuxi.sh``。

流程：``weather_field_scores``（格点有效时刻配对；集合先降维）。
要素表与逐变量指标表**逐条转写** xu 的命令行，只是把名字换成框架里的对应物：

    xu --metrics rmse spectrum acc fa   ->  rmse / zonal_spectrum / acc / activity
    xu --vars  ...                      ->  VARS
    xu --var-metrics z500:rmse,acc,fa   ->  VAR_METRICS（再由
                                            metric_options_from_var_metrics 反转）

单位口径也对齐 xu 的报告层，所以结果 CSV 能逐格对拍：
    z500 报 m²/s²（**不除 g**，xu 的 z500 单位就是 ``m2 s-2``）
    q    报 g/kg（源文件已是 g/kg；xu 内部转 kg/kg、报告层再 ×1000，两步相消）
    tp   报 mm（ERA5 zarr 源是 m，已在 layout 里 ×1000）

``ws10m`` / ``ws850`` / ``ws200`` 任何文件里都没有，由 u/v 分量按
``sqrt(u²+v²)`` 现合成（预报、实况、气候态三侧都合成）。

xu 默认把 tp 排除在格点指标外（走站点 TS）。要复刻它的 ``--tp-grid``，
把 ``"tp"`` 加进下面的 VARS 和 VAR_METRICS 即可。

内存：``grid_valid_time`` 会把评测时段内**全部**有效时刻一次读进来，
16 个要素 × 全球 0.25° 很吃内存。默认给的是冒烟级小区间（1 个起报日），
正式跑直接改下面的 ``start_date`` / ``end_date`` / ``limit``。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# xu: --vars z500 q2m q700 t700 t2m t850 msl u200 v200 u850 v850 u10m v10m ws10m ws850 ws200
VARS = [
    "z500", "q2m", "q700", "t700", "t2m", "t850", "msl",
    "u200", "v200", "u850", "v850", "u10m", "v10m",
    "ws10m", "ws850", "ws200",
]

# xu: --var-metrics（逐条转写；fa -> activity、spectrum -> zonal_spectrum）
VAR_METRICS = {
    "z500": ["rmse", "zonal_spectrum", "acc", "activity"],
    "q2m": ["rmse"],
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
    "ws10m": ["rmse"],
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
    name="weather_field_scores_era5_fuxi",
    description="确定性连续量检验（FuXi × ERA5）：16 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",

    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
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
        # 与 xu 的 --climo-window 一致：15 天环形平滑
        "window": 15,
    },

    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20250102"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_field_scores_era5_fuxi",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",
)
