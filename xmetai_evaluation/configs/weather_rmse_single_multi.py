# -*- coding: utf-8 -*-
"""批量确定性连续量检验：一份文件按顺序评多个模型（FuXi / 风清）× ERA5。

对标老仓 ``weather_rmse_single_multi.py`` 的格局：观测、气候态、时段、指标
全部共用，只有预报源按模型换。每个模型各落各的 ``output_dir``，产物与
对应的单模型配置（weather_rmse_single_fuxi / _fengqing）完全同构——下游
的报告与对拍按目录读，不用改任何东西。

用法（cli 认模块里的 ``cfgs`` 列表，逐个执行）：

    xmetai-eval --config weather_rmse_single_multi

单个模型失败会记日志并继续跑后面的，结束时统一报成败；全程串行，
每个 run 内部已经吃满 n_workers，两个模型并行只会超订内存。

要加第三个模型：在 MODELS 里加一行（type 用 components.py 里注册过的
reader 名），要素表不同就顺带在该条目里写自己的 vars / var_metrics。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# 冒烟级要素表（单要素）；正式跑参照 weather_rmse_single_fuxi.py 把全量
# 要素和逐变量指标表放回来——fuxi 与 fengqing 的 VAR_METRICS 不完全相同，
# 放全量时按模型分开写进 MODELS 条目里。
VARS = ["z500"]
VAR_METRICS = {"z500": ["rmse", "zonal_spectrum", "acc", "activity"]}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# xu 的 ACC 是 uncentered（FDP / WeatherBench2 口径），不是经典皮尔逊
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# xu 的谱取到 720 波（全球 0.25° 的 Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

#: 模型表：名字 -> 预报源配置。逐条只写**和别的模型不一样**的字段，
#: 共用的观测/气候态/时段/指标都在下面的 COMMON 里。
MODELS = {
    "fuxi": {
        "forecast_reader": {
            "type": "fuxi_phys",
            "root_dir": "/workspace/data/shenzw/fuxi_single_output",
            "variables": VARS,
            "step_hours": 6.0,      # 布局步长：lead_from="index"，001.nc → +6h
        },
    },
    "fengqing": {
        "forecast_reader": {
            "type": "fengqing_phys",
            "root_dir": "/workspace/data/shenzw/fengqing_output",
            "variables": VARS,
            "step_hours": 6.0,      # 布局步长：lead_from="filename"，文件名 3 位时效 × 6h
        },
    },
}

_COMMON = dict(
    pipeline="weather_field_scores",
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
        # 15 天环形平滑（±7.5 天，跨年首尾相接），与单模型配置同口径
        "smooth_days": 15,
    },
    # 冒烟时段（2 个起报）；正式跑直接改这两个字面量，如 20250102 / 20251215
    start_date="20250102",
    end_date="20250103",
    limit=None,  # 限起报数，直接改这里；None = 不限
    metric_options=METRIC_OPTIONS,
    log_level="INFO",
    writers=["csv_long", "spectrum"],
    # 采样口径：每个起报各报满整段时效，逐 (起报, 时效) 出样本
    options={"sample_by": "init_lead"},
    # 并发与数据加载：与单模型配置同参。冒烟用 serial 也行，正式跑保持 processes。
    execution={
        "mode": "processes",
        "n_workers": 24,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "window:16",
            "reference": "resident",
        },
        "resume": True,
    },
)

cfgs = [
    EvalConfig(
        name=f"weather_rmse_single_{model}",
        description=f"批量确定性连续量检验（{model} × ERA5）：z500 的 RMSE / 谱 / ACC / 活跃度",
        forecast_reader=spec["forecast_reader"],
        output_dir=(
            "/workspace/szwCode/xmetai-evaluate/evaluation_results/"
            f"weather_rmse_single_{model}"
        ),
        **_COMMON,
    )
    for model, spec in MODELS.items()
]
