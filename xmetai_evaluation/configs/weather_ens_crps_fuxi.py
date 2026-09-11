# -*- coding: utf-8 -*-
"""集合连续评分（FuXi 集合 × CRA40）：CRPS / Spread-Error Ratio。

流程：``weather_ens_crps``（格点有效时刻配对）。
对标 ``ref/tiqnqi/xmetai_model_verification_xu`` 的 run_rmse.py --summarize-ens：
集合 z500 的 CRPS 与 spread/RMSE 比（纬度加权，全球）。

实况用 CRA40 再分析（框架现有 ``cra`` reader，gh@500hPa）；xu 库对标的是
ERA5/ART 实况，如需切换需另加 reader。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="weather_ens_crps_fuxi",
    description="weather 集合连续评分：z500 的 CRPS / Spread-Error 比",
    pipeline="weather_ens_crps",

    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fuxi_ens_output"
        ),
        "variable": "z500",
        "step_hours": 6.0,
    },
    observation_reader={
        "type": "cra",
        "root_dir": os.environ.get("CRA_ROOT", "/workspace/data/cra_root"),
        "variable": "z500",
    },

    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251231"),

    output_dir=os.environ.get(
        "EVAL_OUTPUT", "evaluation_results/weather_ens_crps_fuxi"
    ),
    writers=["csv_long"],
    log_level="INFO",
)
