# -*- coding: utf-8 -*-
"""要素检验（Fengqing × CRA40）：RMSE / Bias / ACC。

流程：``fdp_field_scores``（格点有效时刻配对；集合先降维）。
ACC 是距平相关，需要气候态参考：设了 ``CRA_CLI_ROOT`` 才启用，
否则用零场兜底（状态标 partial 并给警告，结果没有意义）。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="fdp_field_scores_fengqing",
    description="FDP 要素检验：z500 的 RMSE / Bias / ACC",
    pipeline="fdp_field_scores",

    forecast_reader={
        "type": "fengqing",
        "root_dir": os.environ.get(
            "FDP_FENGQING_ROOT", "/mnt/d/气象研究/春燕_weather_test_bash/fengqing"
        ),
        "variables": ["z500"],
        "init_times": ["2026-08-19T00:00:00"],
        "lead_times": [24, 48, 72, 96, 120],
    },
    observation_reader={
        "type": "cra",
        "root_dir": os.environ.get(
            "FDP_CRA_ROOT", "/mnt/d/气象研究/春燕_weather_test_bash/cra_root"
        ),
        "variables": ["z500"],
    },
    reference_reader=(
        {"type": "climatology", "root_dir": os.environ["CRA_CLI_ROOT"]}
        if os.environ.get("CRA_CLI_ROOT")
        else None
    ),

    start_date="20260819",
    end_date="20260819",
    output_dir=os.environ.get(
        "EVAL_OUTPUT", "evaluation_results/fdp_field_scores_fengqing"
    ),
    writers=["csv_long", "json"],
    log_level="INFO",
)
