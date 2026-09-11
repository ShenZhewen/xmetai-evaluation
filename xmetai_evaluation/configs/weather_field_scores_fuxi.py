# -*- coding: utf-8 -*-
"""确定性连续量检验（FuXi × CRA40）：RMSE / ACC / 预报活跃度 / 纬向谱。

流程：``weather_field_scores``（格点有效时刻配对；集合先降维）。
对标 ``ref/tiqnqi/xmetai_model_verification_xu`` 的 run_rmse.py / verify_pangu.py：
z500（500hPa 位势高度）为核心要素，谱为纬向波数谱。

ACC / 预报活跃度是距平量，需要气候态参考：设了 ``CRA_CLI_ROOT`` 才启用，
否则用零场兜底（状态标 partial 并给警告，结果没有意义）。

实况用 CRA40 再分析（框架现有 ``cra`` reader，gh@500hPa）；xu 库对标的是
ERA5/ART 实况，如需切换需另加 reader。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="weather_field_scores_fuxi",
    description="weather 确定性连续量检验：z500 的 RMSE / ACC / 活跃度 / 纬向谱",
    pipeline="weather_field_scores",

    forecast_reader={
        "type": "fuxi",
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
        ),
        "variable": "z500",
    },
    observation_reader={
        "type": "cra",
        "root_dir": os.environ.get("CRA_ROOT", "/workspace/data/cra_root"),
        "variable": "z500",
    },
    reference_reader=(
        {"type": "climatology", "root_dir": os.environ["CRA_CLI_ROOT"]}
        if os.environ.get("CRA_CLI_ROOT")
        else None
    ),

    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251231"),

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_field_scores_fuxi",
    ),
    # 配置里的 writers 会**替换**模板自带的那一份，所以要把模板的 details 一起写上，
    # 否则逐波数谱曲线没有落盘的地方
    writers=["csv_long", "details", "json"],
    log_level="INFO",
)
