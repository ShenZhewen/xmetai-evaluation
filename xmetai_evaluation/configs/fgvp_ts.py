# -*- coding: utf-8 -*-
"""FGVP 模型 TS 评估配置示例

复现 /workspace/szwCode/xmetai-evaluate/ref/tiqnqi/xmetai_model_verification/results2/en/
的 TS 评估结果，用于验证框架正确性。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="fgvp_ts",
    description="FGVP 模型降水 TS 评估（与站点观测对比）",

    # 预报数据源：FuXi 格式 NetCDF
    forecast_reader={
        "type": "fuxi",
        "root_dir": os.environ.get(
            "FGVP_OUTPUT",
            "/workspace/data/shenzw/fgvp_output",
        ),
        "variable": "tp",  # 总降水
    },

    # 观测数据源：Diamond 站点数据
    observation_reader={
        "type": "station",
        "root_dir": os.environ.get(
            "STATION_OBS",
            "/workspace/data/worm/r0/2025",
        ),
        "format": "diamond_obs",
        "variable": "precipitation",
    },

    # 变换链
    transforms=[
        # 1. 网格插值到站点
        {
            "type": "grid_to_station",
            "method": "bilinear",
        },
        # 2. 24小时时间窗口累积
        {
            "type": "time_window_accumulator",
            "window_hours": 24,
        },
    ],

    # 指标计算
    metrics=[
        {
            "type": "ts_score",
            "thresholds": [0.1, 10.0, 25.0, 50.0, 100.0, 250.0],  # mm
        },
    ],

    # 时间范围（可通过命令行覆盖）
    start_date="20250101",
    end_date="20250131",
    limit=None,  # 设为 5 可只处理前 5 个初始化时间（调试用）

    # 输出配置
    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "evaluation_results/fgvp_ts",
    ),
    output_format="csv",

    # 运行配置
    log_level="INFO",
    num_workers=1,
)
