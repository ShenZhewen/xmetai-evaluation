# -*- coding: utf-8 -*-
"""确定性降水分类检验（FGVP 模型 × Diamond 站点）。

流程：``ts_det``（站点有效时刻配对 + 24h 累积 + 网格到站点插值 + TS 系列）。
本文件只声明"数据从哪来、评哪段时间、写到哪"，算法口径全在流程模板里。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

#: TS 阈值（含 ≥250mm 的极端量级，与参考评测一致）
TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]

cfg = EvalConfig(
    name="ts_det_fgvp",
    description="FGVP 确定性降水 TS 评估（与站点观测对比）",
    pipeline="ts_det",

    forecast_reader={
        "type": "fuxi",
        "root_dir": os.environ.get("FGVP_OUTPUT", "/workspace/data/shenzw/fgvp_output"),
        "variable": "tp",
    },
    observation_reader={
        "type": "station",
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",
        # 与参考评测一致：仅用 zd_sta_10285.dat 的站点；留空则用全部站点
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ) or None,
    },

    # 流程模板的参数覆盖（不写就用模板默认）
    transform_options={"time_window_accumulator": {"window_hours": 24}},
    metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},

    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251231"),
    limit=None,

    output_dir=os.environ.get("EVAL_OUTPUT", "evaluation_results/ts_det_fgvp"),
    writers=["csv_long", "categorical_wide"],
    log_level="INFO",
)
