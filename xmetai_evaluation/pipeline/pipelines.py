# -*- coding: utf-8 -*-
"""流程模板表：pipeline 规定"怎么算"。

一份流程只声明 protocol / transforms / metrics / 输出视图与协议口径参数
（区域、权重、窗口语义）；"数据从哪来、评哪段时间"由配置（``--config``）给定。

两条用法::

    python -m xmetai_evaluation --list-pipelines            # 看能力清单
    python -m xmetai_evaluation --config my_config.py       # 配置自己声明 pipeline

目前 9 条流程，按三大业务块统一前缀（fdp_ / weather_ / clim_）：

    weather_ts_det           确定性降水分类检验（站点，TS/POD/FAR/BIAS）
    weather_ts_ens           集合降水分类检验（站点）：集合平均 TS/POD/FAR + 逐成员概率 AROC/BS/BSS
    weather_field_scores     确定性连续量检验：RMSE / ACC / 活跃度 / 纬向谱（对标 xu 库）
    weather_ens_crps         集合检验：CRPS / Spread-Error Ratio（对标 xu 库）
    fdp_ens_crps             集合检验：CRPS / Spread-Error Ratio
    fdp_field_scores         要素检验：RMSE / Bias / ACC（ACC 需气候态）
    fdp_precip_ts            降水检验：TS/Bias（FSS 待实现）
    fdp_precip_fss           降水空间检验：FSS（多邻域窗口）
    fdp_activity_spectrum    活跃度比 / 功率谱（二维谱）
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.pipeline.spec import (
    MetricSpec,
    PipelineTemplate,
    TransformSpec,
)

TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0]
PROB_THRESHOLDS = [0.1, 4.0, 13.0, 25.0]
CHINA_REGION = {"lat": [15.0, 55.0], "lon": [70.0, 140.0], "name": "china"}


PIPELINE_TEMPLATES: Dict[str, PipelineTemplate] = {
    "weather_ts_det": PipelineTemplate(
        name="weather_ts_det",
        protocol="station_valid_time",
        description="确定性降水分类检验（站点）：TS/POD/FAR/频率偏差",
        transforms=[
            TransformSpec("time_window_accumulator", {"window_hours": 24}),
            TransformSpec("grid_to_station", {"method": "bilinear"}),
        ],
        metrics=[MetricSpec("ts_score", {"thresholds": TS_THRESHOLDS})],
        writers=["csv_long", "categorical_wide"],
    ),
    "weather_ts_ens": PipelineTemplate(
        name="weather_ts_ens",
        protocol="station_valid_time",
        description="集合降水分类检验（站点）：集合平均 TS/POD/FAR + 逐成员概率 AROC/BS/BSS",
        transforms=[
            TransformSpec("ensemble_mean"),
            TransformSpec("time_window_accumulator", {"window_hours": 24}),
            TransformSpec("grid_to_station", {"method": "bilinear"}),
        ],
        metrics=[
            MetricSpec("ts_score", {"thresholds": TS_THRESHOLDS}),
            MetricSpec("ensemble_probability", {"thresholds": PROB_THRESHOLDS}),
        ],
        writers=["csv_long", "categorical_wide", "probability_wide"],
    ),
    "weather_field_scores": PipelineTemplate(
        name="weather_field_scores",
        protocol="grid_valid_time",
        description="确定性连续量检验：RMSE / ACC / 预报活跃度 / 纬向谱（对标 xu 库）",
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[
            MetricSpec("rmse"),
            MetricSpec("acc"),
            MetricSpec("activity"),
            MetricSpec("zonal_spectrum"),
        ],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "weather_ens_crps": PipelineTemplate(
        name="weather_ens_crps",
        protocol="grid_valid_time",
        description="集合检验：CRPS / Spread-Error Ratio（对标 xu 库）",
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[MetricSpec("crps"), MetricSpec("spread_error")],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "fdp_ens_crps": PipelineTemplate(
        name="fdp_ens_crps",
        protocol="grid_valid_time",
        description="集合检验：CRPS / Spread-Error Ratio（纬度加权，全球）",
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[MetricSpec("crps"), MetricSpec("spread_error")],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "fdp_field_scores": PipelineTemplate(
        name="fdp_field_scores",
        protocol="grid_valid_time",
        description="要素检验：RMSE / Bias / ACC（ACC 需要气候态参考）",
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[MetricSpec("rmse"), MetricSpec("bias"), MetricSpec("acc")],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "fdp_precip_ts": PipelineTemplate(
        name="fdp_precip_ts",
        protocol="station_valid_time",
        description="降水检验（中国区域站点，cos(lat) 加权）：TS/Bias",
        transforms=[TransformSpec("grid_to_station", {"method": "bilinear"})],
        metrics=[MetricSpec("ts_score", {"thresholds": [0.1, 13.0, 25.0]})],
        writers=["csv_long", "categorical_wide"],
        options={
            "region": CHINA_REGION,
            "weights": "cos_lat",
            "observation_window": "reference",
            "window_hours": 6,
            "local_utc_offset_hours": 0,
        },
    ),
    "fdp_precip_fss": PipelineTemplate(
        name="fdp_precip_fss",
        protocol="grid_valid_time",
        description="降水空间检验（网格邻域）：FSS（需网格降水实况，如 CRA/CMPAS）",
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[
            MetricSpec(
                "fss",
                {"thresholds": [13.0], "windows": [1, 3, 5, 15, 31, 63]},
            )
        ],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "fdp_activity_spectrum": PipelineTemplate(
        name="fdp_activity_spectrum",
        protocol="grid_valid_time",
        description="Z500 活跃度比 / 功率谱（活跃度比需要气候态参考）",
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[MetricSpec("activity"), MetricSpec("spectrum")],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
}

def list_pipelines() -> Dict[str, str]:
    """列出流程：名字 -> 说明。"""
    return {name: template.description for name, template in PIPELINE_TEMPLATES.items()}


def get_template(name: str) -> PipelineTemplate:
    """按名字取流程模板。

    返回的是**副本**：模板是全局共享对象，调用方不能就地修改它
    （否则会污染后续所有评测）。

    Raises:
        ConfigError: 流程名未注册。
    """
    if name not in PIPELINE_TEMPLATES:
        raise ConfigError(
            f"未知流程 '{name}'；可用流程: {', '.join(sorted(PIPELINE_TEMPLATES))}"
        )
    template = PIPELINE_TEMPLATES[name]
    return replace(
        template,
        transforms=list(template.transforms),
        metrics=list(template.metrics),
        writers=list(template.writers),
        options=dict(template.options),
    )
