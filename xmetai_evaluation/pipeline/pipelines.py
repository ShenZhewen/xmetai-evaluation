# -*- coding: utf-8 -*-
"""流程模板表：pipeline 规定"怎么算"。

一份流程只声明 protocol / transforms / metrics / 输出视图与协议口径参数
（区域、权重、窗口语义）；"数据从哪来、评哪段时间"由配置（``--config``）给定。

两条用法::

    python -m xmetai_evaluation --list-pipelines            # 看能力清单
    python -m xmetai_evaluation --config my_config.py       # 配置自己声明 pipeline

9 条流程按三大业务块统一前缀：``fdp_``（业务天气评测）、``weather_``（天气模型
验证）、``clim_``（气候，待落地）。每条模板的 ``description`` 就是
``--list-pipelines`` 打印的内容（含用途、输入数据、计算口径、指标阈值与产出文件），
所以这里不再另抄一份流程清单。阈值之类的数字直接引用本文件的常量，改常量即生效。
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
        description=(
            "确定性降水分类检验（站点）：TS/POD/FAR/频率偏差\n"
            "  用途  站点逐站检验确定性降水：落区与量级是否达标\n"
            "  数据  预报 fuxi（tp，mm）/ 观测 station（Diamond 站点降水）/ 参考 无\n"
            "  计算  station_valid_time：24h 累积 → 双线性插值到站，按有效时刻配对\n"
            "        时区按默认 +8 对齐观测（Diamond 站点观测是北京时）\n"
            f"  指标  ts_score  thresholds={TS_THRESHOLDS}\n"
            "        逐阈值逐时效给 TS / POD / FAR / 漏报率 / 频率偏差 BIAS\n"
            "        产出 scores.csv、diagnostics/categorical_wide.csv"
        ),
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
        description=(
            "集合降水分类检验（站点）：集合平均 TS/POD/FAR + 逐成员概率 AROC/BS/BSS\n"
            "  用途  两路检验：集合平均场算 TS 系列；逐成员算概率评分\n"
            "  数据  预报 fuxi_ens（tp，逐 6h，member_* 子目录）/ 观测 station\n"
            "        参考 可选 ref_probability（ref/MMDDHH.000 目录，BSS 用）\n"
            "  计算  station_valid_time：集合平均 → 24h 累积 → 双线性插值到站\n"
            f"  指标  ts_score  thresholds={TS_THRESHOLDS}\n"
            f"        ensemble_probability  thresholds={PROB_THRESHOLDS}\n"
            "        概率按超越式口径（x ≥ 阈值）；BS_ref 默认样本气候频率 r(1-r)，\n"
            "        有外部气候概率时改为 mean((p_clim-o)^2)（仅 6h 窗口生效）\n"
            "        产出 scores.csv、diagnostics/categorical_wide.csv、\n"
            "        diagnostics/probability_wide.csv"
        ),
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
        description=(
            "确定性连续量检验：RMSE / ACC / 预报活跃度 / 纬向谱\n"
            "  用途  四个角度：误差量级、距平空间型、平滑程度、能量谱分布\n"
            "  数据  预报 fuxi（如 z500；输入 m²/s²，按 1/g 换算为 m）\n"
            "        观测 cra（CRA40，如 gh@500hPa）\n"
            "        参考 climatology —— ACC / 活跃度必需，纬向谱不需要\n"
            "  计算  grid_valid_time：格点有效时刻配对；集合先取平均\n"
            "        缺气候态时用零场兜底，状态标 partial 且结果无意义\n"
            "  指标  rmse、acc、activity、zonal_spectrum（max_k=30）\n"
            "        产出 scores.csv"
        ),
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
        description=(
            "集合检验：CRPS / Spread-Error Ratio\n"
            "  用途  CRPS 看集合分布离实况多远；Spread-Error 比看离散度是否标定\n"
            "        ≈1 标定良好，<1 过度自信，>1 欠自信\n"
            "  数据  预报 fuxi_ens / 观测 cra（CRA40）/ 参考 无\n"
            "  计算  grid_valid_time：格点有效时刻配对，纬度加权、全球\n"
            "        CRPS 用闭式解，逐点只统计有限成员，缺测成员不参与\n"
            "        不做非负截断\n"
            "  指标  crps、spread_error（Spread / 集合平均 RMSE / 两者之比）\n"
            "        产出 scores.csv"
        ),
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[MetricSpec("crps"), MetricSpec("spread_error")],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "fdp_ens_crps": PipelineTemplate(
        name="fdp_ens_crps",
        protocol="grid_valid_time",
        description=(
            "集合检验：CRPS / Spread-Error Ratio（纬度加权，全球）\n"
            "  用途  FDP 业务线集合连续评分；指标口径与 weather_ens_crps 相同\n"
            "        （共用同一套 metric 实现），差别在业务线与默认数据源\n"
            "  数据  预报 fengqing（集合）/ 观测 cra（CRA40）/ 参考 无\n"
            "  计算  grid_valid_time：格点有效时刻配对；集合先取平均\n"
            "  指标  crps、spread_error\n"
            "        产出 scores.csv"
        ),
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[MetricSpec("crps"), MetricSpec("spread_error")],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "fdp_field_scores": PipelineTemplate(
        name="fdp_field_scores",
        protocol="grid_valid_time",
        description=(
            "要素检验：RMSE / Bias / ACC（ACC 需要气候态参考）\n"
            "  用途  RMSE 误差量级、Bias 系统性偏差方向、ACC 距平场空间型\n"
            "  数据  预报 fengqing（如 z500）/ 观测 cra（CRA40）\n"
            "        参考 climatology —— ACC 需要，RMSE / Bias 不需要\n"
            "  计算  grid_valid_time：格点有效时刻配对；集合先取平均\n"
            "        ACC 为经典皮尔逊口径（距平场再减域加权均值）；\n"
            "        FDP/WeatherBench2 的 uncentered 口径是 acc_uncentered\n"
            "  指标  rmse、bias、acc\n"
            "        产出 scores.csv"
        ),
        transforms=[TransformSpec("ensemble_mean")],
        metrics=[MetricSpec("rmse"), MetricSpec("bias"), MetricSpec("acc")],
        writers=["csv_long"],
        options={"ensemble_reduction": "mean"},
    ),
    "fdp_precip_ts": PipelineTemplate(
        name="fdp_precip_ts",
        protocol="station_valid_time",
        description=(
            "降水检验（中国区域站点，cos(lat) 加权）：TS/Bias\n"
            "  用途  中国区站点降水：TS 看命中与空报的折中，BIAS 看整体偏多偏少\n"
            "  数据  预报 格点降水 / 观测 station（Diamond 站点降水）/ 参考 无\n"
            "  计算  station_valid_time：双线性插值到站；6h 窗口，不经 24h 累积\n"
            "        区域 15-55N / 70-140E；站点按 cos(lat) 加权\n"
            "        窗口不要求观测完整；按 UTC 对齐\n"
            "        （local_utc_offset_hours=0；默认是北京时 +8）\n"
            "  指标  ts_score  thresholds=[0.1, 13.0, 25.0]\n"
            "        产出 scores.csv、diagnostics/categorical_wide.csv"
        ),
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
        description=(
            "降水空间检验（网格邻域）：FSS（需网格降水实况，如 CRA/CMPAS）\n"
            "  用途  邻域平滑后比对，对“落区对但位置略有偏差”更宽容；\n"
            "        看技巧随尺度的衰减（窗口越大越接近随机基准）\n"
            "  数据  预报 格点降水 / 观测 格点降水实况（CRA / CMPAS）/ 参考 无\n"
            "        这条流程不吃站点观测\n"
            "  计算  grid_valid_time：格点有效时刻配对；集合先取平均\n"
            "  指标  fss  thresholds=[13.0]、windows=[1, 3, 5, 15, 31, 63]\n"
            "        产出 scores.csv"
        ),
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
        description=(
            "Z500 活跃度比 / 功率谱（活跃度比需要气候态参考）\n"
            "  用途  活跃度比 = 预报距平标准差 / 实况，看偏平滑（<1）还是偏噪（>1）；\n"
            "        功率谱看能量在各波数的分布是否失真\n"
            "  数据  预报 z500 / 观测 格点实况 / 参考 climatology（活跃度比必需）\n"
            "  计算  grid_valid_time：格点有效时刻配对；缺气候态则零场兜底、无意义\n"
            "        谱口径与 weather_field_scores 的 zonal_spectrum 不同：\n"
            "        这里对原始场做二维 FFT、不减纬向均值\n"
            "  指标  activity（含 FC/OBS 活跃度与 BIAS）、spectrum（max_k=30）\n"
            "        产出 scores.csv"
        ),
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
