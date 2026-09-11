# -*- coding: utf-8 -*-
"""集合降水分类检验（FuXi 集合 × Diamond 站点）。

流程：``weather_ts_ens``（集合平均 → 窗口累积 → 插值到站点；TS 系列 + 概率评分 AROC/BS/BSS）。
数据：``{FUXI_ENS_OUTPUT}/YYYYMMDD/member_*/001.nc…``（TP，逐 6h）。
BSS 参考：设 ``BSS_REF``（ref/MMDDHH.000 目录）且 ``WINDOW_HOURS=6`` 时用外部气候概率，
否则回退样本气候频率 r(1-r)。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig


def _window_hours() -> float:
    """累积窗口：``WINDOW_HOURS=6`` 切到 6h 口径（参考实现的 AROC/BSS 用 6h）。"""
    return float(os.environ.get("WINDOW_HOURS", "24"))


def _lead_times():
    """``LEAD_TIMES=6,12,18,24`` 只读部分时效，显著降低内存与耗时。"""
    raw = os.environ.get("LEAD_TIMES", "").strip()
    if not raw:
        return None
    return [float(item) for item in raw.split(",") if item.strip()]


def _writers():
    """输出视图：默认最小集；``WRITERS=csv_long,categorical_wide,probability_wide`` 打开宽表。"""
    raw = os.environ.get("WRITERS", "csv_long")
    return [name.strip() for name in raw.split(",") if name.strip()]


WINDOW = _window_hours()

#: TS 阈值（对齐参考实现 DEFAULT_THRESHOLDS）
TS_THRESHOLDS = {
    6.0: [0.1, 13.0, 25.0],
    24.0: [0.1, 10.0, 25.0, 50.0, 100.0],
}.get(WINDOW, [0.1, 10.0, 25.0, 50.0, 100.0])

#: 概率评分阈值（对齐参考实现 DEFAULT_AROC_THRESHOLDS）
PROB_THRESHOLDS = {
    6.0: [0.1, 4.0, 13.0, 25.0],
    24.0: [0.1, 4.0, 13.0, 25.0],
}.get(WINDOW, [0.1, 4.0, 13.0, 25.0])


def _reference_reader():
    """BSS 外部气候概率参考（可选，仅 6h 窗口径，对标原版 ``--ref``）。

    设 ``BSS_REF`` 指向 ref/MMDDHH.000 目录、且 ``WINDOW_HOURS=6`` 时才启用；
    否则 BSS 回退样本气候频率 r(1-r)。
    """
    if WINDOW == 6.0 and os.environ.get("BSS_REF"):
        return {"type": "ref_probability", "root_dir": os.environ["BSS_REF"]}
    return None


cfg = EvalConfig(
    name="weather_ts_ens_fuxi",
    description="FuXi 集合降水评估：集合平均 TS + 概率评分 AROC/BS/BSS",
    pipeline="weather_ts_ens",

    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fuxi_ens_output"
        ),
        "variable": "tp",
        "step_hours": 6.0,
        "lead_times": _lead_times(),
    },
    observation_reader={
        "type": "station",
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ) or None,
    },
    reference_reader=_reference_reader(),

    transform_options={"time_window_accumulator": {"window_hours": WINDOW}},
    metric_options={
        "ts_score": {"thresholds": TS_THRESHOLDS},
        "ensemble_probability": {"thresholds": PROB_THRESHOLDS},
    },

    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251231"),

    output_dir=os.environ.get("EVAL_OUTPUT", "evaluation_results/weather_ts_ens_fuxi"),
    writers=_writers(),
    log_level="INFO",
)
