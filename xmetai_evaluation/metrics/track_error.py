# -*- coding: utf-8 -*-
"""台风路径/强度误差指标。

输入是 ``typhoon_track`` 协议配对好的两条路径（同为 ``(lead_time, storm)``）：

- ``forecast``：诊断出来的预报中心（lat/lon/pmin/vmax）；
- ``observation``：配对上的实况中心（同上四项；对不上的时效为 NaN）。

**协议只管配对，误差在这里算**：大圆距离、沿/横路径分解、强度偏差。逐时效
走标量循环（与 ``pipeline/typhoon.py`` 里 ``match_errors`` 同一套调用），
不向量化——归档对拍要的是逐位一致，不是"数学上等价"。

沿/横路径误差（``at_km`` / ``ct_km``）的定向用**前一个配对上的实况位置**，
不是预报-实况连线，也不是起报点；所以一条路径里**第一个配对时效**的 at/ct
必为空——没有前一个实况就无从定向。实况有缺口时缺的是那个时效，不是固定 6h。

除标量外，本指标把整条逐时效表放进 ``MetricResult.curve``：一个场次一行一个
时效，15 个字段，直接喂给 ``typhoon_cases`` writer 出 ``tc<编号>_<起报>.csv``。
不走 ``value`` 是因为 value 里每个键都会展开成长表明细行，15 字段 × 60 时效
× 全年场次会把长表撑成几十万行，而曲线本来就有专门的出口。
"""

from typing import Any, Dict, List

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, MetricResult, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
)
from xmetai_evaluation.pipeline.typhoon import along_cross_track, great_circle_km

#: 逐时效表里的字段（顺序即写盘顺序，与旧归档的 csv 列一致）
CURVE_FIELDS = (
    "lead_h",
    "valid_bjt",
    "fcst_lat",
    "fcst_lon",
    "fcst_pmin_hpa",
    "fcst_vmax_ms",
    "obs_lat",
    "obs_lon",
    "obs_pmin_hpa",
    "obs_vmax_ms",
    "track_err_km",
    "at_km",
    "ct_km",
    "wind_err_ms",
    "pmin_err_hpa",
)

#: 落进长表的场次级标量（"value 键 -> 中文说明"，键名同时是 curve 里的字段名）
SUMMARY_FIELDS = ("track_err_km", "at_km", "ct_km", "wind_err_ms", "pmin_err_hpa")


def _nanmean(values: np.ndarray) -> float:
    """全 NaN 时返回 NaN，不报警告（全 NaN 是"这条路径一个时效都没配上"）。"""
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.mean(values[finite]))


class TrackError(Metric):
    """路径误差（大圆距离）+ 沿/横路径分解 + 强度偏差。"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="track_error", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    def validate(self, batch: EvaluationBatch) -> None:
        super().validate(batch)
        storm = self._storm_axis(batch)
        if storm.size != 1:
            raise MetricError(
                f"track_error 的一个批次只承载一号台风的一条路径，"
                f"这一批有 {storm.size} 号（{list(storm)}）——一个场次一个样本是 "
                f"typhoon_track 协议的约定，多条路径塞进一批会让逐场次曲线丢掉。",
                variable=self.name,
            )

    # ------------------------------------------------------------ 计算

    @staticmethod
    def _storm_axis(batch: EvaluationBatch):
        data = batch.forecast.payload if hasattr(batch.forecast, "payload") else batch.forecast
        return np.asarray(data["storm"].values)

    @staticmethod
    def _side(batch: EvaluationBatch, which: str) -> xr.Dataset:
        data = getattr(batch, which)
        return data.payload if hasattr(data, "payload") else data

    def _session_context(self, batch: EvaluationBatch) -> Dict[str, Any]:
        """从批次里取这次场次的标识与对时参数（写 curve 用）。"""
        key = dict(batch.sample_keys[0]) if batch.sample_keys else {}
        alignment = dict(batch.alignment or {})
        origin = alignment.get("init_position") or [None, None]
        return {
            "storm": str(self._storm_axis(batch)[0]),
            "tcname": str(alignment.get("tcname") or ""),
            "init_utc": str(key.get("init") or ""),
            "tz_shift": float(alignment.get("tz_shift_hours", 8.0)),
            # 起报点实况位置与搜索参数带着走：typhoon_cases writer 要把它们写进
            # 逐场次的 meta（起报点是画路径图的起点，搜索参数是复现结果的口径）。
            "init_lat": origin[0],
            "init_lon": origin[1],
            # 起始中心相对起报时刻的偏移小时（正常 6，见 babj_seed_moment）。
            # 由协议给的，只写进 meta，不参与计算。
            "seed_offset_h": float(alignment.get("seed_offset_hours", 0.0)),
            "search": dict(alignment.get("search") or {}),
        }

    def _compute(self, batch: EvaluationBatch) -> Dict[str, Any]:
        """逐时效算误差，返回 curve 形态的整张表。"""
        forecast = self._side(batch, "forecast")
        observation = self._side(batch, "observation")

        leads = np.asarray(forecast["lead_time"].values, dtype="f8")
        f_lat = np.asarray(forecast["lat"].values, dtype="f8")[:, 0]
        f_lon = np.asarray(forecast["lon"].values, dtype="f8")[:, 0]
        f_pmin = np.asarray(forecast["pmin"].values, dtype="f8")[:, 0]
        f_vmax = np.asarray(forecast["vmax"].values, dtype="f8")[:, 0]
        o_lat = np.asarray(observation["lat"].values, dtype="f8")[:, 0]
        o_lon = np.asarray(observation["lon"].values, dtype="f8")[:, 0]
        o_pmin = np.asarray(observation["pmin"].values, dtype="f8")[:, 0]
        o_vmax = np.asarray(observation["vmax"].values, dtype="f8")[:, 0]

        nan = float("nan")
        err = np.full(leads.size, nan)
        at = np.full(leads.size, nan)
        ct = np.full(leads.size, nan)
        wind = np.full(leads.size, nan)
        pmin = np.full(leads.size, nan)

        # 逐时效标量推进：与旧链路的 match_errors 同一套调用与顺序，
        # prev 只在**配对成功**的时效上更新，实况缺口不携带陈旧方向。
        prev = None
        for k in range(leads.size):
            if not np.isfinite(o_lat[k]):
                continue
            err[k] = float(great_circle_km(f_lat[k], f_lon[k], o_lat[k], o_lon[k]))
            if prev is not None:
                at[k], ct[k] = along_cross_track(
                    prev[0], prev[1], o_lat[k], o_lon[k], f_lat[k], f_lon[k]
                )
            if np.isfinite(f_vmax[k]):
                wind[k] = float(f_vmax[k] - o_vmax[k])
            pmin[k] = float(f_pmin[k] - o_pmin[k])
            prev = (o_lat[k], o_lon[k])

        context = self._session_context(batch)
        init = context["init_utc"]
        base = _parse_iso(init)
        valid = [
            str(base + _hours(float(lead) + context["tz_shift"])) if base else ""
            for lead in leads
        ]
        return {
            "kind": "typhoon_track",
            **context,
            "lead_h": [float(value) for value in leads],
            "valid_bjt": valid,
            "fcst_lat": f_lat.tolist(),
            "fcst_lon": f_lon.tolist(),
            "fcst_pmin_hpa": f_pmin.tolist(),
            "fcst_vmax_ms": f_vmax.tolist(),
            "obs_lat": o_lat.tolist(),
            "obs_lon": o_lon.tolist(),
            "obs_pmin_hpa": o_pmin.tolist(),
            "obs_vmax_ms": o_vmax.tolist(),
            "track_err_km": err.tolist(),
            "at_km": at.tolist(),
            "ct_km": ct.tolist(),
            "wind_err_ms": wind.tolist(),
            "pmin_err_hpa": pmin.tolist(),
            "n_leads": int(leads.size),
            "n_matched": int(np.isfinite(err).sum()),
        }

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        self.validate(batch)
        curve = self._compute(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"curve": curve, "n_valid": curve["n_matched"]},
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        """把同一场次的状态并成一条曲线。

        协议保证一个场次落在一个工作块里，所以正常情况这里只收到一个状态；
        真并起来时按 lead 拼接——**重复 lead 直接报错**，那是同一条曲线被算了
        两遍，拼出来的数看着正常但已经错了。
        """
        if not states:
            raise MetricError("Cannot merge empty states list")
        first = states[0]
        for state in states[1:]:
            if state.metric_name != first.metric_name or state.metric_version != first.metric_version:
                raise MetricError(
                    f"Cannot merge states from different metrics/versions: "
                    f"{state.metric_name}@{state.metric_version} vs "
                    f"{first.metric_name}@{first.metric_version}"
                )
        if len(states) == 1:
            return first

        curves = [state.data["curve"] for state in states]
        leads: List[float] = []
        for curve in curves:
            leads.extend(curve["lead_h"])
        if len(set(leads)) != len(leads):
            raise MetricError(
                f"台风场次 {curves[0].get('storm')}@{curves[0].get('init_utc')} 的曲线"
                f"在合并时出现重复时效——同一条路径被算了不止一遍，结果不可用。"
            )
        order = sorted(range(len(leads)), key=lambda index: leads[index])
        merged = dict(curves[0])
        for field in CURVE_FIELDS:
            if field in ("lead_h", "valid_bjt"):
                continue
            if field not in merged:
                continue
            values: List[Any] = []
            for curve in curves:
                values.extend(curve[field])
            merged[field] = [values[index] for index in order]
        merged["lead_h"] = [leads[index] for index in order]
        merged["valid_bjt"] = [
            curve["valid_bjt"][k] for curve in curves for k in range(len(curve["lead_h"]))
        ]
        merged["valid_bjt"] = [merged["valid_bjt"][index] for index in order]
        merged["n_leads"] = len(merged["lead_h"])
        merged["n_matched"] = int(np.isfinite(np.asarray(merged["track_err_km"], dtype="f8")).sum())
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"curve": merged, "n_valid": merged["n_matched"]},
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        curve = state.data["curve"]
        n_leads = int(curve["n_leads"])
        n_matched = int(curve["n_matched"])
        values = {
            field: _nanmean(np.asarray(curve[field], dtype="f8")) for field in SUMMARY_FIELDS
        }
        status = ResultStatus.SUCCESS if n_matched else ResultStatus.NO_VALID_DATA
        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=values,
            status=status,
            n_requested=n_leads,
            n_valid=n_matched,
            unit="km",
            aggregation="mean_over_leads",
            product_kind=self.PRODUCT_KIND,
            curve=curve,
            warnings=[] if n_matched else ["这条路径没有任何时效配上实况"],
        )


def _parse_iso(value: str):
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _hours(value: float):
    from datetime import timedelta

    return timedelta(hours=float(value))
