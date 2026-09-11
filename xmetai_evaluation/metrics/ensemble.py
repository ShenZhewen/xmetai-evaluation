# -*- coding: utf-8 -*-
"""集合检验指标：CRPS 与 Spread-Error Ratio。

口径与参考实现 ``ref/fdp/verify/verify/ensemble_verifier.py`` 完全一致：

* 输入是**同一网格上的成员场与实况场**（预报已与实况同网格、同单位）；
* 权重取 ``cos(lat)``（由协议放进 ``batch.weights``）；分子用 nansum（缺测点不参与），
  分母用全部点权重之和——与参考实现的 ``wsum = np.nansum(w2d)`` 一致；
* ``CRPS = mean_m |f_m - o| - 1/(2M²)·ΣΣ|f_m - f_n|``，第二项用排序线性系数；
* ``Spread = sqrt(Σw·Σ_m(f_m - f̄)² / (Σw·(M-1)))``，``RMSE = sqrt(Σw·(f̄-o)²/Σw)``，
  ``Ratio = Spread / RMSE``。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
)


def _values(data: Any) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        return np.asarray(data.values, dtype="f8")
    if isinstance(data, xr.Dataset):
        # 配对后的 forecast/observation 可能是 Dataset（取第一个变量，与其它指标一致）
        return np.asarray(data[list(data.data_vars)[0]].values, dtype="f8")
    if hasattr(data, "payload"):
        payload = data.payload
        if isinstance(payload, xr.Dataset):
            payload = payload[list(payload.data_vars)[0]]
        return np.asarray(payload.values, dtype="f8")
    return np.asarray(data, dtype="f8")


def _weighted_sum(field: np.ndarray, weights: np.ndarray) -> float:
    """Σ w·field（field 的缺测点不参与），与参考实现的 np.nansum 语义一致。"""
    return float(np.nansum(weights * field))


class CRPS(Metric):
    """连续分级概率评分（纬度加权，闭式解）。"""

    PRODUCT_KIND = "ensemble"

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="crps", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.ENSEMBLE_SAMPLES,
            variables=["*"],
            weights_required=False,
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        members = _values(batch.members)
        observation = _values(batch.observation)
        weights = _values(batch.weights) if batch.weights is not None else None
        if members.ndim < 2:
            raise MetricError(
                f"CRPS 需要 (member, ...) 形态的成员场，实际 ndim={members.ndim}",
                variable=self.name,
            )

        obs = observation[np.newaxis, ...]
        ok = np.isfinite(members) & np.isfinite(obs)
        m = ok.sum(axis=0).astype("f8")

        # 逐点平均 |f - o|（仅有效成员，缺测成员不参与）
        d = np.where(ok, np.abs(members - obs), 0.0)
        term1 = d.sum(axis=0) / np.maximum(m, 1.0)

        # 排序线性系数项（NaN 排末位，逐点有效成员数 m，分母 m²）
        n = members.shape[0]
        ordered = np.sort(np.where(ok, members, np.nan), axis=0)
        kk = np.arange(n, dtype="f8") + 1.0
        coeff_shape = (n,) + (1,) * (ordered.ndim - 1)
        coeff = 2.0 * kk.reshape(coeff_shape) - m[np.newaxis] - 1.0
        term2 = np.nansum(ordered * coeff, axis=0) / np.maximum(m * m, 1.0)

        # 闭式 CRPS 本就非负，不 clip（与参考实现/业界一致）；分母只统计有效点权重
        crps_field = term1 - term2
        weight_field = weights if weights is not None else np.ones_like(observation)
        okg = (m > 0) & np.isfinite(crps_field)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "weighted_crps": float(np.nansum(np.where(okg, crps_field, 0.0) * weight_field)),
                "weights_sum": float(np.nansum(np.where(okg, weight_field, 0.0))),
                "n_valid": int(okg.sum()),
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        _check_same_metric(self, states)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "weighted_crps": sum(state.data["weighted_crps"] for state in states),
                "weights_sum": sum(state.data["weights_sum"] for state in states),
                "n_valid": sum(state.data["n_valid"] for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        weights_sum = state.data["weights_sum"]
        n_valid = int(state.data["n_valid"])
        if weights_sum <= 0 or n_valid == 0:
            return MetricResult(
                metric_name=self.name,
                metric_version=self.version,
                value=float("nan"),
                status=ResultStatus.NO_VALID_DATA,
                n_requested=n_valid,
                n_valid=n_valid,
                warnings=["没有有效配对样本"],
            )
        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=float(state.data["weighted_crps"] / weights_sum),
            status=ResultStatus.SUCCESS,
            n_requested=n_valid,
            n_valid=n_valid,
            weights_sum=float(weights_sum),
            aggregation="area_weighted",
            product_kind=self.PRODUCT_KIND,
        )


class SpreadError(Metric):
    """集合离散度-误差比（Spread / RMSE），同时给出 Spread 与集合平均 RMSE。"""

    PRODUCT_KIND = "ensemble"

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="spread_error", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.ENSEMBLE_SAMPLES,
            variables=["*"],
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        members = _values(batch.members)
        observation = _values(batch.observation)
        weights = _values(batch.weights) if batch.weights is not None else None
        if members.ndim < 2:
            raise MetricError(
                f"Spread-Error 需要 (member, ...) 形态的成员场，实际 ndim={members.ndim}",
                variable=self.name,
            )

        member_count = int(members.shape[0])
        mean = np.mean(members, axis=0)
        deviation = np.sum((members - mean[np.newaxis, ...]) ** 2, axis=0)
        error = (mean - observation) ** 2
        weight_field = weights if weights is not None else np.ones_like(observation)

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "member_count": member_count,
                # 与参考实现一致：分子 nansum（缺测点跳过），分母取全部点权重之和
                "weighted_spread": _weighted_sum(deviation, weight_field),
                "weighted_error": _weighted_sum(error, weight_field),
                "weights_sum": float(np.nansum(weight_field)),
                "n_valid": int((np.isfinite(deviation) & np.isfinite(error)).sum()),
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        _check_same_metric(self, states)
        member_count = states[0].data["member_count"]
        for state in states[1:]:
            if state.data["member_count"] != member_count:
                raise MetricError(
                    f"集合成员数不一致（{member_count} vs {state.data['member_count']}）",
                    variable=self.name,
                )
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "member_count": member_count,
                "weighted_spread": sum(state.data["weighted_spread"] for state in states),
                "weighted_error": sum(state.data["weighted_error"] for state in states),
                "weights_sum": sum(state.data["weights_sum"] for state in states),
                "n_valid": sum(state.data["n_valid"] for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        member_count = int(state.data["member_count"])
        weights_sum = state.data["weights_sum"]
        n_valid = int(state.data["n_valid"])
        if weights_sum <= 0 or n_valid == 0:
            return MetricResult(
                metric_name=self.name,
                metric_version=self.version,
                value={"SPREAD": float("nan"), "RMSE": float("nan"), "RATIO": float("nan")},
                status=ResultStatus.NO_VALID_DATA,
                n_requested=n_valid,
                n_valid=n_valid,
                warnings=["没有有效配对样本"],
            )

        spread = float("nan")
        if member_count > 1:
            spread = float(
                np.sqrt(state.data["weighted_spread"] / (weights_sum * (member_count - 1)))
            )
        rmse = float(np.sqrt(state.data["weighted_error"] / weights_sum))
        ratio = float(spread / rmse) if (np.isfinite(spread) and rmse > 0) else float("nan")

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value={"SPREAD": spread, "RMSE": rmse, "RATIO": ratio},
            status=ResultStatus.SUCCESS,
            n_requested=n_valid,
            n_valid=n_valid,
            weights_sum=float(weights_sum),
            aggregation="area_weighted",
            unit="",
            product_kind=self.PRODUCT_KIND,
        )


def _check_same_metric(metric: Metric, states: List[MetricState]) -> None:
    first = states[0]
    for state in states[1:]:
        if state.metric_name != first.metric_name:
            raise MetricError(
                f"Cannot merge states from different metrics: "
                f"{first.metric_name} vs {state.metric_name}",
                variable=metric.name,
            )
