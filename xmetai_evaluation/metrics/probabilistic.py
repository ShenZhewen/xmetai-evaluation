# -*- coding: utf-8 -*-
"""概率分类指标：AROC / BS / BSS（超越式事件口径 ``x >= 阈值``）。

与确定性 TS 的区别：

* TS 吃**确定性场**（集合需先降维），只统计 2x2 列联表；
* 这里的指标吃**原始集合成员**：以"超过阈值的成员比例"作为预报概率
  ``p = k / M``，进而计算概率评分的可靠性与分辨力。

实现口径与参考实现 ``ref/.../vfc/metric_categorical.py::ProbEventHistogram`` 一致：

* 每个阈值维护 ``(M+1, 2)`` 直方图：行 = 超阈值成员数，列 = 观测事件 0/1；
* 观测或**任一成员**非有限时该站点不参与统计；站点等权；
* ``BS = sum(hist[0]*p^2 + hist[1]*(1-p)^2) / N``；
* ``BS_ref = r*(1-r)``（r = 观测事件频率），``BSS = 1 - BS/BS_ref``；
* ``AROC`` 用 ROC 曲线梯形面积，事件与非事件都出现才可算。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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

_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _values(data: Any) -> np.ndarray:
    """从 DataArray / DataBundle / 数组取出 numpy 值。"""
    if isinstance(data, xr.DataArray):
        return np.asarray(data.values, dtype="f8")
    if isinstance(data, xr.Dataset):
        return np.asarray(data[list(data.data_vars)[0]].values, dtype="f8")
    if hasattr(data, "payload"):
        payload = data.payload
        if isinstance(payload, xr.Dataset):
            payload = payload[list(payload.data_vars)[0]]
        return np.asarray(payload.values, dtype="f8")
    return np.asarray(data, dtype="f8")


class EnsembleProbabilityScore(Metric):
    """集合概率评分（AROC / BS / BSS），逐阈值输出。"""

    PRODUCT_KIND = "probabilistic"

    def __init__(self, thresholds: List[Tuple[str, float]], params: Dict[str, Any] = None):
        """
        Args:
            thresholds: 阈值列表 ``[(等级名, 阈值mm), ...]``（超越式口径）。
            params: 可选参数。
        """
        super().__init__(name="prob", version="1.0.0", params=params or {})
        self.thresholds = list(thresholds)

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.ENSEMBLE_SAMPLES,
            variables=["*"],
            thresholds=[value for _, value in self.thresholds],
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        members = _values(batch.members)
        if members.ndim != 2:
            raise MetricError(
                f"概率指标需要 (member, sample) 形态的成员场，实际 ndim={members.ndim}",
                variable=self.name,
            )
        observation = _values(batch.observation)
        if observation.ndim != 1:
            observation = observation.reshape(-1)

        mask = None
        if batch.valid_mask is not None:
            mask = _values(batch.valid_mask)
            if mask.ndim == 2:
                mask = mask.all(axis=0)
            mask = mask.astype(bool)

        ok = np.isfinite(observation) & np.all(np.isfinite(members), axis=0)
        if mask is not None and mask.shape == ok.shape:
            ok = ok & mask

        weights = None
        if batch.weights is not None:
            weights = _values(batch.weights)
            if weights.ndim == 2 and weights.shape == members.shape:
                weights = weights[0]  # 站点权重与成员无关
            if weights.shape != observation.shape:
                raise MetricError(
                    f"权重形状 {weights.shape} 与样本形状 {observation.shape} 不一致",
                    variable=self.name,
                )
        weights_ok = weights[ok] if weights is not None else None

        member_count = int(members.shape[0])
        members_ok = members[:, ok]
        observation_ok = observation[ok]

        histograms: Dict[str, np.ndarray] = {}
        for name, threshold in self.thresholds:
            exceeded = (members_ok >= threshold).sum(axis=0).astype("i8")
            event = (observation_ok >= threshold).astype("i8")
            index = exceeded * 2 + event
            if weights_ok is None:
                histograms[name] = np.bincount(
                    index, minlength=2 * (member_count + 1)
                ).reshape(member_count + 1, 2).astype("f8")
            else:
                # 有站点权重时直方图存"权重和"，与参考实现的加权 BSS/AROC 一致
                histograms[name] = np.bincount(
                    index, weights=weights_ok, minlength=2 * (member_count + 1)
                ).reshape(member_count + 1, 2)

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "member_count": member_count,
                "histograms": histograms,
                "n_points": int(ok.sum()),
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        first = states[0]
        member_count = first.data["member_count"]
        merged = {
            name: np.zeros_like(histogram)
            for name, histogram in first.data["histograms"].items()
        }
        for state in states:
            if state.metric_name != first.metric_name:
                raise MetricError(
                    f"Cannot merge states from different metrics: "
                    f"{first.metric_name} vs {state.metric_name}"
                )
            if state.data["member_count"] != member_count:
                raise MetricError(
                    f"集合成员数不一致（{member_count} vs {state.data['member_count']}），"
                    f"无法合并概率评分；请检查各起报的成员数",
                    variable=self.name,
                )
            for name, histogram in state.data["histograms"].items():
                merged[name] = merged.get(name, np.zeros_like(histogram)) + histogram
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "member_count": member_count,
                "histograms": merged,
                "n_points": sum(state.data.get("n_points", 0) for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        member_count = int(state.data["member_count"])
        n_points = int(state.data.get("n_points", 0))
        results: Dict[str, Any] = {}
        for name, threshold in self.thresholds:
            results[name] = self._score(
                state.data["histograms"][name], member_count, name, threshold, n_points
            )

        total = n_points
        if total == 0:
            status = ResultStatus.NO_VALID_DATA
        elif any(np.isnan(entry["AROC"]) for entry in results.values()):
            status = ResultStatus.PARTIAL
        else:
            status = ResultStatus.SUCCESS

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=results,
            status=status,
            n_requested=total,
            n_valid=total,
            aggregation="probability_histogram",
            unit="1",
            product_kind=self.PRODUCT_KIND,
        )

    @staticmethod
    def _score(
        histogram: np.ndarray,
        member_count: int,
        name: str,
        threshold: float,
        n_points: int = 0,
    ) -> Dict[str, Any]:
        total = float(histogram.sum())
        if total <= 0:
            return {
                "threshold": threshold,
                "threshold_name": name,
                "AROC": np.nan,
                "BS": np.nan,
                "BS_ref": np.nan,
                "BSS": np.nan,
                "base_rate": np.nan,
                "n_points": n_points,
            }

        event = float(histogram[:, 1].sum())
        non_event = float(histogram[:, 0].sum())
        probability = np.arange(member_count + 1, dtype="f8") / float(member_count)

        brier = float(
            (
                histogram[:, 0] * probability**2
                + histogram[:, 1] * (1 - probability) ** 2
            ).sum()
            / total
        )
        base_rate = event / total
        brier_reference = base_rate * (1 - base_rate)

        if event > 0 and non_event > 0:
            hit = np.array(
                [histogram[index:, 1].sum() / event for index in range(member_count + 1)]
            )
            false_alarm = np.array(
                [
                    histogram[index:, 0].sum() / non_event
                    for index in range(member_count + 1)
                ]
            )
            x = np.concatenate(([0.0], false_alarm[::-1]))
            y = np.concatenate(([0.0], hit[::-1]))
            aroc = float(_TRAPZ(y, x=x))
        else:
            aroc = float("nan")

        return {
            "threshold": threshold,
            "threshold_name": name,
            "AROC": aroc,
            "BS": brier,
            "BS_ref": brier_reference,
            "BSS": float(1 - brier / brier_reference) if brier_reference > 0 else float("nan"),
            "base_rate": base_rate,
            "n_points": n_points,
        }
