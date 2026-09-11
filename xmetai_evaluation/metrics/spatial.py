# -*- coding: utf-8 -*-
"""空间检验指标：FSS（Fractions Skill Score，邻域分数技巧评分）。

口径与参考实现 ``ref/fdp/verify/verify/tp_deterministic_verifier.py::compute_fss`` 一致：

* 缺测按 0 处理，再按阈值二值化；
* 邻域覆盖率 PF/PO = n×n 窗口内二值场的均值（域外按 0 计）；
* ``FSS = 1 - Σ(PF-PO)² / Σ(PF²+PO²)``，分母为 0 时不可定义；
* 求和只在预报与实况都非缺测的格点上做。

邻域滤波默认用 ``scipy.ndimage.uniform_filter(mode='constant', cval=0)``
（与参考实现逐位一致）；没有 scipy 时回退到等价的积分图实现（仅支持奇数窗口）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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
    single_variable,
)

try:  # pragma: no cover - 取决于运行环境
    from scipy.ndimage import uniform_filter as _uniform_filter
except ImportError:  # pragma: no cover
    _uniform_filter = None

DEFAULT_WINDOWS: Tuple[int, ...] = (1, 3, 5, 15, 31, 63)


def neighborhood_fraction(binary: np.ndarray, size: int) -> np.ndarray:
    """n×n 窗口内的覆盖率（域外按 0 计）。

    与 ``scipy.ndimage.uniform_filter(binary, size=size, mode='constant', cval=0)``
    等价；无 scipy 时用积分图实现（只支持奇数窗口）。
    """
    if _uniform_filter is not None:
        return _uniform_filter(binary, size=size, mode="constant", cval=0.0)
    if size % 2 == 0:
        raise ValueError(f"没有 scipy 时只支持奇数窗口，收到 {size}")

    half = (size - 1) // 2
    padded = np.pad(binary, half, mode="constant", constant_values=0.0)
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype="f8")
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def _values(data: Any) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        return np.asarray(data.values, dtype="f8")
    if isinstance(data, xr.Dataset):
        return np.asarray(single_variable(data, "输入").values, dtype="f8")
    if hasattr(data, "payload"):
        payload = data.payload
        if isinstance(payload, xr.Dataset):
            payload = single_variable(payload, "输入")
        return np.asarray(payload.values, dtype="f8")
    return np.asarray(data, dtype="f8")


class FractionsSkillScore(Metric):
    """FSS：逐 (阈值 × 邻域窗口) 输出。"""

    PRODUCT_KIND = "spatial"

    def __init__(
        self,
        thresholds: Sequence[Tuple[str, float]],
        windows: Sequence[int] = DEFAULT_WINDOWS,
        params: Dict[str, Any] = None,
    ):
        super().__init__(name="fss", version="1.0.0", params=params or {})
        self.thresholds = [(str(name), float(value)) for name, value in thresholds]
        self.windows = tuple(int(window) for window in windows)

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
            thresholds=[value for _, value in self.thresholds],
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast = _values(batch.forecast)
        observation = _values(batch.observation)
        if forecast.shape != observation.shape:
            raise MetricError(
                f"FSS 需要同形状的预报与实况场：{forecast.shape} vs {observation.shape}",
                variable=self.name,
            )

        keep = ~(np.isnan(forecast) | np.isnan(observation))
        if batch.valid_mask is not None:
            mask = _values(batch.valid_mask)
            if mask.shape == keep.shape:
                keep = keep & mask.astype(bool)

        forecast_zero = np.nan_to_num(forecast, nan=0.0)
        observation_zero = np.nan_to_num(observation, nan=0.0)

        stats: Dict[Tuple[str, int], Dict[str, float]] = {}
        for name, threshold in self.thresholds:
            forecast_binary = (forecast_zero >= threshold).astype("f8")
            observation_binary = (observation_zero >= threshold).astype("f8")
            for window in self.windows:
                pf = neighborhood_fraction(forecast_binary, window)
                po = neighborhood_fraction(observation_binary, window)
                if pf.shape != keep.shape:
                    raise MetricError(
                        f"邻域滤波输出形状 {pf.shape} 与场形状 {keep.shape} 不一致",
                        variable=self.name,
                    )
                pf = pf[keep]
                po = po[keep]
                stats[(name, window)] = {
                    "numerator": float(np.sum((pf - po) ** 2)),
                    "denominator": float(np.sum(pf**2 + po**2)),
                    "n": int(pf.size),
                }

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"stats": stats},
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        first = states[0]
        for state in states[1:]:
            if state.metric_name != first.metric_name:
                raise MetricError(
                    f"Cannot merge states from different metrics: "
                    f"{first.metric_name} vs {state.metric_name}",
                    variable=self.name,
                )
        merged: Dict[Tuple[str, int], Dict[str, float]] = {}
        for state in states:
            for key, values in state.data["stats"].items():
                target = merged.setdefault(
                    key, {"numerator": 0.0, "denominator": 0.0, "n": 0}
                )
                target["numerator"] += values["numerator"]
                target["denominator"] += values["denominator"]
                target["n"] += values["n"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"stats": merged},
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        stats = state.data["stats"]
        results: Dict[str, Any] = {}
        total = 0
        for name, threshold in self.thresholds:
            for window in self.windows:
                values = stats.get((name, window))
                if values is None:
                    continue
                denominator = values["denominator"]
                fss = (
                    float(1.0 - values["numerator"] / denominator)
                    if denominator > 0
                    else float("nan")
                )
                results[f"{name}|w{window}"] = {
                    "threshold": threshold,
                    "window": window,
                    "FSS": fss,
                    "n_points": values["n"],
                }
                total += values["n"]

        if not results:
            status = ResultStatus.NO_VALID_DATA
        elif all(np.isnan(entry["FSS"]) for entry in results.values()):
            status = ResultStatus.UNDEFINED
        else:
            status = ResultStatus.SUCCESS

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=results,
            status=status,
            n_requested=total,
            n_valid=total,
            aggregation="neighborhood_fraction",
            unit="1",
            product_kind=self.PRODUCT_KIND,
        )
