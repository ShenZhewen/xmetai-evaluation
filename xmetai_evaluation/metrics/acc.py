"""
ACC (距平相关系数) Metric

ACC = correlation(forecast_anomaly, observation_anomaly)
其中 anomaly = field - climatology

按照 README 第 13.2 节实现 Metric 接口。
"""

import numpy as np
import xarray as xr
from typing import List, Dict, Any, Optional

from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
)
from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from xmetai_evaluation.core.errors import MetricError


class ACC(Metric):
    """
    距平相关系数（Anomaly Correlation Coefficient）

    ACC = correlation(forecast_anomaly, observation_anomaly)

    需要提供气候态数据（climatology）。
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="acc", version="1.0.0", params=params or {})
        self.climatology_path = params.get("climatology_path") if params else None

    def requirements(self) -> MetricRequirements:
        """ACC 需要确定性场和气候态参考"""
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
            reference_data=["climatology"],  # 需要气候态
            can_merge_along=["sample", "time", "init_time"],
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        """
        累积相关统计量

        Args:
            batch: 评测批次（必须包含 reference 气候态）

        Returns:
            包含相关统计量的状态
        """
        # 提取数据
        if isinstance(batch.forecast, xr.DataArray):
            forecast = batch.forecast
        else:
            forecast = batch.forecast.payload

        if isinstance(batch.observation, xr.DataArray):
            observation = batch.observation
        else:
            observation = batch.observation.payload

        # 获取气候态
        if batch.reference is None:
            # 如果没有气候态，使用零场（临时方案）
            climatology = xr.zeros_like(observation)
            has_climatology = False
        else:
            if isinstance(batch.reference, xr.DataArray):
                climatology = batch.reference
            else:
                climatology = batch.reference.payload
            has_climatology = True

        # 计算距平
        forecast_anomaly = forecast - climatology
        observation_anomaly = observation - climatology

        # 应用掩码
        valid_mask = batch.valid_mask
        forecast_anomaly_masked = forecast_anomaly.where(valid_mask)
        observation_anomaly_masked = observation_anomaly.where(valid_mask)

        # 应用权重
        if batch.weights is not None:
            weights_masked = batch.weights.where(valid_mask)
        else:
            weights_masked = xr.ones_like(valid_mask, dtype=float).where(valid_mask)

        # 计算加权统计量（用于相关系数）
        # sum(w * f_anom * o_anom)
        sum_weighted_product = float(
            (weights_masked * forecast_anomaly_masked * observation_anomaly_masked)
            .sum(skipna=True)
            .values
        )

        # sum(w * f_anom^2)
        sum_weighted_forecast_sq = float(
            (weights_masked * forecast_anomaly_masked**2).sum(skipna=True).values
        )

        # sum(w * o_anom^2)
        sum_weighted_obs_sq = float(
            (weights_masked * observation_anomaly_masked**2).sum(skipna=True).values
        )

        # sum(w)
        weights_sum = float(weights_masked.sum(skipna=True).values)

        # 有效点数
        n_valid = int(valid_mask.sum().values)

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "sum_weighted_product": sum_weighted_product,
                "sum_weighted_forecast_sq": sum_weighted_forecast_sq,
                "sum_weighted_obs_sq": sum_weighted_obs_sq,
                "weights_sum": weights_sum,
                "n_valid": n_valid,
                "has_climatology": has_climatology,
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        """
        合并多个状态

        Args:
            states: 状态列表

        Returns:
            合并后的状态
        """
        if not states:
            raise MetricError("Cannot merge empty states list")

        # 验证所有状态来自同一指标和版本
        first_state = states[0]
        for state in states[1:]:
            if state.metric_name != first_state.metric_name:
                raise MetricError(
                    f"Cannot merge states from different metrics: "
                    f"{first_state.metric_name} vs {state.metric_name}"
                )
            if state.metric_version != first_state.metric_version:
                raise MetricError(
                    f"Cannot merge states from different versions: "
                    f"{first_state.metric_version} vs {state.metric_version}"
                )

        # 合并统计量
        total_sum_weighted_product = sum(s.data["sum_weighted_product"] for s in states)
        total_sum_weighted_forecast_sq = sum(s.data["sum_weighted_forecast_sq"] for s in states)
        total_sum_weighted_obs_sq = sum(s.data["sum_weighted_obs_sq"] for s in states)
        total_weights_sum = sum(s.data["weights_sum"] for s in states)
        total_n_valid = sum(s.data["n_valid"] for s in states)
        total_n_accumulated = sum(s.n_accumulated for s in states)
        has_climatology = any(s.data["has_climatology"] for s in states)

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "sum_weighted_product": total_sum_weighted_product,
                "sum_weighted_forecast_sq": total_sum_weighted_forecast_sq,
                "sum_weighted_obs_sq": total_sum_weighted_obs_sq,
                "weights_sum": total_weights_sum,
                "n_valid": total_n_valid,
                "has_climatology": has_climatology,
            },
            n_accumulated=total_n_accumulated,
        )

    def finalize(self, state: MetricState) -> MetricResult:
        """
        计算最终 ACC

        Args:
            state: 最终状态

        Returns:
            MetricResult
        """
        n_valid = state.data["n_valid"]
        weights_sum = state.data["weights_sum"]

        warnings = []
        if not state.data.get("has_climatology", False):
            warnings.append("No climatology provided, using zero anomaly")

        # 处理无有效数据
        if n_valid == 0 or weights_sum == 0:
            return MetricResult(
                metric_name=self.name,
                metric_version=self.version,
                value=float("nan"),
                status=ResultStatus.NO_VALID_DATA,
                n_requested=0,
                n_valid=0,
                warnings=warnings + ["No valid data points"],
            )

        # 计算 ACC = sum(w * f * o) / sqrt(sum(w * f^2) * sum(w * o^2))
        numerator = state.data["sum_weighted_product"]
        denominator = np.sqrt(
            state.data["sum_weighted_forecast_sq"] * state.data["sum_weighted_obs_sq"]
        )

        # 处理零方差
        if denominator == 0 or np.isnan(denominator):
            return MetricResult(
                metric_name=self.name,
                metric_version=self.version,
                value=float("nan"),
                status=ResultStatus.UNDEFINED,
                n_requested=n_valid,
                n_valid=n_valid,
                warnings=warnings + ["Zero variance in forecast or observation anomaly"],
            )

        acc = numerator / denominator

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=float(acc),
            status=ResultStatus.SUCCESS if not warnings else ResultStatus.PARTIAL,
            n_requested=n_valid,
            n_valid=n_valid,
            weights_sum=weights_sum,
            aggregation="spatial_correlation",
            warnings=warnings,
        )
