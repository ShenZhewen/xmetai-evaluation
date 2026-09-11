"""
Bias (平均误差) Metric

Bias = mean(forecast - observation)

按照 README 第 13.2 节实现 Metric 接口。
"""

import xarray as xr
from typing import List, Dict, Any

from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
    single_variable,
)
from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from xmetai_evaluation.core.errors import MetricError


class Bias(Metric):
    """
    平均误差（Bias）

    Bias = mean(forecast - observation)
    正值表示预报偏高，负值表示预报偏低。
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="bias", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        """Bias 需要确定性场"""
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],  # 可用于任意变量
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        """
        累积误差和

        Args:
            batch: 评测批次

        Returns:
            包含 sum_error 和 n_valid 的状态
        """
        # 提取数据
        if isinstance(batch.forecast, xr.DataArray):
            forecast = batch.forecast
        else:
            forecast = batch.forecast.payload if hasattr(batch.forecast, "payload") else batch.forecast

        if isinstance(batch.observation, xr.DataArray):
            observation = batch.observation
        else:
            observation = batch.observation.payload if hasattr(batch.observation, "payload") else batch.observation

        if isinstance(forecast, xr.Dataset):
            forecast = single_variable(forecast, "预报")
        if isinstance(observation, xr.Dataset):
            observation = single_variable(observation, "观测")

        # 计算误差
        error = forecast - observation

        # 应用掩码
        valid_mask = batch.valid_mask
        error_masked = error.where(valid_mask)

        # 计算误差和
        sum_error = float(error_masked.sum(skipna=True).values)

        # 计算有效点数
        n_valid = int(valid_mask.sum().values)

        # 如果提供了权重，使用加权
        if batch.weights is not None:
            weights_masked = batch.weights.where(valid_mask)
            weighted_error = error_masked * weights_masked
            sum_error = float(weighted_error.sum(skipna=True).values)
            weights_sum = float(weights_masked.sum(skipna=True).values)
        else:
            weights_sum = float(n_valid)

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "sum_error": sum_error,
                "weights_sum": weights_sum,
                "n_valid": n_valid,
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
        total_sum_error = sum(s.data["sum_error"] for s in states)
        total_weights_sum = sum(s.data["weights_sum"] for s in states)
        total_n_valid = sum(s.data["n_valid"] for s in states)
        total_n_accumulated = sum(s.n_accumulated for s in states)

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "sum_error": total_sum_error,
                "weights_sum": total_weights_sum,
                "n_valid": total_n_valid,
            },
            n_accumulated=total_n_accumulated,
        )

    def finalize(self, state: MetricState) -> MetricResult:
        """
        计算最终 Bias

        Args:
            state: 最终状态

        Returns:
            MetricResult
        """
        n_valid = state.data["n_valid"]
        weights_sum = state.data["weights_sum"]

        # 处理无有效数据
        if n_valid == 0 or weights_sum == 0:
            return MetricResult(
                metric_name=self.name,
                metric_version=self.version,
                value=float("nan"),
                status=ResultStatus.NO_VALID_DATA,
                n_requested=0,
                n_valid=0,
                warnings=["No valid data points"],
            )

        # 计算 Bias
        bias = state.data["sum_error"] / weights_sum

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=float(bias),
            status=ResultStatus.SUCCESS,
            n_requested=n_valid,
            n_valid=n_valid,
            weights_sum=weights_sum,
            aggregation="mean_over_samples",
            product_kind=self.PRODUCT_KIND,
        )
