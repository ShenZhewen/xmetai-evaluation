"""
RMSE 参考实现

用于验证 Metric 基类和可合并统计状态，按照 README Change 3 要求。
"""

import numpy as np
import xarray as xr
from typing import List, Dict, Any

from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
    field_unit,
    single_variable,
)
from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from xmetai_evaluation.core.errors import MetricError


class RMSE(Metric):
    """
    均方根误差（Root Mean Square Error）

    这是一个最小参考实现，用于测试整批与分块合并一致性。
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="rmse", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        """RMSE 需要确定性场"""
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],  # 可用于任意变量
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        """
        累积误差平方和

        Args:
            batch: 评测批次

        Returns:
            包含 sum_squared_error 和 n_valid 的状态
        """
        # 提取数据
        if isinstance(batch.forecast, xr.DataArray):
            forecast = batch.forecast
        else:
            # 如果是 DataBundle，提取 payload
            forecast = batch.forecast.payload if hasattr(batch.forecast, "payload") else batch.forecast

        if isinstance(batch.observation, xr.DataArray):
            observation = batch.observation
        else:
            observation = batch.observation.payload if hasattr(batch.observation, "payload") else batch.observation

        if isinstance(forecast, xr.Dataset):
            forecast = single_variable(forecast, "预报")
        if isinstance(observation, xr.Dataset):
            observation = single_variable(observation, "观测")

        # 单位随变量走（见 field_unit 的说明），实况侧优先
        unit = field_unit(observation, forecast)

        # 计算误差
        error = forecast - observation

        # 应用掩码
        valid_mask = batch.valid_mask
        error_masked = error.where(valid_mask)

        # 计算平方误差和
        squared_error = error_masked**2
        sum_squared_error = float(squared_error.sum(skipna=True).values)

        # 计算有效点数
        n_valid = int(valid_mask.sum().values)

        # 如果提供了权重，使用加权
        if batch.weights is not None:
            weights_masked = batch.weights.where(valid_mask)
            weighted_squared_error = squared_error * weights_masked
            sum_squared_error = float(weighted_squared_error.sum(skipna=True).values)
            weights_sum = float(weights_masked.sum(skipna=True).values)
        else:
            weights_sum = float(n_valid)

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "sum_squared_error": sum_squared_error,
                "weights_sum": weights_sum,
                "n_valid": n_valid,
                "unit": unit,
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
        total_sum_squared_error = sum(s.data["sum_squared_error"] for s in states)
        total_weights_sum = sum(s.data["weights_sum"] for s in states)
        total_n_valid = sum(s.data["n_valid"] for s in states)
        total_n_accumulated = sum(s.n_accumulated for s in states)
        # 同一次运行里各样本的单位必然一致，取第一个非空的即可
        unit = next((s.data.get("unit") for s in states if s.data.get("unit")), "")

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "sum_squared_error": total_sum_squared_error,
                "weights_sum": total_weights_sum,
                "n_valid": total_n_valid,
                "unit": unit,
            },
            n_accumulated=total_n_accumulated,
        )

    def finalize(self, state: MetricState) -> MetricResult:
        """
        计算最终 RMSE

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
                n_requested=0,  # 无法确定请求数
                n_valid=0,
                warnings=["No valid data points"],
            )

        # 计算 RMSE
        mse = state.data["sum_squared_error"] / weights_sum
        rmse = np.sqrt(mse)

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=float(rmse),
            status=ResultStatus.SUCCESS,
            n_requested=n_valid,  # 简化：假设所有有效点都被请求
            n_valid=n_valid,
            weights_sum=weights_sum,
            unit=str(state.data.get("unit") or ""),
            aggregation="mean_over_samples",
            product_kind=self.PRODUCT_KIND,
        )
