"""
Metric 基类与协议

按照 README 第 13.2 节实现 Metric 的五个概念边界：
requirements / validate / accumulate / merge / finalize
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

import xarray as xr

from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from xmetai_evaluation.core.variables import ForecastKind
from xmetai_evaluation.core.errors import MetricError


class ProductType(Enum):
    """产品类型要求"""

    DETERMINISTIC_FIELD = "deterministic_field"  # 确定性场
    ENSEMBLE_SAMPLES = "ensemble_samples"  # 原始集合成员
    EVENT_PROBABILITY = "event_probability"  # 事件概率产品
    INDEX_SERIES = "index_series"  # 指数序列（如MJO RMM）


@dataclass
class MetricRequirements:
    """
    指标输入要求

    按照 README 第 13.2 节：不再用单一 requires_all_members 代替全部输入契约。
    """

    product_type: ProductType  # 产品类型
    variables: List[str]  # 所需变量
    reference_data: Optional[List[str]] = None  # 气候态/概率/投影基底等
    requires_complete_dimensions: Optional[List[str]] = None  # 必须完整保留的维度
    can_merge_along: Optional[List[str]] = None  # 允许分块合并的维度
    min_samples: int = 1  # 最低样本数
    min_members: Optional[int] = None  # 最低成员数（适用于集合）
    thresholds: Optional[List[float]] = None  # 阈值参数
    weights_required: bool = False  # 是否必须提供权重
    output_dimensions: Optional[List[str]] = None  # 输出维度


@dataclass
class MetricState:
    """
    可合并的统计状态

    accumulate 和 merge 操作的中间结果
    """

    metric_name: str
    metric_version: str
    data: Dict[str, Any]  # 统计量（如 sum_squared_error, n_valid）
    coordinates: Dict[str, Any] = field(default_factory=dict)  # 结果坐标
    n_accumulated: int = 0  # 已累积的批次数


class Metric(ABC):
    """
    指标基类

    按照 README 第 13.2 节规定的五个概念边界实现。
    """

    def __init__(self, name: str, version: str, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            name: 指标名称
            version: 版本号
            params: 可选参数（如阈值、窗口等）
        """
        self.name = name
        self.version = version
        self.params = params or {}

    @abstractmethod
    def requirements(self) -> MetricRequirements:
        """
        返回所需变量、产品类型、维度、参考数据和参数约束

        Returns:
            MetricRequirements
        """
        pass

    def validate(self, batch: EvaluationBatch) -> None:
        """
        检查单位、维度、成员、坐标、时间语义和阈值

        按照 README 第 13.2 节：校验不能依赖数组位置。

        Args:
            batch: 评测批次

        Raises:
            MetricError: 如果输入不符合要求
        """
        reqs = self.requirements()

        # 检查产品类型（这里简化处理，实际需要检查 batch.forecast 的元数据）
        # 子类可以覆盖此方法进行更详细的校验

        # 检查维度
        if isinstance(batch.forecast, xr.DataArray):
            forecast_dims = set(batch.forecast.dims)
            if reqs.requires_complete_dimensions:
                for dim in reqs.requires_complete_dimensions:
                    if dim not in forecast_dims:
                        raise MetricError(
                            f"Required dimension '{dim}' not found in forecast. "
                            f"Available: {forecast_dims}",
                            variable=self.name,
                        )

        # 检查权重
        if reqs.weights_required and batch.weights is None:
            raise MetricError(
                f"Metric '{self.name}' requires weights but none provided",
                variable=self.name,
            )

    @abstractmethod
    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        """
        只计算本批状态，不写磁盘，不修改输入对象

        Args:
            batch: 评测批次

        Returns:
            MetricState
        """
        pass

    @abstractmethod
    def merge(self, states: List[MetricState]) -> MetricState:
        """
        合并同一指标、同一协议、同一结果键的状态

        按照 README 第 13.2 节：参数和版本不一致时失败。

        Args:
            states: 状态列表

        Returns:
            合并后的状态

        Raises:
            MetricError: 如果状态不兼容
        """
        pass

    @abstractmethod
    def finalize(self, state: MetricState) -> MetricResult:
        """
        生成 MetricResult，负责零分母、无有效数据和部分完成等状态

        Args:
            state: 最终状态

        Returns:
            MetricResult
        """
        pass

    def compute(self, batch: EvaluationBatch) -> MetricResult:
        """
        便捷方法：完整计算流程（适用于小型指标）

        Args:
            batch: 评测批次

        Returns:
            MetricResult
        """
        self.validate(batch)
        state = self.accumulate(batch)
        return self.finalize(state)
