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
    requires_complete_dimensions: Optional[List[str]] = None  # 必须完整保留的维度
    thresholds: Optional[List[float]] = None  # 阈值参数
    weights_required: bool = False  # 是否必须提供权重


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

    #: 结果产品类型，写进统一长表的 product_kind 列
    PRODUCT_KIND: str = "deterministic"

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

    def needs_members(self) -> bool:
        """该指标是否必须拿到原始集合成员（概率/集合类指标为 True）。"""
        return self.requirements().product_type is ProductType.ENSEMBLE_SAMPLES

    def needs_reference(self) -> bool:
        """该指标是否要用参考源（气候态、气候概率等）。

        Runner 据此决定要不要构建参考源：配置里配了参考、但这一段没指标用它，
        就不该去建、更不该在每个样本上白查一次。默认 False——不需要参考的指标
        占多数，跟 ``needs_members`` 一样由用到参考的子类覆写成 True。
        """
        return False

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
        forecast_data = (
            batch.forecast.payload if hasattr(batch.forecast, "payload") else batch.forecast
        )
        dims = getattr(forecast_data, "dims", ())
        if reqs.product_type is ProductType.DETERMINISTIC_FIELD and "member" in dims:
            member_count = int(getattr(forecast_data, "sizes", {}).get("member", 1))
            if member_count > 1:
                raise MetricError(
                    f"指标 '{self.name}' 要求确定性场，但输入含 member 维"
                    f"（{member_count} 个成员）；请在流程声明里加入降维变换"
                    f"（如 {{'type': 'ensemble_mean'}}）",
                    variable=self.name,
                )
        if (
            reqs.product_type is ProductType.ENSEMBLE_SAMPLES
            and getattr(batch, "members", None) is None
        ):
            raise MetricError(
                f"指标 '{self.name}' 需要集合成员（member 维），但这一批配对数据里没有成员场；"
                f"请确认预报数据源确实含成员，且流程没有把成员提前平均掉",
                variable=self.name,
            )

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


def single_variable(data: Any, side: str) -> xr.DataArray:
    """从 Dataset 里取出唯一的变量场。

    指标一次只评一个变量。多变量批次必须由 ``metric_options`` 的 ``variables``
    路由到具体变量后再交给指标（Runner 会调 ``matcher.narrow_batch``）。

    这里拿到多变量的 Dataset 说明配置漏了路由：静默取第一个变量会得出一个
    看起来正常、实际只代表某一个变量的分数，所以直接报错。
    """
    names = list(data.data_vars)
    if len(names) != 1:
        raise MetricError(
            f"{side} 传入了 {len(names)} 个变量 {names}，但该指标一次只评一个变量；"
            "请在配置的 metric_options 里声明变量路由，例如 "
            '{"rmse": {"variables": ["z500", "t2m"]}}'
        )
    return data[names[0]]
