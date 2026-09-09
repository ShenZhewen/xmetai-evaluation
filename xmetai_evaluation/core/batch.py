"""评估批次数据结构"""
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationBatch:
    """
    配对后的预报-观测批次，用于指标计算

    Attributes:
        forecast: 预报数据（xarray.DataArray）
        observation: 观测数据（xarray.DataArray）
        metadata: 可选的元数据（如样本权重、掩码等）
    """
    forecast: Any
    observation: Any
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
