"""指标计算模块。

指标只负责"算什么差异"：输入 EvaluationBatch，输出可合并状态与 MetricResult。
"""

from xmetai_evaluation.metrics.acc import ACC
from xmetai_evaluation.metrics.base import Metric, MetricRequirements, MetricState
from xmetai_evaluation.metrics.bias import Bias
from xmetai_evaluation.metrics.categorical import ContingencyTable, TSScore
from xmetai_evaluation.metrics.ensemble import CRPS, SpreadError
from xmetai_evaluation.metrics.probabilistic import EnsembleProbabilityScore
from xmetai_evaluation.metrics.rmse import RMSE

__all__ = [
    "Metric",
    "MetricRequirements",
    "MetricState",
    "ContingencyTable",
    "RMSE",
    "Bias",
    "ACC",
    "TSScore",
    "EnsembleProbabilityScore",
    "CRPS",
    "SpreadError",
]
