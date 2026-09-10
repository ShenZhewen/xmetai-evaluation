"""Pipeline 模块：声明、协议、配对与唯一执行内核。"""

from xmetai_evaluation.pipeline.matcher import Matcher
from xmetai_evaluation.pipeline.runner import Runner
from xmetai_evaluation.pipeline.spec import (
    MetricSpec,
    PipelineSpec,
    SourceSpec,
    TransformSpec,
)

__all__ = ["Matcher", "Runner", "PipelineSpec", "SourceSpec", "TransformSpec", "MetricSpec"]
