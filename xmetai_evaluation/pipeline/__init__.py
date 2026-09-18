"""Pipeline 模块：声明、协议、配对与唯一执行内核。

Runner 走模块级 ``__getattr__`` 惰性导入：执行层（``execution/``）要用
matcher，而 runner 又要导入执行层，急切导入会形成环。
"""

from xmetai_evaluation.pipeline.matcher import Matcher
from xmetai_evaluation.pipeline.spec import (
    MetricSpec,
    PipelineSpec,
    SourceSpec,
    TransformSpec,
)

__all__ = ["Matcher", "Runner", "PipelineSpec", "SourceSpec", "TransformSpec", "MetricSpec"]


def __getattr__(name: str):
    if name == "Runner":
        from xmetai_evaluation.pipeline.runner import Runner

        return Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
