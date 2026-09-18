# -*- coding: utf-8 -*-
"""执行层：能力画像 -> 执行策略 -> 工作块计划 -> 块执行与归并。

评测 = 三个数据集做差（预报 / 观测 / 参考）。预报是驱动集：工作块永远
按起报时刻从预报侧切；观测与参考要读多少，由块的起报推出的有效时刻
跨度决定，驻留方式（slice / resident / window:W）由策略层按数据源
形态推导、配置可覆盖。

    profiles.py   指标族 -> 联合资源画像（要不要参考/成员/整场，轻重）
    strategy.py   画像 + 数据源形态 -> 执行策略（并发形态、块大小、驻留）
    plan.py       一段声明 -> 工作块序列 + 加载声明（父进程，只做便宜探测）
    loader.py     resident / window / slice 三种驻留策略的统一入口
    executor.py   块收集 + serial/threads/processes 调度 + 统一归并
"""

from xmetai_evaluation.execution.executor import (
    ChunkOutcome,
    MergedOutcome,
    MetricRun,
    collect_chunk_states,
    execute_chunks,
    merge_outcomes,
    metric_runs,
    needs_reference,
)
from xmetai_evaluation.execution.loader import RoleSpec, RunLoader
from xmetai_evaluation.execution.plan import WorkChunk, WorkPlan, build_plan
from xmetai_evaluation.execution.profiles import ResourceProfile, union_profile
from xmetai_evaluation.execution.strategy import ExecutionStrategy, derive_strategy

__all__ = [
    "ChunkOutcome",
    "MergedOutcome",
    "MetricRun",
    "ResourceProfile",
    "RoleSpec",
    "RunLoader",
    "WorkChunk",
    "WorkPlan",
    "build_plan",
    "collect_chunk_states",
    "derive_strategy",
    "execute_chunks",
    "ExecutionStrategy",
    "merge_outcomes",
    "metric_runs",
    "needs_reference",
    "union_profile",
]
