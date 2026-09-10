# -*- coding: utf-8 -*-
"""结果层：统一长表与落盘。

对外只有两个概念：
    build_tables(results, ...) -> ResultTables   把 MetricResult 展平成统一长表
    ResultStore(...).write(results, ...)         把长表与运行记录落盘
"""

from xmetai_evaluation.results.store import ResultStore, RunContext
from xmetai_evaluation.results.table import (
    COVERAGE_COLUMNS,
    DETAIL_COLUMNS,
    SCORE_COLUMNS,
    ResultTables,
    build_tables,
    categorical_wide,
)

__all__ = [
    "COVERAGE_COLUMNS",
    "DETAIL_COLUMNS",
    "SCORE_COLUMNS",
    "ResultTables",
    "build_tables",
    "categorical_wide",
    "ResultStore",
    "RunContext",
]
