"""评测结果可视化模块。

对外两件事：

* ``PrecipitationPlotter``：降水分类检验（TS 系列）出图，并可一键生成报告；
* ``build_ts_report`` / ``ts_report`` 的 CLI：把 TS 结果表写成 Markdown 报告。
"""

from xmetai_evaluation.visualization.precipitation_plots import (
    PrecipitationPlotter,
    prepare_ts_dataframe,
)
from xmetai_evaluation.visualization.ts_report import build_ts_report, diagnose, summarize

__all__ = [
    "PrecipitationPlotter",
    "prepare_ts_dataframe",
    "build_ts_report",
    "summarize",
    "diagnose",
]
