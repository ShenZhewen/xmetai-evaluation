"""评测结果可视化模块。

对外三件事：

* ``PrecipitationPlotter``：降水分类检验（TS 系列）出图，并可一键生成报告；
* ``FieldScorePlotter``：确定性连续场检验（RMSE / ACC / 活跃度 / 纬向谱）出图
  与报告，报告正文见 ``field_report``；
* ``detect_capability`` / ``build_report``：给一个产物目录，自动认出这是哪条流程
  再派发到对应模板（认不出或还没做模板会明确报错，不会静默回落）。

``build_ts_report`` / ``build_field_report`` 是两条链路各自的报告正文。
"""

from xmetai_evaluation.visualization.field_plots import (
    FieldScorePlotter,
    prepare_field_dataframe,
)
from xmetai_evaluation.visualization.field_report import (
    METRIC_SEMANTICS,
    build_field_report,
)
from xmetai_evaluation.visualization.precipitation_plots import (
    PrecipitationPlotter,
    prepare_ts_dataframe,
)
from xmetai_evaluation.visualization.report import build_report, detect_capability
from xmetai_evaluation.visualization.ts_report import build_ts_report, diagnose, summarize

__all__ = [
    "PrecipitationPlotter",
    "prepare_ts_dataframe",
    "build_ts_report",
    "FieldScorePlotter",
    "prepare_field_dataframe",
    "build_field_report",
    "METRIC_SEMANTICS",
    "build_report",
    "detect_capability",
    "summarize",
    "diagnose",
]
