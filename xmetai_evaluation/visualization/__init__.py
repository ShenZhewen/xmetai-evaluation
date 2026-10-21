"""评测结果可视化模块。

对外入口按链路分组：

* 降水分类检验（TS 系列）：``PrecipitationPlotter`` 出图，``build_ts_report`` 写报告，
  ``summarize`` / ``diagnose`` 是报告用的统计；
* 确定性连续场：``FieldScorePlotter`` 出图，``build_field_report`` 写报告，
  指标语义与谱带见 ``METRIC_SEMANTICS`` / ``SPECTRUM_BANDS`` / ``spectrum_bands``；
* 通用派发：``detect_capability`` / ``build_report`` 从产物目录认流程再派发，
  认不出或没模板时明确报错，不静默回落。
"""

from xmetai_evaluation.visualization.field_plots import (
    FieldScorePlotter,
    load_init_spectra,
    load_wavenumber_spectra,
    prepare_field_dataframe,
    spectrum_series,
)
from xmetai_evaluation.visualization.field_report import (
    METRIC_SEMANTICS,
    SPECTRUM_BANDS,
    build_field_report,
    spectrum_bands,
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
    "load_wavenumber_spectra",
    "load_init_spectra",
    "spectrum_series",
    "build_field_report",
    "METRIC_SEMANTICS",
    "SPECTRUM_BANDS",
    "spectrum_bands",
    "build_report",
    "detect_capability",
    "summarize",
    "diagnose",
]
