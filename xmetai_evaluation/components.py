# -*- coding: utf-8 -*-
"""内置组件注册。

所有内置 Reader / Transform / Metric / Writer 的工厂在这里登记一次，
cli 与各流程只按注册名查表构造组件，不再按类型分支判断。

扩展方式：
    新增数据源或算法 -> 在这里补一个注册项（或在自己的插件模块里注册）
    新增评测         -> 只加配置，代码不动
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.core.registry import (
    get_registry,
    register_metric,
    register_protocol,
    register_reader,
    register_transform,
    register_writer,
)


@dataclass
class SourceHandle:
    """一个数据源的读写句柄：Reader + Catalog + 原始配置。"""

    reader: Any
    catalog: Any
    config: Dict[str, Any]

    @property
    def source_id(self) -> str:
        return self.config.get("source_id") or getattr(self.reader, "source_id", "")

    def option(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


def _require_root(params: Dict[str, Any], reader_name: str) -> Path:
    root = params.get("root_dir") or params.get("root")
    if not root:
        raise ConfigError(f"reader '{reader_name}' 需要 root_dir")
    return Path(root)


def _gridded_source(
    layout, params: Dict[str, Any], default_source_id: str, step_hours=None
) -> SourceHandle:
    """通用格点数据源：具名布局 + 通用 Reader/Catalog。

    数据源之间的差别只有布局声明（见 ``io/layouts.py``），
    所以这里不需要为每个模型写一个 Reader 类。
    """
    from xmetai_evaluation.io.gridded import GriddedCatalog, GriddedReader

    selected = layout.with_step_hours(step_hours) if step_hours is not None else layout
    source_id = params.get("source_id", default_source_id)
    root = _require_root(params, selected.name)
    return SourceHandle(
        reader=GriddedReader(source_id=source_id, layout=selected),
        catalog=GriddedCatalog(root, selected),
        config=params,
    )


# --------------------------------------------------------------------------
# Reader：只负责"文件怎么变成标准 Dataset"
# --------------------------------------------------------------------------


def _fuxi_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import FUXI_LAYOUT

    return _gridded_source(
        FUXI_LAYOUT, params, "fuxi", float(params.get("step_hours", 6.0))
    )


def _fuxi_ensemble_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import FUXI_ENS_LAYOUT

    return _gridded_source(
        FUXI_ENS_LAYOUT, params, "fuxi_ens", float(params.get("step_hours", 6.0))
    )


def _station_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.station_reader import (
        DiamondStationCatalog,
        DiamondStationReader,
        load_station_whitelist,
    )

    source_id = params.get("source_id", "diamond_station")
    root = _require_root(params, "station")
    whitelist = load_station_whitelist(
        params.get("station_whitelist") or params.get("station_list")
    )
    return SourceHandle(
        reader=DiamondStationReader(source_id=source_id, station_whitelist=whitelist),
        catalog=DiamondStationCatalog(root, station_whitelist=whitelist),
        config=params,
    )


def _fengqing_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import FENGQING_LAYOUT

    return _gridded_source(FENGQING_LAYOUT, params, "fengqing")


def _phys_source(layout_name: str, default_source_id: str):
    """按 xu 报告口径（z500 = m²/s²、q = g/kg）读预报的通用工厂。

    与 ``_fuxi_source`` 等的差别只在那份布局声明，所以共用 ``_gridded_source``；
    ``step_hours`` 缺省时用布局自己声明的值。
    """

    def factory(**params) -> SourceHandle:
        from xmetai_evaluation.io.layouts import LAYOUTS

        override = params.get("step_hours")
        return _gridded_source(
            LAYOUTS[layout_name],
            params,
            default_source_id,
            float(override) if override is not None else None,
        )

    return factory


def _cra_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import CRA_LAYOUT

    return _gridded_source(CRA_LAYOUT, params, "cra")


def _era5_zarr_source(**params) -> SourceHandle:
    """ERA5 再分析实况（zarr）。

    与通用格点源的区别是数据形态：一个 store 装全部时次，不靠文件名发现，
    所以有自己的 Catalog；变量/单位/分组仍走 ``ERA5_ZARR_LAYOUT`` 声明。
    一个 run 可以给两个 store（``{"pl": ..., "sfc": ...}``）。
    """
    from xmetai_evaluation.io.era5_zarr_reader import Era5ZarrCatalog, Era5ZarrReader

    source_id = params.get("source_id", "era5_zarr")
    stores = params.get("stores") or {}
    if not stores:
        raise ConfigError("reader 'era5_zarr' 需要 stores（如 {'pl': ..., 'sfc': ...}）")
    return SourceHandle(
        reader=Era5ZarrReader(source_id=source_id, stores=stores),
        catalog=Era5ZarrCatalog(stores),
        config=params,
    )


def _climatology_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.climatology_reader import (
        ClimatologyCatalog,
        ClimatologyReader,
    )

    source_id = params.get("source_id", "climatology")
    root = _require_root(params, "climatology")
    engine = str(params.get("engine", "cfgrib"))
    return SourceHandle(
        reader=ClimatologyReader(source_id=source_id, engine=engine),
        catalog=ClimatologyCatalog(root),
        config=params,
    )


def _daily_climatology_source(**params) -> SourceHandle:
    """单文件日序气候态（与按 MMDDHH 找文件的 climatology 是两种索引方式）。"""
    from xmetai_evaluation.io.daily_climatology_reader import (
        DEFAULT_WINDOW,
        DailyClimatologyCatalog,
        DailyClimatologyReader,
    )

    source_id = params.get("source_id", "daily_climatology")
    root = _require_root(params, "daily_climatology")
    return SourceHandle(
        reader=DailyClimatologyReader(
            source_id=source_id,
            window=int(params.get("window", DEFAULT_WINDOW)),
            scales=params.get("scales"),
            units=params.get("units"),
        ),
        catalog=DailyClimatologyCatalog(root),
        config=params,
    )


def _ref_probability_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.ref_probability_reader import RefProbabilityReader

    source_id = params.get("source_id", "ref_probability")
    root = _require_root(params, "ref_probability")
    return SourceHandle(
        reader=RefProbabilityReader(root_dir=root, source_id=source_id),
        catalog=None,
        config=params,
    )


# --------------------------------------------------------------------------
# Transform：只负责"怎么把两个数据集变成可比的"
# --------------------------------------------------------------------------


def _transform_grid_to_station(method: str = "bilinear", **params):
    from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator

    return GridToStationInterpolator(method)


def _transform_time_window(window_hours: float, time_dim: str = "lead_time", **params):
    from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator

    return TimeWindowAccumulator(window_hours=float(window_hours), time_dim=time_dim)


def _transform_ensemble_mean(member_dim: str = "member", **params):
    from xmetai_evaluation.transforms.regrid import EnsembleMeanTransform

    return EnsembleMeanTransform(member_dim=member_dim)


# --------------------------------------------------------------------------
# Metric：只负责"算什么差异"
# --------------------------------------------------------------------------


def _metric_rmse(**params):
    from xmetai_evaluation.metrics.rmse import RMSE

    return RMSE(params)


def _metric_bias(**params):
    from xmetai_evaluation.metrics.bias import Bias

    return Bias(params)


def _metric_acc(climatology_path: Optional[str] = None, **params):
    from xmetai_evaluation.metrics.acc import ACC

    return ACC({"climatology_path": climatology_path, **params})


def _metric_acc_uncentered(climatology_path: Optional[str] = None, **params):
    from xmetai_evaluation.metrics.acc import ACC

    return ACC({"climatology_path": climatology_path, "centered": False, **params})


def _normalize_thresholds(thresholds: Optional[List[Any]]) -> List[Any]:
    """阈值统一成 ``[(等级名, 阈值), ...]``；纯数字自动命名 ``≥<值>``。"""
    normalized: List[Any] = []
    for item in thresholds or []:
        if isinstance(item, (tuple, list)):
            normalized.append((str(item[0]), float(item[1])))
        else:
            normalized.append((f"≥{float(item):g}", float(item)))
    return normalized


def _metric_ts_score(thresholds: Optional[List[Any]] = None, **params):
    from xmetai_evaluation.metrics.categorical import TSScore

    return TSScore(thresholds=_normalize_thresholds(thresholds), params=params)


def _metric_ensemble_probability(thresholds: Optional[List[Any]] = None, **params):
    from xmetai_evaluation.metrics.probabilistic import EnsembleProbabilityScore

    return EnsembleProbabilityScore(
        thresholds=_normalize_thresholds(thresholds), params=params
    )


def _metric_crps(**params):
    from xmetai_evaluation.metrics.ensemble import CRPS

    return CRPS(params)


def _metric_spread_error(**params):
    from xmetai_evaluation.metrics.ensemble import SpreadError

    return SpreadError(params)


def _metric_fss(thresholds: Optional[List[Any]] = None, windows=None, **params):
    from xmetai_evaluation.metrics.spatial import DEFAULT_WINDOWS, FractionsSkillScore

    return FractionsSkillScore(
        thresholds=_normalize_thresholds(thresholds),
        windows=windows or DEFAULT_WINDOWS,
        params=params,
    )


def _metric_activity(**params):
    from xmetai_evaluation.metrics.specialized import ActivityRatio

    return ActivityRatio(params)


def _metric_spectrum(max_wavenumber: int = 30, **params):
    from xmetai_evaluation.metrics.specialized import PowerSpectrum

    return PowerSpectrum(max_wavenumber=int(max_wavenumber), params=params)


def _metric_zonal_spectrum(max_wavenumber: int = 30, **params):
    from xmetai_evaluation.metrics.specialized import ZonalSpectrum

    return ZonalSpectrum(max_wavenumber=int(max_wavenumber), params=params)


# --------------------------------------------------------------------------
# Writer：只负责"长表之外的派生视图"
# --------------------------------------------------------------------------


def _writer_csv_long(tables, context, output_dir: Path) -> Path:
    """统一长表本身由 ResultStore 无条件写出，这里只为让配置能显式声明它。"""
    return Path(output_dir) / "scores.csv"


def _writer_coverage(tables, context, output_dir: Path) -> Path:
    path = Path(output_dir) / "coverage.csv"
    tables.coverage.to_csv(path, index=False, encoding="utf-8")
    return path


def _writer_details(tables, context, output_dir: Path) -> Path:
    """列联表计数等诊断量（长表形态）。"""
    path = Path(output_dir) / "diagnostics" / "scores_detail.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tables.details.to_csv(path, index=False, encoding="utf-8")
    return path


def _writer_json(tables, context, output_dir: Path) -> Path:
    path = Path(output_dir) / "scores.json"
    records = json.loads(tables.score_table.to_json(orient="records", force_ascii=False))
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "run_id": context.run_id,
                "model_id": context.model_id,
                "dataset_id": context.dataset_id,
                "protocol_id": context.protocol_id,
                "scores": records,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    return path


def _writer_categorical_wide(tables, context, output_dir: Path) -> Path:
    from xmetai_evaluation.results.table import categorical_wide

    path = Path(output_dir) / "diagnostics" / "categorical_wide.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    categorical_wide(tables).to_csv(path, index=False, encoding="utf-8")
    return path


def _writer_probability_wide(tables, context, output_dir: Path) -> Path:
    from xmetai_evaluation.results.table import probability_wide

    path = Path(output_dir) / "diagnostics" / "probability_wide.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_wide(tables).to_csv(path, index=False, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Protocol：只负责"样本空间怎么遍历、怎么配对"
# --------------------------------------------------------------------------


def _protocol_station_valid_time(spec):
    from xmetai_evaluation.pipeline.protocols import StationValidTimeProtocol

    return StationValidTimeProtocol(spec)


def _protocol_grid_valid_time(spec):
    from xmetai_evaluation.pipeline.protocols import GridValidTimeProtocol

    return GridValidTimeProtocol(spec)


_REGISTERED = False


def register_builtin_components() -> None:
    """注册全部内置组件（幂等）。"""
    global _REGISTERED
    if _REGISTERED:
        return
    # 先占用标记，避免注册过程中重入导致重复注册。
    _REGISTERED = True

    register_reader("fuxi", "1.0.0", _fuxi_source, "FuXi 网格预报：root/YYYYMMDD/NNN.nc")
    register_reader(
        "fuxi_ens",
        "1.0.0",
        _fuxi_ensemble_source,
        "FuXi 集合预报：root/YYYYMMDD/member_*/NNN.nc",
    )
    register_reader("station", "1.0.0", _station_source, "Diamond 格式站点观测")
    register_reader(
        "diamond_station", "1.0.0", _station_source, "Diamond 格式站点观测（别名）"
    )
    register_reader("fengqing", "1.0.0", _fengqing_source, "Fengqing 集合预报")
    register_reader(
        "fuxi_phys",
        "1.0.0",
        _phys_source("fuxi_phys", "fuxi"),
        "FuXi 网格预报（xu 报告口径：z500 = m²/s²、q = g/kg）",
    )
    register_reader(
        "fuxi_ens_phys",
        "1.0.0",
        _phys_source("fuxi_ens_phys", "fuxi_ens"),
        "FuXi 集合预报（xu 报告口径：z500 = m²/s²、q = g/kg）",
    )
    register_reader(
        "fengqing_phys",
        "1.0.0",
        _phys_source("fengqing_phys", "fengqing"),
        "风清单卡预报（xu 报告口径：z500 = m²/s²、q = g/kg）",
    )
    register_reader("cra", "1.0.0", _cra_source, "CRA40 再分析实况")
    register_reader(
        "era5_zarr",
        "1.0.0",
        _era5_zarr_source,
        "ERA5 再分析实况（zarr store，气压层/地面分 store）",
    )
    register_reader(
        "climatology",
        "1.0.0",
        _climatology_source,
        "CRA CLI_6HOUR 气候态参考场（按 月日+时次 索引）",
    )
    register_reader(
        "daily_climatology",
        "1.0.0",
        _daily_climatology_source,
        "日序气候态参考场（单文件，按年内日序索引，含环形平滑）",
    )
    register_reader(
        "ref_probability",
        "1.0.0",
        _ref_probability_source,
        "BSS 逐 6h 气候概率参考（ref/MMDDHH.000，站号+4阈值概率）",
    )

    register_transform(
        "grid_to_station", "1.0.0", _transform_grid_to_station, "网格场插值到站点"
    )
    register_transform(
        "time_window_accumulator",
        "1.0.0",
        _transform_time_window,
        "固定窗口时间累积（如 24h 降水）",
    )
    register_transform("ensemble_mean", "1.0.0", _transform_ensemble_mean, "集合均值（DataBundle -> DataBundle）")

    register_metric("rmse", "1.0.0", _metric_rmse, "均方根误差")
    register_metric("bias", "1.0.0", _metric_bias, "平均误差")
    register_metric("acc", "1.0.0", _metric_acc, "距平相关系数")
    register_metric(
        "acc_uncentered",
        "1.0.0",
        _metric_acc_uncentered,
        "距平相关系数（uncentered，FDP/WeatherBench2 口径）",
    )
    register_metric("ts_score", "1.0.0", _metric_ts_score, "分类检验 TS/POD/FAR/BIAS")
    register_metric(
        "ensemble_probability",
        "1.0.0",
        _metric_ensemble_probability,
        "集合概率评分 AROC/BS/BSS（需要原始成员）",
    )
    register_metric("crps", "1.0.0", _metric_crps, "集合 CRPS（纬度加权闭式解）")
    register_metric(
        "spread_error", "1.0.0", _metric_spread_error, "集合离散度-误差比（含 Spread 与 RMSE）"
    )
    register_metric(
        "fss", "1.0.0", _metric_fss, "邻域分数技巧评分（空间检验，需网格场）"
    )
    register_metric(
        "activity", "1.0.0", _metric_activity, "活跃度比（距平加权标准差之比，需气候态）"
    )
    register_metric(
        "spectrum", "1.0.0", _metric_spectrum, "功率谱（二维 FFT，对原始场，0-max_k 波）"
    )
    register_metric(
        "zonal_spectrum",
        "1.0.0",
        _metric_zonal_spectrum,
        "纬向功率谱（去纬向均值 rfft，cos 纬度加权）",
    )

    register_protocol(
        "station_valid_time",
        "1.0.0",
        _protocol_station_valid_time,
        "格点预报插值到站点，按有效时刻配对；样本键为 lead_h",
    )
    register_protocol(
        "grid_valid_time",
        "1.0.0",
        _protocol_grid_valid_time,
        "格点预报插值到实况网格，按 valid_time 配对；样本键为 valid_time",
    )

    register_writer("csv_long", "1.0.0", lambda **kw: _writer_csv_long, "统一长表 scores.csv")
    register_writer(
        "coverage", "1.0.0", lambda **kw: _writer_coverage, "请求/有效样本覆盖（可由长表派生）"
    )
    register_writer(
        "details",
        "1.0.0",
        lambda **kw: _writer_details,
        "列联表计数等诊断明细（可由宽表替代）",
    )
    register_writer("json", "1.0.0", lambda **kw: _writer_json, "评分 JSON 快照")
    register_writer(
        "categorical_wide", "1.0.0", lambda **kw: _writer_categorical_wide, "分类检验宽表"
    )
    register_writer(
        "probability_wide",
        "1.0.0",
        lambda **kw: _writer_probability_wide,
        "集合概率评分宽表（AROC/BS/BSS）",
    )


def available_components() -> Dict[str, List[str]]:
    """列出各类已注册组件的短名称（用于 --list 与错误提示）。"""
    from xmetai_evaluation.core.registry import ComponentType

    register_builtin_components()
    return {kind.value: get_registry().names(kind) for kind in ComponentType}
