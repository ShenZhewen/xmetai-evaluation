# -*- coding: utf-8 -*-
"""格点数据源的布局声明。

一个数据源与另一个数据源的差别只有三件事：

1. 文件长什么样（目录模板、文件名模式、时效从哪来）；
2. 变量怎么映射（源变量名 -> 标准名）；
3. 单位怎么换算（scale / offset）。

把这三件事写成 ``GriddedLayout``，读取逻辑就只需要 ``io/gridded.py`` 里的一份实现。
新增模型（NJU-Earth / PuYun / W2S / AIFS 等）只需在这里加一条声明。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

from xmetai_evaluation.core.variables import DataKind, TemporalKind

GRAVITY = 9.80665


@dataclass(frozen=True)
class VarSpec:
    """标准变量的解码声明。

    Attributes:
        source: 文件里的变量名；``None`` 表示"文件里唯一的变量"（如 cfgrib 按过滤器取出的单变量）。
        unit: 换算后的标准单位。
        scale: 数值换算系数，标准值 = 源值 * scale + offset。
        offset: 数值换算偏移。
        source_unit: 换算前的单位，写入 ``attrs["original_units"]``。
        temporal_kind: 该变量的时间语义。
    """

    source: Optional[str] = None
    unit: str = "unknown"
    scale: float = 1.0
    offset: float = 0.0
    source_unit: str = ""
    temporal_kind: TemporalKind = TemporalKind.INSTANTANEOUS


@dataclass(frozen=True)
class PatternSpec:
    """文件名模式。

    ``init`` 命名组缺省时，时刻取所在目录代表的时刻（如 ``root/YYYYMMDD/001.nc``）；
    ``lead`` 命名组在 ``lead_from="filename"`` 的预报布局里必需。
    分组名 ``group`` 可以来自正则，也可以由 ``group`` 字段直接声明。
    """

    regex: str
    group: Optional[str] = None


@dataclass(frozen=True)
class GriddedLayout:
    """一个格点数据源的完整布局声明。"""

    name: str
    patterns: Tuple[PatternSpec, ...]
    variables: Dict[str, VarSpec] = field(default_factory=dict)
    kind: DataKind = DataKind.GRIDDED_FORECAST
    engine: str = "netcdf"
    #: 默认后端打开失败时的回退后端（Windows 非 ASCII 路径下 netCDF4 会报 ENOENT）
    engine_fallback: Optional[str] = "h5netcdf"
    root_template: str = "{root}/{init:%Y%m%d}"
    init_format: str = "%Y%m%d%H"
    time_kind: str = "forecast"
    lead_from: str = "filename"
    step_hours: float = 6.0
    member_dim: Optional[str] = None
    ensure_member_dim: bool = False
    #: 成员子目录通配（如 "member_*"）；设置后成员从子目录名解析
    member_glob: Optional[str] = None
    #: 成员维度名（成员来自目录结构时使用）
    member_coord: str = "member"
    groups: Dict[str, str] = field(default_factory=dict)
    grib_filters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    squeeze_dims: Tuple[str, ...] = ()
    transpose_dims: Tuple[str, ...] = ()
    case_insensitive: bool = False
    key_kind: str = "init_lead_group"

    @property
    def per_variable_open(self) -> bool:
        """是否需要按变量分别打开文件（cfgrib 需要按 filter_by_keys 过滤）。"""
        return bool(self.grib_filters)

    def groups_for(self, variables: Sequence[str]) -> set:
        """请求变量涉及的文件分组。"""
        return {
            self.groups[name]
            for name in (str(variable).lower() for variable in variables)
            if name in self.groups
        }

    def spec_for(self, standard_name: str) -> Optional[VarSpec]:
        return self.variables.get(str(standard_name))

    def standard_name_for(self, file_variable: str) -> Optional[str]:
        """源变量名 -> 标准名；未声明时返回 None（由调用方按小写兜底）。"""
        for standard_name, spec in self.variables.items():
            if spec.source is None:
                continue
            if spec.source == file_variable:
                return standard_name
            if self.case_insensitive and spec.source.lower() == str(file_variable).lower():
                return standard_name
        return None

    def with_step_hours(self, step_hours: float) -> "GriddedLayout":
        return replace(self, step_hours=float(step_hours))


FUXI_LAYOUT = GriddedLayout(
    name="fuxi",
    patterns=(PatternSpec(r"(?P<lead_index>\d{3})\.nc"),),
    variables={
        "tp": VarSpec(source="tp", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
        "t2m": VarSpec(source="t2m", unit="K"),
        "z500": VarSpec(source="z500", unit="m", scale=1.0 / GRAVITY, source_unit="m^2/s^2"),
    },
    kind=DataKind.GRIDDED_FORECAST,
    lead_from="index",
    step_hours=6.0,
    case_insensitive=True,
    key_kind="init",
)

#: FuXi 集合预报：root/YYYYMMDD/member_*/001.nc…，时效按成员目录内文件序号 * step
FUXI_ENS_LAYOUT = GriddedLayout(
    name="fuxi_ens",
    patterns=(PatternSpec(r"(?P<lead_index>\d{3})\.nc"),),
    variables={
        "tp": VarSpec(source="tp", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
        "t2m": VarSpec(source="t2m", unit="K"),
        "z500": VarSpec(source="z500", unit="m", scale=1.0 / GRAVITY, source_unit="m^2/s^2"),
    },
    kind=DataKind.GRIDDED_FORECAST,
    lead_from="index",
    step_hours=6.0,
    case_insensitive=True,
    member_glob="member_*",
    member_dim="member",
    key_kind="init_member_lead",
)

FENGQING_LAYOUT = GriddedLayout(
    name="fengqing",
    patterns=(
        PatternSpec(
            r"Fengqing_1\.0_GLB_(?P<group>PLEVELS|SURFACE)_OP25_6HOR_ENS_FCST_"
            r"(?P<init>\d{10})_(?P<lead>\d{3})\.nc"
        ),
    ),
    variables={
        "z500": VarSpec(source="Z500", unit="m", scale=1.0 / GRAVITY, source_unit="m^2/s^2"),
        "tp": VarSpec(source="TP", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
        "t2m": VarSpec(source="T2M", unit="K"),
        "msl": VarSpec(source="MSL", unit="Pa"),
        "u10": VarSpec(source="U10", unit="m/s"),
        "v10": VarSpec(source="V10", unit="m/s"),
    },
    kind=DataKind.GRIDDED_FORECAST,
    member_dim="member",
    ensure_member_dim=True,
    groups={
        "z500": "PLEVELS",
        "z": "PLEVELS",
        "gh": "PLEVELS",
        "t": "PLEVELS",
        "u": "PLEVELS",
        "v": "PLEVELS",
        "q": "PLEVELS",
        "tp": "SURFACE",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "u10": "SURFACE",
        "v10": "SURFACE",
    },
    squeeze_dims=("level", "time", "dtime"),
    transpose_dims=("member", "lat", "lon"),
    key_kind="init_lead_group",
)

CRA_LAYOUT = GriddedLayout(
    name="cra",
    patterns=(
        PatternSpec(r"ART_ATM_GLB_0P25_6HOR_ANAL_(?P<init>\d{10})\.grib2", group="ATM"),
        PatternSpec(
            r"CRA40LAND_SURFACE_(?P<init>\d{10})_GLB_0P25_HOUR_V1_0_0\.grib",
            group="SURFACE",
        ),
    ),
    variables={
        "z500": VarSpec(unit="m"),
        "gh": VarSpec(unit="m"),
        "t2m": VarSpec(unit="K"),
        "msl": VarSpec(unit="Pa"),
        "u10": VarSpec(unit="m/s"),
        "v10": VarSpec(unit="m/s"),
        "tp": VarSpec(unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
    },
    kind=DataKind.GRIDDED_OBSERVATION,
    engine="cfgrib",
    time_kind="observation",
    groups={
        "z500": "ATM",
        "z": "ATM",
        "gh": "ATM",
        "t": "ATM",
        "u": "ATM",
        "v": "ATM",
        "q": "ATM",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "u10": "SURFACE",
        "v10": "SURFACE",
        "tp": "SURFACE",
    },
    grib_filters={
        "z500": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "gh": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
        "msl": {"shortName": "msl", "typeOfLevel": "meanSea"},
        "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
        "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
        "tp": {"shortName": "tp", "stepType": "accum"},
    },
    key_kind="time_group",
)

LAYOUTS: Dict[str, GriddedLayout] = {
    FUXI_LAYOUT.name: FUXI_LAYOUT,
    FENGQING_LAYOUT.name: FENGQING_LAYOUT,
    CRA_LAYOUT.name: CRA_LAYOUT,
}
