# -*- coding: utf-8 -*-
"""气候态参考场 Reader（CRA ``CLI_6HOUR`` 口径）。

文件名按"月日 + 时次"索引，与年份无关::

    {root}/ART_ATM_GLB_0P25_CLI_ANAL_{MMDD}{HH}.grib2
    例如 2026-08-20T06 对应的气候态是 ART_ATM_GLB_0P25_CLI_ANAL_082006.grib2

读取过滤器与 CRA 实况一致（如 500hPa 位势高度：shortName=gh, level=500）。
输出 ``(valid_time, lat, lon)`` 的标准 DataBundle，作为 ``EvaluationBatch.reference``
供 ACC / 活跃度 / 功率谱等距平类指标使用。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    DataIndex,
    DataRequest,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError
from xmetai_evaluation.core.variables import DataKind, TemporalKind
from xmetai_evaluation.io.base import DataCatalog, Reader

DEFAULT_GRIB_FILTERS: Dict[str, Dict[str, Any]] = {
    "z500": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
    "gh": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
    "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
    "msl": {"shortName": "msl", "typeOfLevel": "meanSea"},
    "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
    "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
}

DEFAULT_UNITS: Dict[str, str] = {
    "z500": "m",
    "gh": "m",
    "t2m": "K",
    "msl": "Pa",
    "u10": "m/s",
    "v10": "m/s",
}


class ClimatologyCatalog(DataCatalog):
    """按"月日+时次"查找气候态文件（与年份无关）。"""

    def __init__(
        self,
        root_dir: Path,
        template: str = "ART_ATM_GLB_0P25_CLI_ANAL_{mmdd}{hh}.grib2",
    ):
        self.root_dir = Path(root_dir)
        self.template = template

    def file_name(self, moment: datetime) -> str:
        return self.template.format(
            mmdd=f"{moment.month:02d}{moment.day:02d}", hh=f"{moment.hour:02d}"
        )

    def discover(self, request: DataRequest) -> DataIndex:
        if not self.root_dir.exists():
            raise DiscoveryError(
                f"气候态根目录不存在: {self.root_dir}", source_id=request.source_id
            )
        if not request.init_times:
            raise DiscoveryError(
                "气候态请求需要 init_times（作为有效时刻）", source_id=request.source_id
            )

        mapping: Dict[datetime, Path] = {}
        for moment in request.init_times:
            # 1) 精确名（模板去后缀 + 常见后缀）
            stem = (self.root_dir / self.file_name(moment)).with_suffix("")
            for suffix in (".grib2", ".grib", ".nc"):
                candidate = stem.with_suffix(suffix)
                if candidate.exists():
                    mapping[moment] = candidate
                    break
            else:
                # 2) 宽松匹配：先按"月日+时次"，再退到"月日"
                key = f"{moment.month:02d}{moment.day:02d}"
                hour = f"{moment.hour:02d}"
                for pattern in (
                    f"*{key}{hour}*.grib2",
                    f"*{key}{hour}*.nc",
                    f"*{key}*.grib2",
                    f"*{key}*.nc",
                ):
                    matches = sorted(self.root_dir.glob(pattern))
                    if matches:
                        mapping[moment] = matches[0]
                        break

        if not mapping:
            raise DiscoveryError(
                f"没有找到任何气候态文件（请求 {len(request.init_times)} 个时刻）",
                source_id=request.source_id,
            )
        return DataIndex(source_id=request.source_id, available=[mapping])


class ClimatologyReader(Reader):
    """读取气候态文件并拼成 ``(valid_time, lat, lon)``。

    ``engine='cfgrib'``（默认，生产数据）或 ``'netcdf'``（其它存储格式／测试）。
    """

    def __init__(
        self,
        source_id: str = "climatology",
        version: str = "1.0.0",
        engine: str = "cfgrib",
        grib_filters: Optional[Dict[str, Dict[str, Any]]] = None,
        units: Optional[Dict[str, str]] = None,
    ):
        super().__init__(source_id=source_id, version=version)
        self.engine = engine
        self.grib_filters = grib_filters or DEFAULT_GRIB_FILTERS
        self.units = units or DEFAULT_UNITS

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)
        mapping: Dict[datetime, Path] = index.available[0]

        frames: List[xr.Dataset] = []
        input_files: List[str] = []
        for moment in sorted(mapping):
            dataset = self._read_one(mapping[moment], request.variables)
            if dataset is None:
                continue
            frames.append(dataset.expand_dims(valid_time=[moment]))
            text = str(mapping[moment])
            if text not in input_files:
                input_files.append(text)
        if not frames:
            raise DecodeError(
                f"气候态里没有请求的变量 {list(request.variables)}",
                source_id=request.source_id,
            )

        combined = frames[0] if len(frames) == 1 else xr.concat(frames, dim="valid_time")
        return DataBundle(
            payload=combined,
            kind=DataKind.REFERENCE,
            source_id=request.source_id,
            standard_vars={str(name): str(name) for name in combined.data_vars},
            semantic=SemanticMetadata(
                units={
                    str(name): self.units.get(str(name), "unknown")
                    for name in combined.data_vars
                },
                temporal_kind=TemporalKind.INSTANTANEOUS,
                grid_type="regular_latlon",
            ),
            provenance=Provenance(
                input_files=input_files,
                reader_id=self.source_id,
                reader_version=self.version,
            ),
        )

    def _read_one(self, path: Path, variables: Sequence[str]) -> Optional[xr.Dataset]:
        datasets: List[xr.Dataset] = []
        for variable in variables:
            name = str(variable)
            if self.engine == "cfgrib":
                filters = self.grib_filters.get(name)
                if filters is None:
                    continue
                with xr.open_dataset(
                    path, engine="cfgrib", backend_kwargs={"filter_by_keys": filters}
                ) as source:
                    dataset = source.load()
            else:
                with xr.open_dataset(path) as source:
                    dataset = source.load()
                if name not in dataset.data_vars:
                    continue
                dataset = dataset[[name]]

            renames = {
                alias: standard
                for alias, standard in (("latitude", "lat"), ("longitude", "lon"))
                if alias in dataset.coords or alias in dataset.dims
            }
            if renames:
                dataset = dataset.rename(renames)
            if name not in dataset.data_vars and len(dataset.data_vars) == 1:
                dataset = dataset.rename({list(dataset.data_vars)[0]: name})
            if name in dataset.data_vars:
                datasets.append(dataset[[name]])

        if not datasets:
            return None
        return datasets[0] if len(datasets) == 1 else xr.merge(datasets)
