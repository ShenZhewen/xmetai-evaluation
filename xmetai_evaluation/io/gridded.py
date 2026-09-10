# -*- coding: utf-8 -*-
"""通用格点 Reader：一份实现驱动所有 ``GriddedLayout``。

分工：

    GriddedCatalog  只负责文件发现，产出索引，不打开文件；
    GriddedReader   只负责解码、变量/坐标/单位规范化，产出标准 DataBundle。

输出契约（README 第 4 节）：维度只用标准名、变量只用标准名、
单位在 reader 内归一化完毕、``valid_time = init_time + lead_time`` 显式存在。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    DataIndex,
    DataRequest,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError
from xmetai_evaluation.core.variables import TemporalKind
from xmetai_evaluation.io.base import DataCatalog, Reader
from xmetai_evaluation.io.layouts import GriddedLayout

# 常见坐标别名 -> 标准坐标名
COORDINATE_ALIASES = {"latitude": "lat", "longitude": "lon"}


@dataclass(frozen=True)
class GriddedRecord:
    """一条"文件 <-> 时刻/时效/分组"的记录。"""

    path: Path
    group: Optional[str] = None
    member: Optional[Any] = None
    init_time: Optional[datetime] = None
    lead_time: Optional[float] = None

    @property
    def valid_time(self) -> Optional[datetime]:
        if self.init_time is None or self.lead_time is None:
            return self.init_time
        return self.init_time + timedelta(hours=float(self.lead_time))


class GriddedCatalog(DataCatalog):
    """按布局声明扫描文件的通用目录。"""

    def __init__(self, root_dir: Path, layout: GriddedLayout):
        self.root_dir = Path(root_dir)
        self.layout = layout

    def discover(self, request: DataRequest) -> DataIndex:
        if not self.root_dir.exists():
            raise DiscoveryError(
                f"Root directory does not exist: {self.root_dir}",
                source_id=request.source_id,
            )
        if not request.init_times:
            raise DiscoveryError(
                f"{self.layout.name} catalog requires init_times in request",
                source_id=request.source_id,
            )

        needed_groups = self.layout.groups_for(request.variables)
        records: List[GriddedRecord] = []
        for moment in request.init_times:
            date_dir = Path(self.layout.root_template.format(root=self.root_dir, init=moment))
            if not date_dir.exists():
                continue
            records.extend(self._scan_dir(date_dir, moment, needed_groups, request))

        if not records:
            raise DiscoveryError(
                f"No {self.layout.name} files found in {self.root_dir} "
                f"for requested times and variables",
                source_id=request.source_id,
            )
        return DataIndex(source_id=request.source_id, available=[self._build_index(records)])

    def _scan_dir(
        self,
        date_dir: Path,
        moment: datetime,
        needed_groups: set,
        request: DataRequest,
    ) -> List[GriddedRecord]:
        records: List[GriddedRecord] = []
        for member, directory in self._iter_member_dirs(date_dir):
            matched: List[GriddedRecord] = []
            for path in sorted(item for item in directory.iterdir() if item.is_file()):
                parsed = self._parse_name(path, moment)
                if parsed is None:
                    continue
                file_time, lead, group = parsed
                if file_time != moment:
                    continue
                if needed_groups and group not in needed_groups:
                    continue
                matched.append(
                    GriddedRecord(
                        path=path,
                        group=group,
                        member=member,
                        init_time=file_time,
                        lead_time=lead,
                    )
                )

            if self.layout.lead_from == "index":
                matched = [
                    replace(record, lead_time=(position + 1) * self.layout.step_hours)
                    for position, record in enumerate(matched)
                ]

            for record in matched:
                if (
                    record.lead_time is not None
                    and request.lead_times
                    and record.lead_time not in request.lead_times
                ):
                    continue
                records.append(record)
        return records

    def _iter_member_dirs(self, date_dir: Path) -> Iterator[Tuple[Any, Path]]:
        """产出 (成员标识, 目录)；没有成员层时产出 (None, 起报目录)。"""
        if not self.layout.member_glob:
            yield None, date_dir
            return
        directories = [
            path for path in date_dir.glob(self.layout.member_glob) if path.is_dir()
        ]
        for path in sorted(directories, key=self._member_sort_key):
            yield self._member_value(path), path

    @staticmethod
    def _member_value(path: Path) -> Any:
        """成员目录 -> 成员标识；默认取目录名里的数字。"""
        match = re.search(r"(\d+)", path.name)
        return int(match.group(1)) if match else path.name

    def _member_sort_key(self, path: Path) -> Tuple[int, Any]:
        value = self._member_value(path)
        if isinstance(value, int):
            return (0, value)
        return (1, str(value))

    def _parse_name(
        self, path: Path, moment: datetime
    ) -> Optional[Tuple[datetime, Optional[float], Optional[str]]]:
        """解析文件名；没有 init 命名组时，时刻取所在目录代表的时刻。"""
        for pattern in self.layout.patterns:
            match = re.fullmatch(pattern.regex, path.name)
            if match is None:
                continue
            groups = match.groupdict()
            init_raw = groups.get("init")
            file_time = (
                datetime.strptime(init_raw, self.layout.init_format) if init_raw else moment
            )
            group = pattern.group or groups.get("group")
            lead: Optional[float] = None
            if self.layout.lead_from == "filename" and self.layout.time_kind == "forecast":
                lead_raw = groups.get("lead")
                if lead_raw is None:
                    continue
                lead = float(lead_raw)
            return file_time, lead, group
        return None

    def _build_index(self, records: Sequence[GriddedRecord]) -> Dict[Any, Any]:
        """把记录列表压成索引（形状由布局的 key_kind 决定）。"""
        kind = self.layout.key_kind
        if kind == "init":
            by_init: Dict[datetime, List[Path]] = {}
            for record in sorted(records, key=lambda item: (item.init_time, item.lead_time)):
                by_init.setdefault(record.init_time, []).append(record.path)
            return by_init
        if kind == "init_member_lead":
            return {
                (record.init_time, record.member, record.lead_time): record.path
                for record in records
            }
        if kind == "time_group":
            return {(record.init_time, record.group): record.path for record in records}
        return {
            (record.init_time, record.lead_time, record.group): record.path
            for record in records
        }

    def _get_file_types_for_variables(self, variables: Sequence[str]) -> set:
        """变量 -> 需要的文件分组（保留旧调用形态，实际由布局声明决定）。"""
        return self.layout.groups_for(variables)


class GriddedReader(Reader):
    """按布局声明解码的通用格点 Reader。"""

    def __init__(self, source_id: str, layout: GriddedLayout, version: str = "2.0.0"):
        super().__init__(source_id=source_id, version=version)
        self.layout = layout
        self.step_hours = layout.step_hours

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        return self._read_records(request, self.records_from_index(index))

    def read_one(
        self, request: DataRequest, index: DataIndex, init_time: datetime
    ) -> DataBundle:
        """只读取单个起报，避免全年预报同时驻留内存。"""
        records = [
            record
            for record in self.records_from_index(index)
            if record.init_time == init_time
        ]
        if not records:
            raise DecodeError(
                f"No {self.layout.name} files for init time {init_time}",
                source_id=request.source_id,
            )
        return self._read_records(request, records)

    def records_from_index(self, index: DataIndex) -> List[GriddedRecord]:
        """把 Catalog 的索引还原成记录列表。"""
        if not index.available:
            raise DecodeError("No files in index", source_id=self.source_id)
        entries = index.available[0]
        kind = self.layout.key_kind
        records: List[GriddedRecord] = []

        if kind == "init":
            for moment, paths in entries.items():
                for position, path in enumerate(sorted(paths)):
                    records.append(
                        GriddedRecord(
                            path=Path(path),
                            init_time=moment,
                            lead_time=(position + 1) * self.layout.step_hours,
                        )
                    )
        elif kind == "time_group":
            for (moment, group), path in entries.items():
                records.append(GriddedRecord(path=Path(path), group=group, init_time=moment))
        elif kind == "init_member_lead":
            for (moment, member, lead), path in entries.items():
                records.append(
                    GriddedRecord(
                        path=Path(path), member=member, init_time=moment, lead_time=lead
                    )
                )
        else:
            for (moment, lead, group), path in entries.items():
                records.append(
                    GriddedRecord(path=Path(path), group=group, init_time=moment, lead_time=lead)
                )
        return records

    def max_lead_hours(self, index: DataIndex) -> float:
        """索引里的最大预报时效（小时）。"""
        leads = [
            record.lead_time
            for record in self.records_from_index(index)
            if record.lead_time is not None
        ]
        return float(max(leads)) if leads else 0.0

    def available_init_times(self, index: DataIndex) -> List[datetime]:
        """索引里出现的起报时刻（去重、升序）。

        调用方不需要知道索引内部是 {init: [paths]} 还是 {(init, member, lead): path}。
        """
        return sorted(
            {
                record.init_time
                for record in self.records_from_index(index)
                if record.init_time is not None
            }
        )

    def input_files(self, index: DataIndex) -> List[str]:
        """索引涉及的文件清单（去重、保持顺序）。"""
        files: List[str] = []
        for record in self.records_from_index(index):
            text = str(record.path)
            if text not in files:
                files.append(text)
        return files

    def _read_records(
        self, request: DataRequest, records: Sequence[GriddedRecord]
    ) -> DataBundle:
        self._warn_on_member_mismatch(records)
        frames: List[Tuple[GriddedRecord, xr.Dataset]] = []
        for record in records:
            dataset = self._read_record(record, request.variables)
            if dataset is not None:
                frames.append((record, dataset))
        if not frames:
            raise DecodeError(
                f"None of requested variables {list(request.variables)} found in "
                f"{len(records)} files",
                source_id=request.source_id,
            )
        return self._build_bundle(request, self._combine(frames), frames)

    def _warn_on_member_mismatch(self, records: Sequence[GriddedRecord]) -> None:
        """成员数在不同起报之间不一致时给出警告（避免静默少样本）。"""
        if not self.layout.member_glob:
            return
        counts: Dict[Optional[datetime], set] = {}
        for record in records:
            counts.setdefault(record.init_time, set()).add(record.member)
        sizes = {moment: len(members) for moment, members in counts.items()}
        if len(set(sizes.values())) > 1:
            warnings.warn(
                f"各起报的集合成员数不一致: "
                f"{ {str(k): v for k, v in sorted(sizes.items())} }；"
                f"缺失成员会以缺测参与平均"
            )

    def _read_record(
        self, record: GriddedRecord, variables: Sequence[str]
    ) -> Optional[xr.Dataset]:
        if self.layout.per_variable_open:
            datasets: List[xr.Dataset] = []
            for variable in variables:
                filters = self.layout.grib_filters.get(str(variable))
                if filters is None:
                    continue
                try:
                    with self._open(record.path, filters) as source:
                        dataset = source.load()
                except DecodeError:
                    raise
                except Exception as exc:
                    warnings.warn(f"Failed to read {variable} from {record.path}: {exc}")
                    continue
                normalized = self._normalize(dataset, [str(variable)])
                if normalized is not None:
                    datasets.append(normalized)
            if not datasets:
                return None
            return datasets[0] if len(datasets) == 1 else xr.merge(datasets)

        try:
            with self._open(record.path, None) as source:
                dataset = source.load()
        except DecodeError:
            raise
        except Exception as exc:
            raise DecodeError(
                f"Failed to read {record.path}: {exc}",
                source_id=self.source_id,
                path=str(record.path),
                cause=exc,
            ) from exc
        return self._normalize(dataset, variables)

    def _open(self, path: Path, filter_by_keys: Optional[Dict[str, Any]]):
        if self.layout.engine == "cfgrib":
            return xr.open_dataset(
                path, engine="cfgrib", backend_kwargs={"filter_by_keys": filter_by_keys}
            )
        try:
            return xr.open_dataset(path)
        except OSError as exc:
            # 非 ASCII 路径下 netCDF4 的 C 层会报 ENOENT（文件其实存在），
            # 回退到 h5netcdf（h5py 走 UTF-8 路径）再试一次。
            fallback = self.layout.engine_fallback
            if not fallback:
                raise
            try:
                dataset = xr.open_dataset(path, engine=fallback)
            except Exception:
                raise exc
            warnings.warn(
                f"默认后端打开失败（{exc}），已回退到 engine={fallback}: {path}"
            )
            return dataset

    def _normalize(self, dataset: xr.Dataset, variables: Sequence[str]) -> Optional[xr.Dataset]:
        """坐标别名、变量改名、单位换算、维度规范化。"""
        dataset = self._standardize_coordinates(dataset)

        renames = {}
        for file_variable in dataset.data_vars:
            standard_name = self.layout.standard_name_for(str(file_variable))
            if standard_name is None:
                spec = self.layout.spec_for(str(file_variable).lower())
                if spec is not None and spec.source is None:
                    continue  # 单变量文件，稍后按请求改名
                standard_name = str(file_variable).lower()
            if standard_name != file_variable:
                renames[file_variable] = standard_name
        if renames:
            dataset = dataset.rename(renames)

        available = [str(name) for name in variables if name in dataset.data_vars]
        if not available:
            unresolved = [
                str(name)
                for name in variables
                if (self.layout.spec_for(str(name)) is not None)
                and self.layout.spec_for(str(name)).source is None
            ]
            if len(dataset.data_vars) == 1 and unresolved:
                dataset = dataset.rename({list(dataset.data_vars)[0]: unresolved[0]})
                available = [unresolved[0]]
        if not available:
            return None
        dataset = dataset[available]

        for name in available:
            spec = self.layout.spec_for(name)
            if spec is None:
                continue
            if spec.scale != 1.0 or spec.offset != 0.0:
                dataset[name] = dataset[name] * spec.scale + spec.offset
                dataset[name].attrs["units"] = spec.unit
                if spec.source_unit:
                    dataset[name].attrs["original_units"] = spec.source_unit

        if self.layout.squeeze_dims:
            singleton = [
                dim
                for dim in self.layout.squeeze_dims
                if dim in dataset.dims and dataset.sizes[dim] == 1
            ]
            if singleton:
                dataset = dataset.squeeze(dim=singleton, drop=True)

        if self.layout.transpose_dims:
            for name in dataset.data_vars:
                order = [dim for dim in self.layout.transpose_dims if dim in dataset[name].dims]
                if order and len(order) == dataset[name].ndim:
                    dataset[name] = dataset[name].transpose(*order)

        if (
            self.layout.member_dim
            and self.layout.ensure_member_dim
            and self.layout.member_dim not in dataset.dims
        ):
            dataset = dataset.expand_dims(self.layout.member_dim)

        return dataset

    def _standardize_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        renames = {
            alias: standard
            for alias, standard in COORDINATE_ALIASES.items()
            if alias in dataset.coords or alias in dataset.dims
        }
        return dataset.rename(renames) if renames else dataset

    def _combine(self, frames: Sequence[Tuple[GriddedRecord, xr.Dataset]]) -> xr.Dataset:
        """按时间语义拼接：预报拼 init_time x lead_time，观测拼 valid_time。"""
        if self.layout.time_kind == "forecast":
            by_init: Dict[datetime, Dict[Optional[float], Dict[Any, List[xr.Dataset]]]] = {}
            for record, dataset in frames:
                by_init.setdefault(record.init_time, {}).setdefault(
                    record.lead_time, {}
                ).setdefault(record.member, []).append(dataset)

            init_datasets = []
            for init_time in sorted(by_init):
                lead_datasets = []
                for lead_time in sorted(by_init[init_time], key=lambda value: (value is None, value)):
                    merged = self._merge_members(by_init[init_time][lead_time])
                    lead_datasets.append(merged.expand_dims(lead_time=[lead_time]))
                combined = xr.concat(lead_datasets, dim="lead_time")
                init_datasets.append(combined.expand_dims(init_time=[init_time]))
            return xr.concat(init_datasets, dim="init_time")

        by_time: Dict[datetime, List[xr.Dataset]] = {}
        for record, dataset in frames:
            by_time.setdefault(record.valid_time, []).append(dataset)
        time_datasets = []
        for moment in sorted(by_time):
            parts = by_time[moment]
            merged = parts[0] if len(parts) == 1 else xr.merge(parts)
            time_datasets.append(merged.expand_dims(valid_time=[moment]))
        return xr.concat(time_datasets, dim="valid_time")

    def _merge_members(self, by_member: Dict[Any, List[xr.Dataset]]) -> xr.Dataset:
        """同一 (init, lead) 的多个成员拼成 member 维；同一成员的多个文件合并。"""
        members = sorted(by_member, key=lambda value: (value is None, value))
        merged_parts = []
        for member in members:
            parts = by_member[member]
            merged_parts.append(parts[0] if len(parts) == 1 else xr.merge(parts))
        if len(merged_parts) == 1:
            return merged_parts[0]

        name = self.layout.member_coord
        if name in merged_parts[0].dims:
            # 成员已经在文件内的维度里（如 Fengqing 内置 member 维）
            return merged_parts[0] if len(merged_parts) == 1 else xr.merge(merged_parts)
        stacked = xr.concat(merged_parts, dim=name)
        return stacked.assign_coords({name: list(members)})

    def _build_bundle(
        self,
        request: DataRequest,
        combined: xr.Dataset,
        frames: Sequence[Tuple[GriddedRecord, xr.Dataset]],
    ) -> DataBundle:
        units: Dict[str, str] = {}
        temporal: Dict[str, TemporalKind] = {}
        for name in combined.data_vars:
            spec = self.layout.spec_for(str(name))
            units[str(name)] = str(
                combined[name].attrs.get("units") or (spec.unit if spec else "unknown")
            )
            temporal[str(name)] = spec.temporal_kind if spec else TemporalKind.INSTANTANEOUS

        member_count = None
        if self.layout.member_dim and self.layout.member_dim in combined.sizes:
            member_count = int(combined.sizes[self.layout.member_dim])

        input_files: List[str] = []
        for record, _ in frames:
            text = str(record.path)
            if text not in input_files:
                input_files.append(text)

        semantic = SemanticMetadata(
            units=units,
            temporal_kind=temporal,
            grid_type="regular_latlon",
            member_count=member_count,
        )
        provenance = Provenance(
            input_files=input_files,
            reader_id=self.source_id,
            reader_version=self.version,
        )
        return DataBundle(
            payload=combined,
            kind=self.layout.kind,
            source_id=request.source_id,
            standard_vars={str(name): str(name) for name in combined.data_vars},
            semantic=semantic,
            provenance=provenance,
        )
