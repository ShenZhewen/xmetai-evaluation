"""
FuXi 模型输出 Reader

数据布局：root/YYYYMMDD/001.nc, 002.nc, ..., 060.nc
每个文件包含单个时效（6h 步长）的全球 0.25° 网格场。

支持：
- 确定性预报（root/YYYYMMDD/*.nc）
- 集合预报（root/YYYYMMDD/member_*/*.nc）- 后续扩展
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

import xarray as xr
import numpy as np

from xmetai_evaluation.io.base import Reader, DataCatalog
from xmetai_evaluation.core.contracts import (
    DataRequest,
    DataIndex,
    DataBundle,
    DataKind,
    SemanticMetadata,
    Provenance,
)
from xmetai_evaluation.core.errors import DiscoveryError, DecodeError
from xmetai_evaluation.core.variables import TemporalKind


class FuXiCatalog(DataCatalog):
    """
    FuXi 输出文件目录

    扫描 root/YYYYMMDD/ 目录，发现所有时效文件。
    """

    def __init__(self, root_dir: Path, step_hours: float = 6.0):
        """
        Args:
            root_dir: 数据根目录
            step_hours: 时效步长（小时），默认 6
        """
        self.root_dir = Path(root_dir)
        self.step_hours = step_hours

    def discover(self, request: DataRequest) -> DataIndex:
        """
        发现指定起报日期的所有时效文件

        Args:
            request: 需要包含 init_times（起报时间列表）

        Returns:
            DataIndex，available 包含 {init_date: [file_paths]}
        """
        if not self.root_dir.exists():
            raise DiscoveryError(
                f"Root directory does not exist: {self.root_dir}",
                source_id=request.source_id,
            )

        if not request.init_times:
            raise DiscoveryError(
                "FuXiCatalog requires init_times in request",
                source_id=request.source_id,
            )

        # 按起报时间扫描文件
        files_by_init = {}
        for init_time in request.init_times:
            # 起报日期目录：YYYYMMDD
            init_dir = self.root_dir / init_time.strftime("%Y%m%d")
            if not init_dir.exists():
                # 跳过缺失的起报日期（允许部分缺失）
                continue

            # 查找时效文件：001.nc, 002.nc, ..., 060.nc
            nc_files = sorted(init_dir.glob("*.nc"))
            if not nc_files:
                continue

            files_by_init[init_time] = nc_files

        if not files_by_init:
            raise DiscoveryError(
                f"No forecast files found in {self.root_dir} for requested init_times",
                source_id=request.source_id,
            )

        # 返回索引
        return DataIndex(
            source_id=request.source_id,
            available=[files_by_init],  # 列表包含一个字典
            ambiguous=[],
        )


class FuXiReader(Reader):
    """
    FuXi 输出 Reader

    读取多个时效文件，拼接为完整时间序列 (lead_time, lat, lon)。

    变量映射：
    - TP -> tp（降水）
    - T2M -> t2m（2米温度）
    - 保留原始变量名的其他场
    """

    def __init__(self, source_id: str = "fuxi", version: str = "1.0.0", step_hours: float = 6.0):
        super().__init__(source_id=source_id, version=version)
        self.step_hours = step_hours

    def read_one(self, request: DataRequest, index: DataIndex, init_time: datetime) -> DataBundle:
        """只读取单个起报，避免全年预报同时驻留内存。"""
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)
        files_by_init = index.available[0]
        if init_time not in files_by_init:
            raise DecodeError(
                f"No forecast files for init time {init_time}",
                source_id=request.source_id,
            )

        lead_datasets = []
        input_files = []
        for i, fpath in enumerate(files_by_init[init_time]):
            try:
                with xr.open_dataset(fpath) as source:
                    ds = source.load()
                if request.variables and request.variables != ["*"]:
                    var_map = {v.upper(): v for v in request.variables}
                    rename = {
                        file_var: var_map[file_var.upper()]
                        for file_var in ds.data_vars
                        if file_var.upper() in var_map
                    }
                    if not rename:
                        raise DecodeError(
                            f"None of requested variables {request.variables} found in {fpath}",
                            source_id=request.source_id, path=str(fpath),
                        )
                    ds = ds.rename(rename)[list(rename.values())]
                ds = ds.expand_dims(lead_time=[(i + 1) * self.step_hours])
                lead_datasets.append(ds)
                input_files.append(str(fpath))
            except DecodeError:
                raise
            except Exception as exc:
                raise DecodeError(
                    f"Failed to read {fpath}: {exc}",
                    source_id=request.source_id, path=str(fpath), cause=exc,
                ) from exc

        combined = xr.concat(lead_datasets, dim="lead_time").expand_dims(init_time=[init_time])
        units = {
            var: combined[var].attrs.get("units", "mm" if var.upper() == "TP" else "unknown")
            for var in combined.data_vars
        }
        semantic = SemanticMetadata(
            units=units,
            temporal_kind=TemporalKind.INTERVAL_ACCUMULATION if "tp" in units else TemporalKind.INSTANTANEOUS,
            grid_type="regular_latlon",
        )
        return DataBundle(
            payload=combined,
            kind=DataKind.GRIDDED_FORECAST,
            source_id=request.source_id,
            standard_vars={var: var for var in combined.data_vars},
            semantic=semantic,
            provenance=Provenance(
                input_files=input_files,
                reader_id=self.source_id,
                reader_version=self.version,
            ),
        )

        """
        读取所有时效文件并拼接

        Args:
            request: 数据请求（需要 init_times, variables）
            index: 文件索引（来自 FuXiCatalog）

        Returns:
            DataBundle，payload 为 xr.Dataset with dims (init_time, lead_time, lat, lon)
        """
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)

        files_by_init = index.available[0]  # Dict[datetime, List[Path]]

        # 读取所有起报时间的数据
        all_datasets = []
        all_input_files = []

        for init_time in sorted(files_by_init.keys()):
            file_paths = files_by_init[init_time]

            # 读取并拼接该起报时间的所有时效
            lead_datasets = []
            for i, fpath in enumerate(file_paths):
                try:
                    ds = xr.open_dataset(fpath)

                    # 提取请求的变量
                    if request.variables and request.variables != ["*"]:
                        # 变量名映射（大写 -> 小写）
                        var_map = {v.upper(): v for v in request.variables}
                        available_vars = []
                        for file_var in ds.data_vars:
                            std_var = var_map.get(file_var.upper())
                            if std_var:
                                if std_var != file_var:
                                    ds = ds.rename({file_var: std_var})
                                available_vars.append(std_var)

                        if not available_vars:
                            raise DecodeError(
                                f"None of requested variables {request.variables} found in {fpath}",
                                source_id=request.source_id,
                                path=str(fpath),
                            )

                        ds = ds[available_vars]

                    # 添加 lead_time 坐标（小时）
                    lead_hours = (i + 1) * self.step_hours
                    ds = ds.expand_dims(lead_time=[lead_hours])

                    lead_datasets.append(ds)
                    all_input_files.append(str(fpath))

                except Exception as e:
                    raise DecodeError(
                        f"Failed to read {fpath}: {e}",
                        source_id=request.source_id,
                        path=str(fpath),
                        cause=e,
                    )

            # 拼接该起报时间的所有时效
            init_ds = xr.concat(lead_datasets, dim="lead_time")
            init_ds = init_ds.expand_dims(init_time=[init_time])
            all_datasets.append(init_ds)

        # 拼接所有起报时间
        combined = xr.concat(all_datasets, dim="init_time")

        # 构建语义元数据
        units = {}
        for var in combined.data_vars:
            var_upper = var.upper()
            if var_upper == "TP":
                units[var] = "mm"
            elif var_upper in ["T2M", "T700", "T850"]:
                units[var] = "K"
            elif var_upper in ["U10M", "V10M", "U850", "V850"]:
                units[var] = "m/s"
            elif var_upper == "MSL":
                units[var] = "Pa"
            elif var_upper == "Z500":
                units[var] = "m^2/s^2"
            else:
                units[var] = combined[var].attrs.get("units", "unknown")

        semantic = SemanticMetadata(
            units=units,
            temporal_kind=TemporalKind.INTERVAL_ACCUMULATION if "tp" in units else TemporalKind.INSTANTANEOUS,
            grid_type="regular_latlon",
        )

        # 构建溯源
        provenance = Provenance(
            input_files=all_input_files,
            reader_id=self.source_id,
            reader_version=self.version,
        )

        # 标准变量映射
        standard_vars = {var: var for var in combined.data_vars}

        return DataBundle(
            payload=combined,
            kind=DataKind.GRIDDED_FORECAST,
            source_id=request.source_id,
            standard_vars=standard_vars,
            semantic=semantic,
            provenance=provenance,
        )

    @classmethod
    def with_catalog(cls, root_dir: Path, source_id: str = "fuxi", step_hours: float = 6.0):
        """
        便捷构造器：自动创建配套的 Catalog

        Returns:
            (reader, catalog) tuple
        """
        reader = cls(source_id=source_id, step_hours=step_hours)
        catalog = FuXiCatalog(root_dir=root_dir, step_hours=step_hours)
        return reader, catalog
