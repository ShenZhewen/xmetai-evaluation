"""
Fengqing 模型输出 Reader

数据布局：root/YYYYMMDD/Fengqing_1.0_GLB_{PLEVELS|SURFACE}_OP25_6HOR_ENS_FCST_YYYYMMDDHH_LLL.nc

支持：
- 集合预报（21成员）
- PLEVELS: Z500（需要单位转换 m²/s² -> m）
- SURFACE: T2M, MSL, U10, V10, TP
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


class FengqingCatalog(DataCatalog):
    """
    Fengqing 输出文件目录

    扫描 root/YYYYMMDD/ 目录，发现 PLEVELS 和 SURFACE 文件。
    """

    # 文件名正则：Fengqing_1.0_GLB_PLEVELS_OP25_6HOR_ENS_FCST_2026081900_006.nc
    FILE_PATTERN = re.compile(
        r"Fengqing_1\.0_GLB_(PLEVELS|SURFACE)_OP25_6HOR_ENS_FCST_(\d{10})_(\d{3})\.nc"
    )

    def __init__(self, root_dir: Path):
        """
        Args:
            root_dir: 数据根目录
        """
        self.root_dir = Path(root_dir)

    def discover(self, request: DataRequest) -> DataIndex:
        """
        发现指定起报日期的所有文件

        Args:
            request: 需要包含 init_times、variables

        Returns:
            DataIndex，available 包含 {(init_time, lead_hours, file_type): file_path}
        """
        if not self.root_dir.exists():
            raise DiscoveryError(
                f"Root directory does not exist: {self.root_dir}",
                source_id=request.source_id,
            )

        if not request.init_times:
            raise DiscoveryError(
                "FengqingCatalog requires init_times in request",
                source_id=request.source_id,
            )

        # 确定需要哪些文件类型（PLEVELS or SURFACE）
        file_types_needed = self._get_file_types_for_variables(request.variables)

        # 按起报时间扫描文件
        files_by_key = {}
        for init_time in request.init_times:
            # 起报日期目录：YYYYMMDD
            init_dir = self.root_dir / init_time.strftime("%Y%m%d")
            if not init_dir.exists():
                continue

            # 查找所有 Fengqing 文件
            for nc_file in init_dir.glob("Fengqing_*.nc"):
                match = self.FILE_PATTERN.match(nc_file.name)
                if not match:
                    continue

                file_type, init_str, lead_str = match.groups()

                # 检查文件类型是否需要
                if file_type not in file_types_needed:
                    continue

                # 解析起报时间和时效
                file_init_time = datetime.strptime(init_str, "%Y%m%d%H")
                if file_init_time != init_time:
                    continue

                lead_hours = int(lead_str)

                # 如果指定了 lead_times，过滤
                if request.lead_times and lead_hours not in request.lead_times:
                    continue

                key = (init_time, lead_hours, file_type)
                files_by_key[key] = nc_file

        if not files_by_key:
            raise DiscoveryError(
                f"No Fengqing files found in {self.root_dir} for requested init_times and variables",
                source_id=request.source_id,
            )

        return DataIndex(
            source_id=request.source_id,
            available=[files_by_key],
        )

    def _get_file_types_for_variables(self, variables: List[str]) -> set:
        """确定变量所需的文件类型"""
        plevels_vars = {"z500", "z", "gh", "t", "u", "v", "q"}
        surface_vars = {"t2m", "msl", "u10", "v10", "tp"}

        file_types = set()
        for var in variables:
            var_lower = var.lower()
            if var_lower in plevels_vars:
                file_types.add("PLEVELS")
            if var_lower in surface_vars:
                file_types.add("SURFACE")

        return file_types


class FengqingReader(Reader):
    """
    Fengqing 输出 Reader

    读取 NetCDF 文件，处理集合维度，转换单位。

    变量映射：
    - Z500 -> z500（单位转换：m²/s² / 9.80665 -> m）
    - TP -> tp
    - T2M -> t2m
    - 等
    """

    # 标准重力加速度
    G = 9.80665

    def __init__(self, source_id: str = "fengqing", version: str = "1.0.0"):
        super().__init__(source_id=source_id, version=version)

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        """
        读取 Fengqing 数据

        Args:
            request: 数据请求
            index: 文件索引

        Returns:
            DataBundle with dims (init_time, lead_time, member, lat, lon)
        """
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)

        files_by_key = index.available[0]

        # 按 (init_time, lead_time) 组织数据
        datasets_by_init_lead = {}
        input_files = []

        for (init_time, lead_hours, file_type), fpath in files_by_key.items():
            try:
                with xr.open_dataset(fpath) as source:
                    ds = source.load()

                # 提取请求的变量并转换单位
                ds = self._process_variables(ds, request.variables, file_type)

                # 确保 member 维度存在
                if "member" not in ds.dims:
                    ds = ds.expand_dims("member")

                # 添加坐标
                key = (init_time, lead_hours)
                datasets_by_init_lead[key] = ds
                input_files.append(str(fpath))

            except DecodeError:
                raise
            except Exception as exc:
                raise DecodeError(
                    f"Failed to read {fpath}: {exc}",
                    source_id=request.source_id,
                    path=str(fpath),
                    cause=exc,
                ) from exc

        if not datasets_by_init_lead:
            raise DecodeError(
                "No valid data read from files",
                source_id=request.source_id,
            )

        # 组织为 (init_time, lead_time, member, lat, lon)
        combined = self._combine_datasets(datasets_by_init_lead)

        # 构建语义元数据
        units = {var: self._get_unit(var) for var in combined.data_vars}
        semantic = SemanticMetadata(
            units=units,
            temporal_kind=TemporalKind.INSTANTANEOUS,  # 瞬时场
            grid_type="regular_latlon",
            member_count=combined.sizes.get("member", 1),
        )

        # 标准变量映射
        standard_vars = {var: var for var in combined.data_vars}

        return DataBundle(
            payload=combined,
            kind=DataKind.GRIDDED_FORECAST,
            source_id=request.source_id,
            standard_vars=standard_vars,
            semantic=semantic,
            provenance=Provenance(
                input_files=input_files,
                reader_id=self.source_id,
                reader_version=self.version,
            ),
        )

    def _process_variables(self, ds: xr.Dataset, requested_vars: List[str], file_type: str) -> xr.Dataset:
        """处理变量：重命名、单位转换、选择"""
        # 变量映射（文件变量名 -> 标准名）
        var_mapping = {
            "Z500": "z500",
            "TP": "tp",
            "T2M": "t2m",
            "MSL": "msl",
            "U10": "u10",
            "V10": "v10",
        }

        # 重命名
        rename_dict = {}
        for file_var in ds.data_vars:
            std_var = var_mapping.get(file_var, file_var.lower())
            if std_var != file_var:
                rename_dict[file_var] = std_var

        if rename_dict:
            ds = ds.rename(rename_dict)

        # 单位转换
        if "z500" in ds.data_vars:
            # Z500: m²/s² -> m
            ds["z500"] = ds["z500"] / self.G
            ds["z500"].attrs["units"] = "m"
            ds["z500"].attrs["original_units"] = "m^2/s^2"

        # 选择请求的变量
        available_vars = [v for v in requested_vars if v in ds.data_vars]
        if not available_vars:
            raise DecodeError(
                f"None of requested variables {requested_vars} found in dataset. "
                f"Available: {list(ds.data_vars)}"
            )

        ds = ds[available_vars]

        return ds

    def _combine_datasets(self, datasets_by_init_lead: Dict) -> xr.Dataset:
        """组合数据为 (init_time, lead_time, member, lat, lon)"""
        # 按 init_time 分组
        by_init = {}
        for (init_time, lead_hours), ds in datasets_by_init_lead.items():
            if init_time not in by_init:
                by_init[init_time] = []
            by_init[init_time].append((lead_hours, ds))

        # 拼接每个 init_time 的 lead_time
        init_datasets = []
        for init_time in sorted(by_init.keys()):
            lead_data = sorted(by_init[init_time], key=lambda x: x[0])
            lead_datasets = [ds.expand_dims(lead_time=[lead_hours]) for lead_hours, ds in lead_data]
            init_ds = xr.concat(lead_datasets, dim="lead_time")
            init_ds = init_ds.expand_dims(init_time=[init_time])
            init_datasets.append(init_ds)

        # 拼接所有 init_time
        combined = xr.concat(init_datasets, dim="init_time")

        return combined

    def _get_unit(self, var: str) -> str:
        """获取变量单位"""
        unit_map = {
            "z500": "m",
            "tp": "mm",
            "t2m": "K",
            "msl": "Pa",
            "u10": "m/s",
            "v10": "m/s",
        }
        return unit_map.get(var, "unknown")

    @classmethod
    def with_catalog(cls, root_dir: Path, source_id: str = "fengqing"):
        """
        便捷构造器：自动创建配套的 Catalog

        Returns:
            (reader, catalog) tuple
        """
        reader = cls(source_id=source_id)
        catalog = FengqingCatalog(root_dir=root_dir)
        return reader, catalog
