"""
CRA (CRA40/CRA-LAND) 再分析数据 Reader

数据布局：
- cra_root/YYYYMMDD/ART_ATM_GLB_0P25_6HOR_ANAL_YYYYMMDDHH.grib2（大气变量）
- cra_root/YYYYMMDD/CRA40LAND_SURFACE_YYYYMMDDHH_GLB_0P25_HOUR_V1_0_0.grib（地面变量）

支持变量：
- z500: shortName='gh', typeOfLevel='isobaricInhPa', level=500, units='gpm'
- t2m: shortName='2t', typeOfLevel='heightAboveGround', level=2, units='K'
- msl: shortName='msl', typeOfLevel='meanSea', units='Pa'
- u10: shortName='10u', typeOfLevel='heightAboveGround', level=10, units='m s**-1'
- v10: shortName='10v', typeOfLevel='heightAboveGround', level=10, units='m s**-1'
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
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


class CRACatalog(DataCatalog):
    """
    CRA 再分析数据目录

    扫描 cra_root/YYYYMMDD/ 目录，按变量分类文件（ATM/LAND）。
    """

    # 文件名正则
    ATM_PATTERN = re.compile(r"ART_ATM_GLB_0P25_6HOR_ANAL_(\d{10})\.grib2")
    SURFACE_PATTERN = re.compile(r"CRA40LAND_SURFACE_(\d{10})_GLB_0P25_HOUR_V1_0_0\.grib")

    def __init__(self, root_dir: Path):
        """
        Args:
            root_dir: 数据根目录
        """
        self.root_dir = Path(root_dir)

    def discover(self, request: DataRequest) -> DataIndex:
        """
        发现指定时间的 CRA 文件

        Args:
            request: 需要包含 init_times 或 valid_times（作为观测时间）

        Returns:
            DataIndex，available 包含 {(valid_time, file_type): file_path}
        """
        if not self.root_dir.exists():
            raise DiscoveryError(
                f"Root directory does not exist: {self.root_dir}",
                source_id=request.source_id,
            )

        # 对于观测数据，使用 init_times 作为 valid_times
        valid_times = request.init_times
        if not valid_times:
            raise DiscoveryError(
                "CRACatalog requires init_times (as valid_times) in request",
                source_id=request.source_id,
            )

        # 确定需要哪些文件类型
        file_types_needed = self._get_file_types_for_variables(request.variables)

        # 扫描文件
        files_by_key = {}
        for valid_time in valid_times:
            # 日期目录：YYYYMMDD
            date_dir = self.root_dir / valid_time.strftime("%Y%m%d")
            if not date_dir.exists():
                continue

            # 查找 ATM 文件
            if "ATM" in file_types_needed:
                for grib_file in date_dir.glob("ART_ATM_*.grib2"):
                    match = self.ATM_PATTERN.match(grib_file.name)
                    if not match:
                        continue
                    time_str = match.group(1)
                    file_time = datetime.strptime(time_str, "%Y%m%d%H")
                    if file_time == valid_time:
                        files_by_key[(valid_time, "ATM")] = grib_file

            # 查找 SURFACE 文件
            if "SURFACE" in file_types_needed:
                for grib_file in date_dir.glob("CRA40LAND_*.grib"):
                    match = self.SURFACE_PATTERN.match(grib_file.name)
                    if not match:
                        continue
                    time_str = match.group(1)
                    file_time = datetime.strptime(time_str, "%Y%m%d%H")
                    if file_time == valid_time:
                        files_by_key[(valid_time, "SURFACE")] = grib_file

        if not files_by_key:
            raise DiscoveryError(
                f"No CRA files found in {self.root_dir} for requested times and variables",
                source_id=request.source_id,
            )

        return DataIndex(
            source_id=request.source_id,
            available=[files_by_key],
        )

    def _get_file_types_for_variables(self, variables: List[str]) -> set:
        """确定变量所需的文件类型"""
        atm_vars = {"z500", "z", "gh", "t", "u", "v", "q"}
        surface_vars = {"t2m", "msl", "u10", "v10", "tp"}

        file_types = set()
        for var in variables:
            var_lower = var.lower()
            if var_lower in atm_vars:
                file_types.add("ATM")
            if var_lower in surface_vars:
                file_types.add("SURFACE")

        return file_types


class CRAReader(Reader):
    """
    CRA 再分析数据 Reader

    使用 cfgrib 按 filter_by_keys 读取指定变量。

    变量过滤规则：
    - z500: {'shortName': 'gh', 'typeOfLevel': 'isobaricInhPa', 'level': 500}
    - t2m: {'shortName': '2t', 'typeOfLevel': 'heightAboveGround', 'level': 2}
    - msl: {'shortName': 'msl', 'typeOfLevel': 'meanSea'}
    - u10: {'shortName': '10u', 'typeOfLevel': 'heightAboveGround', 'level': 10}
    - v10: {'shortName': '10v', 'typeOfLevel': 'heightAboveGround', 'level': 10}
    """

    # 变量过滤配置
    VARIABLE_FILTERS = {
        "z500": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "gh": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
        "msl": {"shortName": "msl", "typeOfLevel": "meanSea"},
        "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
        "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
    }

    def __init__(self, source_id: str = "cra", version: str = "1.0.0"):
        super().__init__(source_id=source_id, version=version)

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        """
        读取 CRA 数据

        Args:
            request: 数据请求
            index: 文件索引

        Returns:
            DataBundle with dims (valid_time, lat, lon)
        """
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)

        files_by_key = index.available[0]

        # 按 valid_time 和 variable 读取
        datasets_by_time = {}
        input_files = []

        for (valid_time, file_type), fpath in files_by_key.items():
            for var in request.variables:
                if var not in self.VARIABLE_FILTERS:
                    continue

                try:
                    # 使用 cfgrib 读取特定变量
                    filter_by_keys = self.VARIABLE_FILTERS[var]

                    # 使用 xarray + cfgrib engine
                    ds = xr.open_dataset(
                        fpath,
                        engine="cfgrib",
                        backend_kwargs={"filter_by_keys": filter_by_keys},
                    ).load()

                    # 标准化坐标名：latitude->lat, longitude->lon
                    ds = self._standardize_coordinates(ds)

                    # 重命名变量为标准名
                    source_var = list(ds.data_vars)[0]  # cfgrib 通常只返回一个变量
                    if source_var != var:
                        ds = ds.rename({source_var: var})

                    # 按 valid_time 组织
                    if valid_time not in datasets_by_time:
                        datasets_by_time[valid_time] = []
                    datasets_by_time[valid_time].append(ds)
                    input_files.append(str(fpath))

                except Exception as exc:
                    # 如果某个变量读取失败，继续尝试其他变量
                    # 但记录警告
                    import warnings
                    warnings.warn(
                        f"Failed to read {var} from {fpath}: {exc}"
                    )
                    continue

        if not datasets_by_time:
            raise DecodeError(
                f"No valid data read from CRA files for variables {request.variables}",
                source_id=request.source_id,
            )

        # 合并为 (valid_time, lat, lon)
        combined = self._combine_datasets(datasets_by_time)

        # 构建语义元数据
        units = {var: self._get_unit(var) for var in combined.data_vars}
        semantic = SemanticMetadata(
            units=units,
            temporal_kind=TemporalKind.INSTANTANEOUS,
            grid_type="regular_latlon",
        )

        # 标准变量映射
        standard_vars = {var: var for var in combined.data_vars}

        return DataBundle(
            payload=combined,
            kind=DataKind.GRIDDED_OBSERVATION,
            source_id=request.source_id,
            standard_vars=standard_vars,
            semantic=semantic,
            provenance=Provenance(
                input_files=list(set(input_files)),  # 去重
                reader_id=self.source_id,
                reader_version=self.version,
            ),
        )

    def _standardize_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """标准化坐标名"""
        rename_dict = {}
        if "latitude" in ds.coords:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds.coords:
            rename_dict["longitude"] = "lon"

        if rename_dict:
            ds = ds.rename(rename_dict)

        return ds

    def _combine_datasets(self, datasets_by_time: Dict) -> xr.Dataset:
        """合并数据为 (valid_time, lat, lon)"""
        time_datasets = []
        for valid_time in sorted(datasets_by_time.keys()):
            # 合并该时间的所有变量
            var_datasets = datasets_by_time[valid_time]
            if len(var_datasets) == 1:
                merged = var_datasets[0]
            else:
                merged = xr.merge(var_datasets)

            # 添加 valid_time 维度
            merged = merged.expand_dims(valid_time=[valid_time])
            time_datasets.append(merged)

        # 拼接所有时间
        combined = xr.concat(time_datasets, dim="valid_time")

        return combined

    def _get_unit(self, var: str) -> str:
        """获取变量单位"""
        unit_map = {
            "z500": "m",  # gpm = geopotential meters
            "gh": "m",
            "t2m": "K",
            "msl": "Pa",
            "u10": "m/s",
            "v10": "m/s",
        }
        return unit_map.get(var, "unknown")

    @classmethod
    def with_catalog(cls, root_dir: Path, source_id: str = "cra"):
        """
        便捷构造器：自动创建配套的 Catalog

        Returns:
            (reader, catalog) tuple
        """
        reader = cls(source_id=source_id)
        catalog = CRACatalog(root_dir=root_dir)
        return reader, catalog
