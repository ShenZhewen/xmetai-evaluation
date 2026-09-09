"""
NetCDF Reader 最小实现

按照 README Change 5 要求：
- 只读取 xarray 能直接打开的简单 NetCDF
- 不处理复杂单位转换
- 不处理多文件拼接
- 用于验证 Reader 协议和端到端流程
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

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
from xmetai_evaluation.core.errors import DiscoveryError, DecodeError, ContractError
from xmetai_evaluation.core.variables import TemporalKind


class SimpleNetCDFCatalog(DataCatalog):
    """
    简单 NetCDF 目录

    假设文件命名规则：{source_id}_{variable}_{init_time}.nc
    或单个文件包含所有数据。

    这是最小实现，Change 5 范围。
    """

    def __init__(self, root_dir: Path):
        """
        Args:
            root_dir: 数据根目录
        """
        self.root_dir = Path(root_dir)

    def discover(self, request: DataRequest) -> DataIndex:
        """
        发现文件

        Args:
            request: 数据请求

        Returns:
            DataIndex
        """
        if not self.root_dir.exists():
            raise DiscoveryError(
                f"Root directory does not exist: {self.root_dir}",
                source_id=request.source_id,
            )

        # 简单实现：查找所有 .nc 文件
        nc_files = list(self.root_dir.glob("*.nc"))

        if not nc_files:
            raise DiscoveryError(
                f"No NetCDF files found in {self.root_dir}",
                source_id=request.source_id,
            )

        # 构建索引（简化：不解析文件名）
        index = DataIndex(
            source_id=request.source_id,
            available=nc_files,
            ambiguous=[],
        )

        return index


class SimpleNetCDFReader(Reader):
    """
    简单 NetCDF Reader

    只处理 xarray 能直接打开的标准 NetCDF 文件。
    不做复杂的单位转换、坐标解析或多文件合并。

    Change 5 最小实现。
    """

    def __init__(self, source_id: str = "netcdf", version: str = "1.0.0"):
        super().__init__(source_id=source_id, version=version)

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        """
        读取 NetCDF 文件

        Args:
            request: 数据请求
            index: 文件索引

        Returns:
            DataBundle
        """
        if not index.available:
            raise DecodeError(
                "No files in index",
                source_id=request.source_id,
            )

        # 简单实现：只读取第一个文件
        file_path = index.available[0]

        try:
            # 打开文件
            ds = xr.open_dataset(file_path)

            # 提取请求的变量（如果指定）
            if request.variables and request.variables != ["*"]:
                # 检查变量是否存在
                missing_vars = [v for v in request.variables if v not in ds.data_vars]
                if missing_vars:
                    raise ContractError(
                        f"Variables not found in file: {missing_vars}",
                        source_id=request.source_id,
                        path=str(file_path),
                    )
                ds = ds[request.variables]

            # 构建语义元数据（简化）
            semantic = SemanticMetadata(
                units={var: str(ds[var].attrs.get("units", "unknown")) for var in ds.data_vars},
                temporal_kind=TemporalKind.INSTANTANEOUS,  # 默认
                grid_type="regular_latlon",  # 假设
            )

            # 构建溯源信息
            provenance = Provenance(
                input_files=[str(file_path)],
                reader_id=self.source_id,
                reader_version=self.version,
            )

            # 构建标准变量映射（简化：假设文件变量名就是标准名）
            standard_vars = {var: var for var in ds.data_vars}

            # 构建 DataBundle
            bundle = DataBundle(
                payload=ds,
                kind=DataKind.GRIDDED_FORECAST,
                source_id=request.source_id,
                standard_vars=standard_vars,
                semantic=semantic,
                provenance=provenance,
            )

            return bundle

        except Exception as e:
            if isinstance(e, (ContractError, DecodeError)):
                raise
            raise DecodeError(
                f"Failed to read NetCDF file: {e}",
                source_id=request.source_id,
                path=str(file_path),
                cause=e,
            )

    def load(self, request: DataRequest) -> DataBundle:
        """
        便捷方法：自动发现 + 读取

        Args:
            request: 数据请求（必须包含隐式或显式的根目录）

        Returns:
            DataBundle
        """
        # 简化：从请求的 region 字段获取根目录（如果有）
        # 实际应用中，根目录通常在 Reader 初始化时提供
        if hasattr(self, "root_dir"):
            catalog = SimpleNetCDFCatalog(self.root_dir)
        else:
            raise NotImplementedError(
                "SimpleNetCDFReader.load() requires root_dir to be set. "
                "Use read() with an explicit DataIndex instead."
            )

        index = catalog.discover(request)
        return self.read(request, index)

    @classmethod
    def with_root_dir(cls, root_dir: Path, source_id: str = "netcdf") -> "SimpleNetCDFReader":
        """
        创建带根目录的 Reader

        Args:
            root_dir: 数据根目录
            source_id: 数据源ID

        Returns:
            SimpleNetCDFReader
        """
        reader = cls(source_id=source_id)
        reader.root_dir = root_dir
        return reader
