"""
测试 NetCDF Reader

验证 README Change 5 要求：
- Catalog 发现文件
- Reader 读取数据
- 返回正确的 DataBundle
- 错误处理
"""

import pytest
from pathlib import Path
import xarray as xr
import numpy as np

from xmetai_evaluation.io.base import Reader, DataCatalog
from xmetai_evaluation.io.netcdf_reader import SimpleNetCDFCatalog, SimpleNetCDFReader
from xmetai_evaluation.core.contracts import (
    DataRequest,
    DataIndex,
    DataBundle,
    DataKind,
)
from xmetai_evaluation.core.errors import DiscoveryError, DecodeError, ContractError


@pytest.fixture
def sample_netcdf(tmp_path):
    """创建测试用 NetCDF 文件"""
    # 创建简单数据集
    lat = np.linspace(-10, 10, 5)
    lon = np.linspace(100, 120, 5)
    time = np.array([0, 6, 12, 18])

    data = np.random.randn(4, 5, 5) * 10 + 20

    ds = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], data),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
    )

    # 添加属性
    ds["t2m"].attrs["units"] = "K"
    ds["t2m"].attrs["long_name"] = "2m temperature"

    # 保存到文件
    nc_path = tmp_path / "test_data.nc"
    ds.to_netcdf(nc_path)

    return nc_path


@pytest.fixture
def sample_netcdf_multi_var(tmp_path):
    """创建包含多个变量的 NetCDF 文件"""
    lat = np.linspace(-10, 10, 5)
    lon = np.linspace(100, 120, 5)
    time = np.array([0, 6, 12])

    ds = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], np.random.randn(3, 5, 5) * 10 + 280),
            "tp": (["time", "lat", "lon"], np.random.rand(3, 5, 5) * 50),
            "msl": (["time", "lat", "lon"], np.random.randn(3, 5, 5) * 100 + 101325),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
    )

    ds["t2m"].attrs["units"] = "K"
    ds["tp"].attrs["units"] = "mm"
    ds["msl"].attrs["units"] = "Pa"

    nc_path = tmp_path / "multi_var.nc"
    ds.to_netcdf(nc_path)

    return nc_path


class TestSimpleNetCDFCatalog:
    """测试 NetCDF Catalog"""

    def test_catalog_discover_files(self, tmp_path, sample_netcdf):
        """测试发现文件"""
        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)
        request = DataRequest(source_id="test", variables=["t2m"])

        index = catalog.discover(request)

        assert index.source_id == "test"
        assert len(index.available) == 1
        assert index.available[0] == sample_netcdf

    def test_catalog_no_files(self, tmp_path):
        """测试没有文件的情况"""
        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)
        request = DataRequest(source_id="test", variables=["t2m"])

        with pytest.raises(DiscoveryError, match="No NetCDF files found"):
            catalog.discover(request)

    def test_catalog_directory_not_exists(self, tmp_path):
        """测试目录不存在"""
        non_existent = tmp_path / "does_not_exist"
        catalog = SimpleNetCDFCatalog(root_dir=non_existent)
        request = DataRequest(source_id="test", variables=["t2m"])

        with pytest.raises(DiscoveryError, match="does not exist"):
            catalog.discover(request)

    def test_catalog_multiple_files(self, tmp_path):
        """测试发现多个文件"""
        # 创建多个 NC 文件
        for i in range(3):
            ds = xr.Dataset({"data": (["x"], [i])})
            ds.to_netcdf(tmp_path / f"file_{i}.nc")

        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)
        request = DataRequest(source_id="test", variables=["data"])

        index = catalog.discover(request)

        assert len(index.available) == 3


class TestSimpleNetCDFReader:
    """测试 NetCDF Reader"""

    def test_reader_basic(self, tmp_path, sample_netcdf):
        """测试基本读取"""
        reader = SimpleNetCDFReader(source_id="test_reader")
        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)
        request = DataRequest(source_id="test_reader", variables=["t2m"])

        index = catalog.discover(request)
        bundle = reader.read(request, index)

        assert isinstance(bundle, DataBundle)
        assert bundle.source_id == "test_reader"
        assert bundle.kind == DataKind.GRIDDED_FORECAST
        assert isinstance(bundle.payload, xr.Dataset)
        assert "t2m" in bundle.payload.data_vars
        assert bundle.semantic.units["t2m"] == "K"

    def test_reader_provenance(self, tmp_path, sample_netcdf):
        """测试溯源信息"""
        reader = SimpleNetCDFReader(source_id="test")
        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)
        request = DataRequest(source_id="test", variables=["t2m"])

        index = catalog.discover(request)
        bundle = reader.read(request, index)

        assert bundle.provenance.reader_id == "test"
        assert bundle.provenance.reader_version == "1.0.0"
        assert len(bundle.provenance.input_files) == 1
        assert str(sample_netcdf) in bundle.provenance.input_files[0]

    def test_reader_variable_selection(self, tmp_path, sample_netcdf_multi_var):
        """测试变量选择"""
        reader = SimpleNetCDFReader()
        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)

        # 只请求部分变量
        request = DataRequest(source_id="test", variables=["t2m", "tp"])
        index = catalog.discover(request)
        bundle = reader.read(request, index)

        assert "t2m" in bundle.payload.data_vars
        assert "tp" in bundle.payload.data_vars
        assert "msl" not in bundle.payload.data_vars

    def test_reader_all_variables(self, tmp_path, sample_netcdf_multi_var):
        """测试读取所有变量"""
        reader = SimpleNetCDFReader()
        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)

        # 不指定变量，读取所有
        request = DataRequest(source_id="test", variables=["*"])
        index = catalog.discover(request)
        bundle = reader.read(request, index)

        assert "t2m" in bundle.payload.data_vars
        assert "tp" in bundle.payload.data_vars
        assert "msl" in bundle.payload.data_vars

    def test_reader_missing_variable_fails(self, tmp_path, sample_netcdf):
        """测试请求不存在的变量应该失败"""
        reader = SimpleNetCDFReader()
        catalog = SimpleNetCDFCatalog(root_dir=tmp_path)

        request = DataRequest(source_id="test", variables=["nonexistent"])
        index = catalog.discover(request)

        with pytest.raises(ContractError, match="Variables not found"):
            reader.read(request, index)

    def test_reader_with_root_dir(self, tmp_path, sample_netcdf):
        """测试 with_root_dir 构造器"""
        reader = SimpleNetCDFReader.with_root_dir(
            root_dir=tmp_path,
            source_id="test_with_root",
        )

        request = DataRequest(source_id="test_with_root", variables=["t2m"])
        bundle = reader.load(request)

        assert bundle.source_id == "test_with_root"
        assert "t2m" in bundle.payload.data_vars

    def test_reader_load_without_root_dir_fails(self):
        """测试没有 root_dir 的 load 应该失败"""
        reader = SimpleNetCDFReader()
        request = DataRequest(source_id="test", variables=["t2m"])

        with pytest.raises(NotImplementedError, match="requires root_dir"):
            reader.load(request)

    def test_reader_empty_index_fails(self):
        """测试空索引应该失败"""
        reader = SimpleNetCDFReader()
        request = DataRequest(source_id="test", variables=["t2m"])
        empty_index = DataIndex(source_id="test", available=[], ambiguous=[])

        with pytest.raises(DecodeError, match="No files in index"):
            reader.read(request, empty_index)

    def test_reader_corrupted_file(self, tmp_path):
        """测试损坏文件"""
        # 创建非 NetCDF 文件
        bad_file = tmp_path / "bad.nc"
        bad_file.write_text("This is not a NetCDF file")

        reader = SimpleNetCDFReader()
        request = DataRequest(source_id="test", variables=["t2m"])
        index = DataIndex(source_id="test", available=[bad_file], ambiguous=[])

        with pytest.raises(DecodeError, match="Failed to read NetCDF"):
            reader.read(request, index)
