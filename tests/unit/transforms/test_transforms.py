"""
测试 Transform 模块

验证插值和时间累积功能
"""

import pytest
import numpy as np
import xarray as xr

from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator
from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator


class TestGridToStationInterpolator:
    """测试网格插值到站点"""

    def test_bilinear_interpolation_simple(self):
        """测试简单双线性插值"""
        # 创建 5×5 网格
        lats = np.array([40.0, 40.5, 41.0, 41.5, 42.0])
        lons = np.array([110.0, 110.5, 111.0, 111.5, 112.0])
        data = np.arange(25, dtype=float).reshape(5, 5)

        grid = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )

        # 站点位置（网格中心）
        station_lats = np.array([41.0, 41.5])
        station_lons = np.array([111.0, 111.5])

        interp = GridToStationInterpolator(method="bilinear")
        result = interp.transform(grid, station_lats, station_lons)

        assert result.shape == (2,)
        assert "station" in result.dims
        # 网格点位置应该精确匹配
        assert result.values[0] == pytest.approx(data[2, 2])
        assert result.values[1] == pytest.approx(data[3, 3])

    def test_nearest_interpolation(self):
        """测试最近邻插值"""
        lats = np.linspace(-10, 10, 5)
        lons = np.linspace(100, 120, 6)
        data = np.random.randn(5, 6)

        grid = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )

        station_lats = np.array([0.1, -9.9])
        station_lons = np.array([110.1, 100.1])

        interp = GridToStationInterpolator(method="nearest")
        result = interp.transform(grid, station_lats, station_lons)

        assert result.shape == (2,)
        # 最近邻应该匹配到最近的网格点
        assert np.isfinite(result.values).all()

    def test_with_extra_dimensions(self):
        """测试带额外维度的插值（如时间）"""
        n_time = 3
        lats = np.linspace(-10, 10, 10)
        lons = np.linspace(100, 120, 15)
        data = np.random.randn(n_time, 10, 15)

        grid = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(n_time),
                "lat": lats,
                "lon": lons,
            },
        )

        station_lats = np.array([0.0, 5.0])
        station_lons = np.array([110.0, 115.0])

        interp = GridToStationInterpolator(method="bilinear")
        result = interp.transform(grid, station_lats, station_lons)

        assert result.shape == (n_time, 2)
        assert result.dims == ("time", "station")
        assert "station_lat" in result.coords
        assert "station_lon" in result.coords

    def test_lon_wrapping_0_360(self):
        """测试经度 0-360 范围"""
        lats = np.array([0.0, 10.0])
        lons = np.array([0.0, 90.0, 180.0, 270.0, 359.0])
        data = np.ones((2, 5))

        grid = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )

        # 站点经度用 -180-180 格式
        station_lats = np.array([5.0])
        station_lons = np.array([-90.0])  # 应该映射到 270.0

        interp = GridToStationInterpolator(method="nearest")
        result = interp.transform(grid, station_lats, station_lons)

        assert np.isfinite(result.values[0])


class TestTimeWindowAccumulator:
    """测试时间窗口累积"""

    def test_basic_accumulation(self):
        """测试基本累积"""
        # 创建逐 6h 数据：[6h, 12h, 18h, 24h, 30h, 36h]
        lead_times = np.array([6, 12, 18, 24, 30, 36], dtype=float)
        data = np.ones(6)  # 每个时效降水都是 1

        da = xr.DataArray(
            data,
            dims=["lead_time"],
            coords={"lead_time": lead_times},
        )

        # 24h 累积
        accum = TimeWindowAccumulator(window_hours=24.0, time_dim="lead_time")
        result = accum.transform(da)

        # 期望：前 4 个时效累积（6-24h）
        assert result.dims == ("lead_time",)
        assert len(result) == 3  # 24h, 30h, 36h 三个完整窗口

        # 检查累积值
        assert result.sel(lead_time=24).values == pytest.approx(4.0)  # 前 4 个时效
        assert result.sel(lead_time=30).values == pytest.approx(4.0)  # 12-30h
        assert result.sel(lead_time=36).values == pytest.approx(4.0)  # 18-36h

    def test_6h_accumulation(self):
        """测试 6h 累积"""
        lead_times = np.array([6, 12, 18, 24], dtype=float)
        data = np.array([1.0, 2.0, 3.0, 4.0])

        da = xr.DataArray(
            data,
            dims=["lead_time"],
            coords={"lead_time": lead_times},
        )

        accum = TimeWindowAccumulator(window_hours=6.0, time_dim="lead_time")
        result = accum.transform(da)

        # 6h 累积，每个时效就是它自己
        assert len(result) == 4
        assert result.sel(lead_time=6).values == pytest.approx(1.0)
        assert result.sel(lead_time=12).values == pytest.approx(2.0)

    def test_with_spatial_dimensions(self):
        """测试带空间维度的累积"""
        n_time = 6
        n_lat = 5
        n_lon = 5
        lead_times = np.arange(1, n_time + 1) * 6.0  # [6, 12, 18, 24, 30, 36]
        data = np.random.rand(n_time, n_lat, n_lon)

        da = xr.DataArray(
            data,
            dims=["lead_time", "lat", "lon"],
            coords={
                "lead_time": lead_times,
                "lat": np.arange(n_lat),
                "lon": np.arange(n_lon),
            },
        )

        accum = TimeWindowAccumulator(window_hours=12.0, time_dim="lead_time")
        result = accum.transform(da)

        # 12h 累积 = 2 个时效
        assert result.dims == ("lead_time", "lat", "lon")
        assert len(result.lead_time) == 5  # 12, 18, 24, 30, 36

        # 验证第一个窗口（6+12h）
        expected_first = data[0] + data[1]
        assert np.allclose(result.sel(lead_time=12).values, expected_first)

    def test_insufficient_steps_fails(self):
        """测试时效不足应该失败"""
        lead_times = np.array([6, 12], dtype=float)
        data = np.ones(2)

        da = xr.DataArray(
            data,
            dims=["lead_time"],
            coords={"lead_time": lead_times},
        )

        # 24h 累积需要 4 个 6h 时效，但只有 2 个
        accum = TimeWindowAccumulator(window_hours=24.0, time_dim="lead_time")

        with pytest.raises(ValueError, match="No complete windows"):
            accum.transform(da)

    def test_non_uniform_steps_fails(self):
        """测试非均匀时间步长应该失败"""
        lead_times = np.array([6, 12, 19], dtype=float)  # 不均匀
        data = np.ones(3)

        da = xr.DataArray(
            data,
            dims=["lead_time"],
            coords={"lead_time": lead_times},
        )

        accum = TimeWindowAccumulator(window_hours=6.0, time_dim="lead_time")

        with pytest.raises(ValueError, match="uniform"):
            accum.transform(da)
