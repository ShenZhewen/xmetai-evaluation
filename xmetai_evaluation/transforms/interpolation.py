"""
网格插值到站点

支持方法：
- bilinear: 双线性插值（默认）
- nearest: 最近邻插值
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from typing import Literal


class GridToStationInterpolator:
    """
    网格场插值到站点

    输入：网格数据 (lat, lon) 或 (..., lat, lon)
    输出：站点数据 (..., station)
    """

    def __init__(self, method: Literal["bilinear", "nearest"] = "bilinear"):
        """
        Args:
            method: 插值方法
                - bilinear: 双线性插值
                - nearest: 最近邻插值
        """
        if method not in ["bilinear", "nearest"]:
            raise ValueError(f"Unknown interpolation method: {method}")
        self.method = method

    def transform(
        self,
        grid_data: xr.DataArray,
        station_lats: np.ndarray,
        station_lons: np.ndarray,
    ) -> xr.DataArray:
        """
        插值网格场到站点

        Args:
            grid_data: 网格数据，必须包含 'lat' 和 'lon' 维度
            station_lats: 站点纬度数组 (n_stations,)
            station_lons: 站点经度数组 (n_stations,)

        Returns:
            站点数据，shape = (..., n_stations)
        """
        # 验证输入
        if "lat" not in grid_data.dims or "lon" not in grid_data.dims:
            raise ValueError("grid_data must have 'lat' and 'lon' dimensions")

        # 获取网格坐标
        grid_lats = grid_data.coords["lat"].values
        grid_lons = grid_data.coords["lon"].values

        # 确保网格坐标是单调的
        if not (np.all(np.diff(grid_lats) > 0) or np.all(np.diff(grid_lats) < 0)):
            raise ValueError("Grid latitudes must be monotonic")
        if not (np.all(np.diff(grid_lons) > 0) or np.all(np.diff(grid_lons) < 0)):
            raise ValueError("Grid longitudes must be monotonic")

        # 处理经度范围（0-360 vs -180-180）
        station_lons_adjusted = station_lons.copy()
        if grid_lons.min() >= 0 and grid_lons.max() <= 360:
            # 网格是 0-360
            station_lons_adjusted[station_lons_adjusted < 0] += 360
        elif grid_lons.min() >= -180 and grid_lons.max() <= 180:
            # 网格是 -180-180
            station_lons_adjusted[station_lons_adjusted > 180] -= 360

        # 移动 lat/lon 到最后两个维度
        dim_order = [d for d in grid_data.dims if d not in ["lat", "lon"]] + ["lat", "lon"]
        grid_data_transposed = grid_data.transpose(*dim_order)

        # 获取数据数组
        data = grid_data_transposed.values

        # 方法映射
        method_map = {"bilinear": "linear", "nearest": "nearest"}

        # 站点坐标
        station_points = np.column_stack([station_lats, station_lons_adjusted])

        # 插值
        # 需要将前面的维度展平
        original_shape = data.shape[:-2]  # 除了 lat/lon 的维度
        n_stations = len(station_lats)

        if len(original_shape) == 0:
            # 只有 lat/lon 维度
            interpolator = RegularGridInterpolator(
                (grid_lats, grid_lons),
                data,
                method=method_map[self.method],
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_values = interpolator(station_points)
        else:
            # 有其他维度，需要逐个处理
            n_extra = int(np.prod(original_shape))
            data_flat = data.reshape(n_extra, data.shape[-2], data.shape[-1])

            interp_values = np.empty((n_extra, n_stations), dtype=data.dtype)
            for i in range(n_extra):
                interp_i = RegularGridInterpolator(
                    (grid_lats, grid_lons),
                    data_flat[i],
                    method=method_map[self.method],
                    bounds_error=False,
                    fill_value=np.nan,
                )
                interp_values[i] = interp_i(station_points)

            # 恢复原始维度形状
            interp_values = interp_values.reshape(*original_shape, n_stations)

        # 构建输出 DataArray
        # 保留原始维度（除了 lat/lon），添加 station 维度
        out_dims = [d for d in grid_data.dims if d not in ["lat", "lon"]] + ["station"]
        out_coords = {k: v for k, v in grid_data.coords.items() if k not in ["lat", "lon"]}
        out_coords["station"] = np.arange(n_stations)
        out_coords["station_lat"] = ("station", station_lats)
        out_coords["station_lon"] = ("station", station_lons_adjusted)

        result = xr.DataArray(
            interp_values,
            dims=out_dims,
            coords=out_coords,
            attrs=grid_data.attrs,
        )

        return result
