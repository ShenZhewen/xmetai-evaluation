# -*- coding: utf-8 -*-
"""网格场插值到站点。

支持方法：

- ``bilinear``：双线性（xarray ``interp(method="linear")``，与参考实现一致）；
- ``nearest``：最近邻。

保留 0/360 与 -180/180 经度归位：站点经度先折算到网格经度范围内再插值，
避免格点经度为 0~360 时西经站点落到网格外而整列缺测。
"""

import numpy as np
import xarray as xr
from typing import Literal

# 方法名 -> xarray interp 方法
_METHODS = {"bilinear": "linear", "nearest": "nearest"}


class GridToStationInterpolator:
    """网格场插值到站点。

    输入：网格数据 (lat, lon) 或 (..., lat, lon)
    输出：站点数据 (..., station)
    """

    def __init__(self, method: Literal["bilinear", "nearest"] = "bilinear"):
        """
        Args:
            method: 插值方法（bilinear / nearest）
        """
        if method not in _METHODS:
            raise ValueError(f"Unknown interpolation method: {method}")
        self.method = method

    def transform(
        self,
        grid_data: xr.DataArray,
        station_lats: np.ndarray,
        station_lons: np.ndarray,
    ) -> xr.DataArray:
        """插值网格场到站点。

        Args:
            grid_data: 网格数据，必须包含 'lat' 和 'lon' 维度
            station_lats: 站点纬度数组 (n_stations,)
            station_lons: 站点经度数组 (n_stations,)

        Returns:
            站点数据，shape = (..., n_stations)；网格范围外的站点为 NaN
        """
        if "lat" not in grid_data.dims or "lon" not in grid_data.dims:
            raise ValueError("grid_data must have 'lat' and 'lon' dimensions")

        grid_lats = np.asarray(grid_data.coords["lat"].values, dtype="f8")
        grid_lons = np.asarray(grid_data.coords["lon"].values, dtype="f8")
        if not (np.all(np.diff(grid_lats) > 0) or np.all(np.diff(grid_lats) < 0)):
            raise ValueError("Grid latitudes must be monotonic")
        if not (np.all(np.diff(grid_lons) > 0) or np.all(np.diff(grid_lons) < 0)):
            raise ValueError("Grid longitudes must be monotonic")

        lats = np.asarray(station_lats, dtype="f8")
        lons = np.asarray(station_lons, dtype="f8").copy()
        if grid_lons.min() >= 0 and grid_lons.max() <= 360:
            lons[lons < 0] += 360.0
        elif grid_lons.min() >= -180 and grid_lons.max() <= 180:
            lons[lons > 180] -= 360.0

        dim_order = [d for d in grid_data.dims if d not in ("lat", "lon")] + ["lat", "lon"]
        interpolated = grid_data.transpose(*dim_order).interp(
            lat=xr.DataArray(lats, dims="station"),
            lon=xr.DataArray(lons, dims="station"),
            method=_METHODS[self.method],
        )
        interpolated = interpolated.assign_coords(
            station=np.arange(lats.size),
            station_lat=("station", lats),
            station_lon=("station", lons),
        )
        interpolated.attrs = dict(grid_data.attrs)
        return interpolated
