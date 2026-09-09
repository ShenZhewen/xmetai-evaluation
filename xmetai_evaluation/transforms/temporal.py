"""
时间累积 Transform

支持滑动窗口累积（如 6h/24h 累积降水）
"""

import numpy as np
import xarray as xr
from typing import Literal


class TimeWindowAccumulator:
    """
    时间窗口累积

    将逐时效数据累积为固定窗口（如 6h/24h）。

    示例：
        输入：逐 6h 时效降水 [6h, 12h, 18h, 24h, ...]
        窗口：24h
        输出：[0-24h累积, 6-30h累积, 12-36h累积, ...]
    """

    def __init__(self, window_hours: float, time_dim: str = "lead_time"):
        """
        Args:
            window_hours: 累积窗口（小时）
            time_dim: 时间维度名称，默认 'lead_time'
        """
        self.window_hours = window_hours
        self.time_dim = time_dim

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """
        累积时间窗口

        Args:
            data: 输入数据，必须包含时间维度（lead_time 或 time）

        Returns:
            累积后的数据，保留原始时间坐标但值为累积值
        """
        if self.time_dim not in data.dims:
            raise ValueError(f"Data must have '{self.time_dim}' dimension")

        # 获取时间坐标（单位：小时）
        time_coord = data.coords[self.time_dim].values

        # 计算步长
        if len(time_coord) < 2:
            raise ValueError("Need at least 2 time steps for accumulation")

        time_diffs = np.diff(time_coord)
        if not np.allclose(time_diffs, time_diffs[0], rtol=1e-6):
            raise ValueError("Time steps must be uniform")

        step_hours = float(time_diffs[0])

        # 计算窗口包含多少个时间步
        n_steps_per_window = int(np.round(self.window_hours / step_hours))

        if n_steps_per_window < 1:
            raise ValueError(f"Window ({self.window_hours}h) smaller than step ({step_hours}h)")

        # 使用滑动窗口累积
        # 对于每个时刻 t，累积 [t-window+step : t] 的数据
        n_times = len(time_coord)
        accumulated = []

        for i in range(n_times):
            # 累积窗口：从 max(0, i-n_steps_per_window+1) 到 i（包含）
            start_idx = max(0, i - n_steps_per_window + 1)
            end_idx = i + 1  # 不包含，所以是 i+1

            # 如果窗口不完整（开头几个时刻），跳过
            if end_idx - start_idx < n_steps_per_window:
                continue

            # 累积该窗口
            window_slice = {self.time_dim: slice(start_idx, end_idx)}
            window_sum = data.isel(window_slice).sum(dim=self.time_dim, skipna=False)

            # 添加时间坐标（累积窗口的结束时刻）
            window_sum = window_sum.expand_dims({self.time_dim: [time_coord[i]]})
            accumulated.append(window_sum)

        if not accumulated:
            raise ValueError(
                f"No complete windows: need at least {n_steps_per_window} time steps "
                f"for {self.window_hours}h window with {step_hours}h step"
            )

        # 拼接所有累积结果
        result = xr.concat(accumulated, dim=self.time_dim)

        # 添加属性标记这是累积值
        result.attrs = data.attrs.copy()
        result.attrs["accumulation_window_hours"] = self.window_hours
        result.attrs["accumulation_method"] = "sliding_window"

        return result
