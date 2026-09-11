"""
标准物理量、单位与时间语义

按照 README 第 4 节定义的数据契约，明确物理量标识、单位和时间语义。
"""

from enum import Enum


class TemporalKind(Enum):
    """时间语义"""

    INSTANTANEOUS = "instantaneous"  # 瞬时量
    INTERVAL_ACCUMULATION = "interval_accumulation"  # 指定区间累计量
    SINCE_INIT_ACCUMULATION = "since_init_accumulation"  # 从起报累计
    INTERVAL_MEAN = "interval_mean"  # 指定区间平均量


class DataKind(Enum):
    """数据类别"""

    GRIDDED_FORECAST = "gridded_forecast"  # 网格预报
    STATION_FORECAST = "station_forecast"  # 站点预报
    GRIDDED_OBSERVATION = "gridded_observation"  # 网格观测
    STATION_OBSERVATION = "station_observation"  # 站点观测
    REFERENCE = "reference"  # 气候态、投影基底等
    DERIVED = "derived"  # 派生产品（如集合平均）

