"""
标准物理量、单位与时间语义

按照 README 第 4 节定义的数据契约，明确物理量标识、单位和时间语义。
"""

from enum import Enum


class StandardVariable(Enum):
    """标准物理量"""

    # 温度
    T2M = "t2m"  # 2米温度
    T = "t"  # 气温

    # 降水
    TP = "tp"  # 总降水
    PRECIP = "precip"  # 降水（站点）

    # 风
    U = "u"  # 纬向风
    V = "v"  # 经向风
    U10 = "u10"  # 10米纬向风
    V10 = "v10"  # 10米经向风
    U200 = "u200"  # 200hPa纬向风
    U850 = "u850"  # 850hPa纬向风

    # 位势与气压
    Z = "z"  # 位势
    GH = "gh"  # 位势高度
    Z500 = "z500"  # 500hPa位势高度
    MSL = "msl"  # 海平面气压

    # 辐射
    SSR = "ssr"  # 地表净短波辐射
    SSRD = "ssrd"  # 地表向下短波辐射
    FDIR = "fdir"  # 地表直接短波辐射
    TTR = "ttr"  # 顶层净辐射
    OLR = "olr"  # 向外长波辐射

    # 其他
    Q = "q"  # 比湿
    SP = "sp"  # 地表气压


class TemporalKind(Enum):
    """时间语义"""

    INSTANTANEOUS = "instantaneous"  # 瞬时量
    INTERVAL_ACCUMULATION = "interval_accumulation"  # 指定区间累计量
    SINCE_INIT_ACCUMULATION = "since_init_accumulation"  # 从起报累计
    INTERVAL_MEAN = "interval_mean"  # 指定区间平均量


class ForecastKind(Enum):
    """预报产品类型"""

    DETERMINISTIC = "deterministic"  # 确定性预报
    ENSEMBLE = "ensemble"  # 集合预报
    PROBABILITY = "probability"  # 概率产品
    INDEX = "index"  # 指数产品（如MJO RMM）


class DataKind(Enum):
    """数据类别"""

    GRIDDED_FORECAST = "gridded_forecast"  # 网格预报
    STATION_FORECAST = "station_forecast"  # 站点预报
    GRIDDED_OBSERVATION = "gridded_observation"  # 网格观测
    STATION_OBSERVATION = "station_observation"  # 站点观测
    REFERENCE = "reference"  # 气候态、投影基底等
    DERIVED = "derived"  # 派生产品（如集合平均）

