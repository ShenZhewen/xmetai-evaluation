"""
标准物理量、单位与时间语义

按照 README 第 4 节定义的数据契约，明确物理量标识、单位和时间语义。
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


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


@dataclass(frozen=True)
class DimensionSpec:
    """维度规范"""

    name: str
    description: str
    required: bool = True


# 标准维度定义（按照 README 第 4.1 节）
STANDARD_DIMS = {
    "init_time": DimensionSpec("init_time", "起报时间", required=False),
    "valid_time": DimensionSpec("valid_time", "有效时间", required=False),
    "lead_time": DimensionSpec("lead_time", "预报时效（小时）", required=False),
    "member": DimensionSpec("member", "集合成员索引（0-based）", required=False),
    "lat": DimensionSpec("lat", "纬度（N to S, degrees_north）", required=False),
    "lon": DimensionSpec("lon", "经度（0-360 or -180-180, degrees_east）", required=False),
    "station": DimensionSpec("station", "站点ID或索引", required=False),
    "level": DimensionSpec("level", "垂直层次", required=False),
    "threshold": DimensionSpec("threshold", "阈值", required=False),
    "component": DimensionSpec("component", "向量分量或指数分量", required=False),
    "wavenumber": DimensionSpec("wavenumber", "波数", required=False),
    "sample": DimensionSpec("sample", "样本索引", required=False),
}


# CF标准单位示例（不完整，按需扩展）
CF_UNITS = {
    "temperature": ["K", "degC", "degree_Celsius"],
    "precipitation": ["mm", "kg m-2", "m"],
    "wind": ["m s-1", "m/s"],
    "geopotential": ["m2 s-2", "m^2 s^-2"],
    "geopotential_height": ["m"],
    "pressure": ["Pa", "hPa", "mb"],
    "radiation": ["W m-2", "J m-2"],
}


def validate_unit_compatible(value_unit: str, target_category: str) -> bool:
    """
    简单的单位兼容性检查

    实际实现应使用 cf-units 或 pint 库
    """
    if target_category not in CF_UNITS:
        return False
    return value_unit in CF_UNITS[target_category]
