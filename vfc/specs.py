# -*- coding: utf-8 -*-
"""要素注册表：每个要素一条 VariableSpec，新增要素 = 加一行注册，计算代码不动。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VariableSpec:
    name: str
    deaccumulate: bool = False     # 是否沿时效差分（ERA5 累积量，如 tp）
    unit: str = ""                 # 目标单位说明（仅记录，不做换算）
    desc: str = ""                 # 中文说明
    derived: tuple = ()            # 派生要素分量（如风速 = √(u²+v²)）


VAR_SPECS = {
    "z500": VariableSpec("z500", unit="m2 s-2", desc="500 hPa 位势"),
    "t2m": VariableSpec("t2m", unit="K", desc="2 米温度"),
    "msl": VariableSpec("msl", unit="Pa", desc="海平面气压"),
    "tp": VariableSpec("tp", deaccumulate=True,
                       unit="mm（文件未标注，按量级推断；差分后为逐时段量）",
                       desc="总降水（差分后为逐时段降水量）"),
    "u10m": VariableSpec("u10m", unit="m s-1", desc="10 米纬向风"),
    "v10m": VariableSpec("v10m", unit="m s-1", desc="10 米经向风"),
    "u200": VariableSpec("u200", unit="m s-1", desc="200 hPa 纬向风"),
    "v200": VariableSpec("v200", unit="m s-1", desc="200 hPa 经向风"),
    "t850": VariableSpec("t850", unit="K", desc="850 hPa 温度"),
    "u850": VariableSpec("u850", unit="m s-1", desc="850 hPa 纬向风"),
    "v850": VariableSpec("v850", unit="m s-1", desc="850 hPa 经向风"),
    "q700": VariableSpec("q700", unit="kg kg-1", desc="700 hPa 比湿"),
    "q2m": VariableSpec("q2m", unit="kg kg-1", desc="2 米比湿"),
    "ws850": VariableSpec("ws850", unit="m s-1",
                          desc="850 hPa 风速（=√(u850²+v850²)）",
                          derived=("u850", "v850")),
    "ws10m": VariableSpec("ws10m", unit="m s-1",
                          desc="10 m 风速（=√(u10m²+v10m²)）",
                          derived=("u10m", "v10m")),
    "ws200": VariableSpec("ws200", unit="m s-1",
                          desc="200 hPa 风速（=√(u200²+v200²)）",
                          derived=("u200", "v200")),
}


def get_spec(name: str) -> VariableSpec:
    """注册表里没有的要素给个无预处理默认档（照样能算，元数据里注明）。"""
    return VAR_SPECS.get(name, VariableSpec(name))


def apply_transform(spec: VariableSpec, arr, lead_hours, deacc: bool = True):
    """按要素规格做预处理，返回 (arr, lead_hours)。

    tp 类累积量：沿时效差分成逐时段累积量，首个时效丢弃（差分无定义），
    对标 FDP 惯例；obs/pred 同样处理，口径一致。
    """
    arr = np.asarray(arr)
    lead_hours = np.asarray(lead_hours, dtype="f8")
    if spec.deaccumulate and deacc:
        return np.diff(arr, axis=0), lead_hours[1:]
    return arr, lead_hours
