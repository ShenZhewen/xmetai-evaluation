# -*- coding: utf-8 -*-
"""连续量(regression) 指标层——基础：纬度（面积）权重。"""
from __future__ import annotations
import numpy as np

def lat_weights(lat, weighted: bool = True) -> np.ndarray:
    """纬度权重：weighted=True 时 cos(lat)，否则全 1。

    ``lat`` 单位必须是**度**。这里挡住超范围输入（>90，通常是传了别的
    单位或坐标）；「传弧度」数值更小挡不住，由数据入口
    ``io_nc.FieldFile._check_lat`` 按步长兜底。
    """
    lat = np.asarray(lat, dtype="f8")
    if weighted:
        if lat.size and np.nanmax(np.abs(lat)) > 90.0 + 1e-6:
            raise ValueError(
                "lat 必须是「度」且 |lat| <= 90；当前 max|lat|=%.6f"
                % np.nanmax(np.abs(lat)))
        return np.cos(np.deg2rad(lat))
    return np.ones_like(lat)
