# -*- coding: utf-8 -*-
"""连续量(regression) 指标层——基础：纬度（面积）权重。"""
from __future__ import annotations
import numpy as np

def lat_weights(lat, weighted: bool = True) -> np.ndarray:
    """纬度权重：weighted=True 时 cos(lat)，否则全 1。"""
    lat = np.asarray(lat, dtype="f8")
    return np.cos(np.deg2rad(lat)) if weighted else np.ones_like(lat)
