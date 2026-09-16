# -*- coding: utf-8 -*-
"""vfc.metrics —— 连续量(regression) 指标层：纯 numpy 数组进出，不碰文件。

按指标拆文件（新增指标 = 新文件 + 在 __init__ 注册，互不干扰）：
  _weights.py  纬度（面积）权重
  rmse.py      RMSE（含流式累积器）
  acc_fa.py    ACC / 预报活跃度 FA
  crps.py      CRPS（集合）
  brier.py     Brier（集合）
  spectrum.py  纬向 FFT 功率谱（能量谱）
"""
from ._weights import lat_weights
from .rmse import rmse_by_lead, RMSEAccumulator
from .acc_fa import AnomalyCorrelationAccumulator, ActivityAccumulator
from .crps import crps_by_lead
from .brier import brier_by_lead
from .spectrum import zonal_spectrum, wavenumber_axis, wavelength_km

__all__ = ["lat_weights", "rmse_by_lead", "RMSEAccumulator",
           "AnomalyCorrelationAccumulator", "ActivityAccumulator",
           "crps_by_lead", "brier_by_lead",
           "zonal_spectrum", "wavenumber_axis", "wavelength_km"]
