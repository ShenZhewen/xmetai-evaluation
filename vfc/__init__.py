# -*- coding: utf-8 -*-
"""vfc —— Verification Core：气象预报检验核心库（三部分共享，合成一份即可）。

按人分区（方便三份代码合并，互不覆盖文件）：
  连续量(regression)：入口 run_rmse.py；编排 vfc/regr_pair.py + vfc/regr_ens.py
                      + vfc/regr_summary.py；指标 vfc/metrics/（按指标拆文件）。
  降水(TS)：         run_ts.py + vfc/ts.py + vfc/station.py。
  台风(路径/强度)：   run_tc.py + vfc/typhoon.py。

三层结构（连续量部分）：
  io 层    vfc/io_nc.py, ensemble_io.py, pangu_io.py, aifs_io.py,
           zarr_target.py, climo.py（读文件 + 网格对齐/单位归一，三方共用）
  指标层   vfc/metrics/（纯 numpy，不碰文件；按指标拆文件）
  编排/入口 run_rmse.py -> vfc/regr_pair.py / regr_ens.py -> metrics + io

共享注册表 vfc/specs.py：三方都只增行、不删键。
"""
from .io_nc import FieldFile, check_pair, DataFileError
from .metrics import (RMSEAccumulator, AnomalyCorrelationAccumulator,
                      ActivityAccumulator, lat_weights, rmse_by_lead,
                      crps_by_lead, brier_by_lead,
                      zonal_spectrum, wavenumber_axis, wavelength_km)
from .specs import VAR_SPECS, VariableSpec, get_spec, apply_transform
from .climo import DailyClimatology
from .regr_pair import verify_pair

__all__ = [
    "FieldFile", "check_pair", "DataFileError",
    "RMSEAccumulator", "AnomalyCorrelationAccumulator", "ActivityAccumulator",
    "lat_weights", "rmse_by_lead", "crps_by_lead", "brier_by_lead",
    "VAR_SPECS", "VariableSpec", "get_spec", "apply_transform",
    "DailyClimatology", "zonal_spectrum", "wavenumber_axis", "wavelength_km",
    "verify_pair",
]
