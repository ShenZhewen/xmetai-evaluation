# -*- coding: utf-8 -*-
"""连续量(regression) 指标：RMSE（逐时效纬度加权，含流式累积器）。"""
from __future__ import annotations
import numpy as np
from ._weights import lat_weights

def rmse_by_lead(pred, obs, lat, weighted: bool = True) -> np.ndarray:
    """一次性计算逐时效纬度加权 RMSE（参考实现/小数据用；大文件用 RMSEAccumulator）。

    RMSE(t) = sqrt( sum_ij w_i (f-o)^2 / sum_ij w_i )，NaN 格点自动跳过。"""
    pred = np.asarray(pred, dtype="f8")
    obs = np.asarray(obs, dtype="f8")
    w = lat_weights(lat, weighted)[:, None]
    d2 = (pred - obs) ** 2
    ok = np.isfinite(d2)
    se = np.where(ok, d2, 0.0) * w
    ew = ok.astype("f8") * w
    return np.sqrt(se.sum(axis=(1, 2)) / ew.sum(axis=(1, 2)))

class RMSEAccumulator(object):
    """流式 RMSE 累积器：分块 update、最后 finalize。"""
    def __init__(self, n_lead, lat, weighted=True):
        self.n_lead = int(n_lead)
        self._w = lat_weights(lat, weighted)[:, None]
        self._se = np.zeros(self.n_lead, dtype="f8")
        self._ew = np.zeros(self.n_lead, dtype="f8")
    def update(self, i0, pred, obs):
        pred = np.asarray(pred, dtype="f8")
        obs = np.asarray(obs, dtype="f8")
        t = pred.shape[0]
        if i0 + t > self.n_lead:
            raise ValueError("块越界: i0+t=%d > n_lead=%d" % (i0 + t, self.n_lead))
        d2 = (pred - obs) ** 2
        ok = np.isfinite(d2)
        self._se[i0:i0 + t] += (np.where(ok, d2, 0.0) * self._w).sum(axis=(1, 2))
        self._ew[i0:i0 + t] += (ok.astype("f8") * self._w).sum(axis=(1, 2))
    def finalize(self) -> np.ndarray:
        if np.any(self._ew <= 0):
            raise ValueError("存在无有效格点的时效步，无法计算 RMSE")
        return np.sqrt(self._se / self._ew)
