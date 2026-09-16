# -*- coding: utf-8 -*-
"""连续量(regression) 指标：CRPS（集合，逐时效纬度加权；m=1 退化为 MAE）。"""
from __future__ import annotations
import numpy as np
from ._weights import lat_weights

def crps_by_lead(pred, obs, lat, weighted: bool = True) -> np.ndarray:
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    if pred.ndim == 3:
        pred = pred[None]
    T = obs.shape[0]
    n = pred.shape[0]
    w = lat_weights(lat, weighted)[:, None]
    out = np.empty(T, dtype="f8")
    for t in range(T):
        p = pred[:, t].astype("f8")
        o = obs[t].astype("f8")
        ok = np.isfinite(p) & np.isfinite(o[None])
        m = ok.sum(axis=0).astype("f8")
        d = np.where(ok, np.abs(p - o[None]), 0.0)
        ma = d.sum(axis=0) / np.maximum(m, 1.0)
        ps = np.sort(np.where(ok, p, np.nan), axis=0)
        kk = np.arange(n, dtype="f8") + 1.0
        coeff = 2.0 * kk[:, None, None] - m[None] - 1.0
        pair = np.nansum(ps * coeff, axis=0) / np.maximum(m * m, 1.0)
        cr = ma - pair
        okg = (m > 0) & np.isfinite(cr)
        sw = (okg.astype("f8") * w).sum()
        out[t] = (np.where(okg, cr, 0.0) * w).sum() / max(float(sw), 1e-12)
    return out
