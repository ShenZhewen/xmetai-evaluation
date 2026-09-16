# -*- coding: utf-8 -*-
"""连续量(regression) 指标：Brier（集合；事件 = 场值 >= threshold）。"""
from __future__ import annotations
import numpy as np
from ._weights import lat_weights

def brier_by_lead(pred, obs, lat, threshold: float,
                  weighted: bool = True) -> np.ndarray:
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    if pred.ndim == 3:
        pred = pred[None]
    T = obs.shape[0]
    w = lat_weights(lat, weighted)[:, None]
    out = np.empty(T, dtype="f8")
    for t in range(T):
        p = pred[:, t]
        o = obs[t]
        ok = np.isfinite(p) & np.isfinite(o[None])
        m = ok.sum(axis=0).astype("f8")
        pe = (np.where(ok, (p >= threshold).astype("f8"), 0.0)
              .sum(axis=0) / np.maximum(m, 1.0))
        oe = np.where(np.isfinite(o), (o >= threshold).astype("f8"), np.nan)
        d2 = (pe - oe) ** 2
        okg = (m > 0) & np.isfinite(d2)
        sw = (okg.astype("f8") * w).sum()
        out[t] = (np.where(okg, d2, 0.0) * w).sum() / max(float(sw), 1e-12)
    return out
