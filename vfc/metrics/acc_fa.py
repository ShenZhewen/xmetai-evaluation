# -*- coding: utf-8 -*-
"""连续量(regression) 指标：ACC / 预报活跃度 FA（流式累积器）。"""
from __future__ import annotations
import numpy as np
from ._weights import lat_weights

class AnomalyCorrelationAccumulator(object):
    """逐时效纬度加权距平相关系数（ACC）。

    uncentered（默认，FDP/WeatherBench2 同口径）；centered=经典皮尔逊。"""
    def __init__(self, n_lead, lat, weighted=True, centered=False):
        self.n_lead = int(n_lead)
        self.centered = bool(centered)
        w = lat_weights(lat, weighted)[:, None]
        self._w = w
        z = lambda: np.zeros(self.n_lead, dtype="f8")
        self._sab, self._saa, self._sbb = z(), z(), z()
        self._sa, self._sb, self._sw = z(), z(), z()
    def update(self, i0, f_anom, o_anom):
        f = np.asarray(f_anom, dtype="f8")
        o = np.asarray(o_anom, dtype="f8")
        t = f.shape[0]
        ok = np.isfinite(f) & np.isfinite(o)
        fm = np.where(ok, f, 0.0)
        om = np.where(ok, o, 0.0)
        wok = ok.astype("f8") * self._w
        self._sab[i0:i0 + t] += (fm * om * self._w).sum(axis=(1, 2))
        self._saa[i0:i0 + t] += (fm * fm * self._w).sum(axis=(1, 2))
        self._sbb[i0:i0 + t] += (om * om * self._w).sum(axis=(1, 2))
        self._sa[i0:i0 + t] += (fm * self._w).sum(axis=(1, 2))
        self._sb[i0:i0 + t] += (om * self._w).sum(axis=(1, 2))
        self._sw[i0:i0 + t] += wok.sum(axis=(1, 2))
    def finalize(self) -> np.ndarray:
        if self.centered:
            fa = self._sab - self._sa * self._sb / self._sw
            fv = self._saa - self._sa ** 2 / self._sw
            ov = self._sbb - self._sb ** 2 / self._sw
        else:
            fa, fv, ov = self._sab, self._saa, self._sbb
        den = np.sqrt(fv * ov)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, fa / den, np.nan)
        return r

class ActivityAccumulator(object):
    """逐时效预报活跃度 FA（距平场加权标准差；同时算预报侧/观测侧）。"""
    def __init__(self, n_lead, lat, weighted=True):
        self.n_lead = int(n_lead)
        self._w = lat_weights(lat, weighted)[:, None]
        z = lambda: np.zeros(self.n_lead, dtype="f8")
        self._f_sx, self._f_sxx, self._o_sx, self._o_sxx, self._sw = (
            z(), z(), z(), z(), z())
    def update(self, i0, f_anom, o_anom):
        f = np.asarray(f_anom, dtype="f8")
        o = np.asarray(o_anom, dtype="f8")
        t = f.shape[0]
        w = self._w
        for x, sx, sxx in ((f, self._f_sx, self._f_sxx),
                           (o, self._o_sx, self._o_sxx)):
            ok = np.isfinite(x)
            xm = np.where(ok, x, 0.0)
            sx[i0:i0 + t] += (xm * w).sum(axis=(1, 2))
            sxx[i0:i0 + t] += (xm * xm * w).sum(axis=(1, 2))
        self._sw[i0:i0 + t] += ((np.isfinite(f) & np.isfinite(o)
                                 ).astype("f8") * self._w).sum(axis=(1, 2))
    @staticmethod
    def _std(sx, sxx, sw):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(np.maximum(sxx / sw - (sx / sw) ** 2, 0.0))
    def finalize(self):
        return self._std(self._f_sx, self._f_sxx, self._sw), \
            self._std(self._o_sx, self._o_sxx, self._sw)
