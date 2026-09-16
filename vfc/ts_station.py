# -*- coding: utf-8 -*-
"""分类指标层：格点→站点插值 + 降水 TS 列联表。

* 插值一律用**修正后的坐标数组**定位（绝不硬编码 0.25 步长），经度按
  环形处理（0/360 回绕），纬度非环形（越界裁剪并告警）；
* 列联表（逐阈值汇总额）：hits/misses/false → TS、命中率 POD、空报率 FAR、
  漏报率、预报偏差 BIAS（QX 规范 / GB/T 44956 的分类指标族）；
* 默认阈值表为业务通用值，**以 GB/T 44956—2024 正式文本为准**，可用
  CLI 覆盖：6h=0.1/3/10/20/40 mm，24h=0.1/10/25/50/100/250 mm。
"""
from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS = {
    6.0: [("≥0.1", 0.1), ("≥3", 3.0), ("≥10", 10.0),
          ("≥20", 20.0), ("≥40", 40.0)],
    24.0: [("≥0.1", 0.1), ("≥10", 10.0), ("≥25", 25.0),
           ("≥50", 50.0), ("≥100", 100.0), ("≥250", 250.0)],
}


# ---------------------------------------------------------------- 插值

def interp_to_stations(field, glat, glon, slat, slon, method="bilinear"):
    """(lat, lon) 场 → 站点值数组。glon 需均匀（环形），glat 任意（线性）。"""
    field = np.asarray(field, dtype="f8")
    glat = np.asarray(glat, dtype="f8")
    glon = np.mod(np.asarray(glon, dtype="f8"), 360.0)
    slat = np.asarray(slat, dtype="f8")
    slon = np.mod(np.asarray(slon, dtype="f8"), 360.0)

    if glat[0] > glat[-1]:                          # 统一为纬度升序
        glat = glat[::-1]
        field = field[::-1]
    # 纬度：非环形 bracket（越界裁剪到边缘格点）
    il = np.clip(np.searchsorted(glat, slat) - 1, 0, glat.size - 2)
    wl = np.clip((slat - glat[il]) / (glat[il + 1] - glat[il]), 0.0, 1.0)
    out_lat = slat < glat[0], slat > glat[-1]

    # 经度：环形 bracket（均匀步长，0/360 回绕）
    step = float(np.mean(np.diff(glon)))
    fi = np.floor((slon - glon[0]) / step).astype(int)
    wq = ((slon - glon[0]) / step) - fi
    i0 = np.mod(fi, glon.size)
    i1 = np.mod(fi + 1, glon.size)

    if method == "nearest":
        j0 = np.mod(np.round(fi + wq).astype(int), glon.size)
        jl = np.clip(np.searchsorted(glat, slat) - 1, 0, glat.size - 2)
        jl = np.where((slat - glat[jl]) > 0.5 * (glat[jl + 1] - glat[jl]),
                      jl + 1, jl)
        vals = field[jl, j0]
    elif method == "bilinear":
        v00 = field[il, i0]
        v01 = field[il, i1]
        v10 = field[il + 1, i0]
        v11 = field[il + 1, i1]
        vals = ((1 - wl) * (1 - wq) * v00 + (1 - wl) * wq * v01
                + wl * (1 - wq) * v10 + wl * wq * v11)
    else:
        raise ValueError("未知插值方法 %r（bilinear/nearest）" % method)
    if out_lat[0].any() or out_lat[1].any():
        import warnings
        warnings.warn("%d 个站点纬度越界，已取边缘格点值"
                      % int(out_lat[0].sum() + out_lat[1].sum()))
    return vals


# ---------------------------------------------------------------- 列联表

class TSContingency(object):
    """逐阈值汇总额的降水列联表：update(预报站点值, 观测站点值)。"""

    def __init__(self, thresholds):
        """thresholds: [(等级名, 阈值mm), ...]"""
        self.thresholds = list(thresholds)
        self.h = {name: 0 for name, _ in self.thresholds}
        self.m = {name: 0 for name, _ in self.thresholds}
        self.f = {name: 0 for name, _ in self.thresholds}
        self.n_pairs = 0

    def update(self, fcst, obs):
        fcst = np.asarray(fcst, dtype="f8")
        obs = np.asarray(obs, dtype="f8")
        ok = np.isfinite(fcst) & np.isfinite(obs)
        self.n_pairs += int(ok.sum())
        for name, thr in self.thresholds:
            o = obs[ok] >= thr
            f = fcst[ok] >= thr
            self.h[name] += int((o & f).sum())
            self.m[name] += int((o & ~f).sum())
            self.f[name] += int((~o & f).sum())

    def finalize(self):
        """[(等级, 阈值, h, m, f, TS, POD, FAR, 漏报率, BIAS), ...]"""
        out = []
        for name, thr in self.thresholds:
            h, m, f = self.h[name], self.m[name], self.f[name]
            def _r(a, b):
                return float(a) / b if b else float("nan")
            out.append({
                "grade": name, "threshold_mm": thr,
                "hits": h, "misses": m, "false_alarms": f, "n_pairs": self.n_pairs,
                "TS": _r(h, h + m + f), "POD": _r(h, h + m),
                "FAR": _r(f, f + h), "漏报率": _r(m, m + h),
                "BIAS": _r(h + f, h + m),
            })
        return out
