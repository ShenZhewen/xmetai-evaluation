# -*- coding: utf-8 -*-
"""分类指标层：格点→站点插值 + TS 列联表 + 概率分类指标（AROC/BS/BSS）。

* 插值一律用**修正后的坐标数组**定位（绝不硬编码 0.25 步长），经度按
  环形处理（0/360 回绕），纬度非环形（越界裁剪并告警）；
* 列联表（逐阈值汇总额）：hits/misses/false → TS、命中率 POD、空报率 FAR、
  漏报率、预报偏差 BIAS（QX 规范 / GB/T 44956 的分类指标族）；
* TS 阈值见 ``DEFAULT_THRESHOLDS``（按窗口），可用 CLI ``--thr6/--thr24`` 覆盖；
* 概率分类指标（``ProbEventHistogram``）为**超越式**事件口径（``x ≥ 阈值``），
  AROC 阈值独立于 TS，见 ``DEFAULT_AROC_THRESHOLDS``（6h/24h 同一套）。
"""
from __future__ import annotations

import numpy as np

# NumPy ≥2.0 把 np.trapz 改名为 np.trapezoid，这里做兼容
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

DEFAULT_THRESHOLDS = {
    6.0: [("≥0.1", 0.1), ("≥13", 13.0), ("≥25", 25.0)],
    24.0: [("≥0.1", 0.1), ("≥10", 10.0), ("≥25", 25.0),
           ("≥50", 50.0), ("≥100", 100.0)],
}

# AROC/BS/BSS 的默认阈值（超越式事件口径 x ≥ 阈值），独立于 TS 阈值。
# 6h 与 24h 用同一套；默认 window_aroc=(6,)，只测 6h，传 24 可开启。
DEFAULT_AROC_THRESHOLDS = {
    6.0: [("≥0.1", 0.1), ("≥4", 4.0), ("≥13", 13.0), ("≥25", 25.0)],
    24.0: [("≥0.1", 0.1), ("≥4", 4.0), ("≥13", 13.0), ("≥25", 25.0)],
}


# ---------------------------------------------------------------- 插值

def interp_to_stations(field, glat, glon, slat, slon, method="bilinear"):
    """(…, lat, lon) 场 → (…, n_stations) 站点值数组。glon 需均匀（环形），glat 任意（线性）。

    ``field`` 可以是 ``(lat, lon)``，也可以前面带任意个维（如 ``(M, lat, lon)``
    的集合成员场）；插值权重只算一遍、对前面各维独立作用，结果与逐维调用
    **逐位一致**（浮点乘加序列不变）。
    """
    field = np.asarray(field, dtype="f8")
    glat = np.asarray(glat, dtype="f8")
    glon = np.mod(np.asarray(glon, dtype="f8"), 360.0)
    slat = np.asarray(slat, dtype="f8")
    slon = np.mod(np.asarray(slon, dtype="f8"), 360.0)

    if glat[0] > glat[-1]:                          # 统一为纬度升序
        glat = glat[::-1]
        field = field[..., ::-1, :]                 # 翻转 lat 轴（保留前面各维）
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
        vals = field[..., jl, j0]
    elif method == "bilinear":
        v00 = field[..., il, i0]
        v01 = field[..., il, i1]
        v10 = field[..., il + 1, i0]
        v11 = field[..., il + 1, i1]
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


# ---------------------------------------------------------------- 概率分类指标

class ProbEventHistogram(object):
    """(成员超越计数, 观测二值) 的直方图 → AROC / BS / BSS（多阈值，超越式）。

    一个实例维护多个**阈值**（超越式事件 ``x ≥ 阈值``）的直方图（内部 dict）。

    * **多阈值**：一个实例装多个阈值（dict），与 ``TSContingency`` 同级；
    * **显式剔除 NaN**：观测或任一成员非有限时，该站点不参与统计；
    * **站点等权**：无纬度加权。
    BSS 参考为样本气候频率 r（观测事件发生率，BS_ref = r(1−r)）。

    ``update`` 的预报输入是 ``(M, n_stations)``（M 个成员的插值后站点值），
    观测是 ``(n_stations,)``。每个阈值维护一个 ``(M+1, 2)`` 直方图：
    行 = 超阈值成员数 cnt（0..M，对应概率级 p=cnt/M），列 = 观测事件（0/1）。
    """

    def __init__(self, n_member, thresholds):
        """thresholds: [(等级名, 阈值mm), ...]"""
        self.M = int(n_member)
        self.thresholds = [(name, float(thr)) for name, thr in thresholds]
        self.hists = {name: np.zeros((self.M + 1, 2), dtype="f8")
                      for name, _ in self.thresholds}

    def update(self, members, obs):
        """members: (M, n_stations) 成员预报；obs: (n_stations,) 观测。"""
        m = np.asarray(members, dtype="f8")
        o = np.asarray(obs, dtype="f8")
        # 显式剔除 NaN：观测有效 且 所有成员有效
        ok = np.isfinite(o) & np.all(np.isfinite(m), axis=0)
        m_ok = m[:, ok]
        o_ok = o[ok]
        for name, thr in self.thresholds:
            cnt = (m_ok >= thr).sum(axis=0).astype(int)
            ob = (o_ok >= thr).astype(int)
            idx = cnt * 2 + ob
            self.hists[name] += np.bincount(
                idx, minlength=2 * (self.M + 1)).reshape(self.M + 1, 2)

    def finalize(self):
        """[{"grade", "threshold_mm", "AROC", "BS", "BSS", "n_points",
        "base_rate"}, ...]（每阈值一行）。"""
        rows = []
        for name, thr in self.thresholds:
            rows.append(dict(grade=name, threshold_mm=thr,
                             **self._roc_bs(self.hists[name])))
        return rows

    def _roc_bs(self, hist):
        M = self.M
        tot = hist.sum()
        if tot <= 0:
            return {"AROC": np.nan, "BS": np.nan, "BSS": np.nan,
                    "n_points": 0.0, "base_rate": np.nan}
        obs1 = hist[:, 1].sum()
        obs0 = hist[:, 0].sum()
        p = np.arange(M + 1, dtype="f8") / float(M)
        # BS 只要有预报概率即可算（无事件区间也能惩罚空报）
        bs = float((hist[:, 0] * p ** 2
                    + hist[:, 1] * (1 - p) ** 2).sum() / tot)
        r = float(obs1 / tot)
        bs_ref = r * (1 - r)
        # AROC 需要事件/非事件两类都存在
        if obs1 > 0 and obs0 > 0:
            # ROC：按概率级从高到低累加命中/空报率，梯形面积
            hit = np.array([hist[j:, 1].sum() / obs1 for j in range(M + 1)])
            far = np.array([hist[j:, 0].sum() / obs0 for j in range(M + 1)])
            f = np.concatenate(([0.0], far[::-1]))
            h = np.concatenate(([0.0], hit[::-1]))
            aroc = float(_trapz(h, x=f))
        else:
            aroc = np.nan
        return {"AROC": aroc, "BS": bs,
                "BSS": float(1 - bs / bs_ref) if bs_ref > 0 else np.nan,
                "n_points": float(tot), "base_rate": r}
