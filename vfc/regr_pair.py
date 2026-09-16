# -*- coding: utf-8 -*-
# [连续量/regression] 编排层（单对 verify_pair）——属主：RMSE/ACC/FA/谱 负责人
"""编排层：读 → 对齐 → 变换 → 算 → 写出 CSV / JSON / PNG。

对外主入口 :func:`verify_pair`，CLI（run_rmse.py）与 Python 调用共用。

时效对应（2026-08 实测裁定）：本批 obs/pred 逐指标直配（实测
RMSE(pred[0],obs[0]) 远小于 RMSE(pred[0],obs[1])），但两边都不含 0 时次
——obs[0] 与 pred[0] 同在 "+1 步" 验证时刻，且 0829 组文件头 time 轴
还有 linspace 类步长错误。因此默认 pred_shift=0 直配，用 lead_step 把
时效标签重标为 (1..N)×H（0829 用 12h、0830 用 24h）；未来完整数据
（含 0 时次、步长正确）两个参数都不传。所有变换（tp 差分、气候态取值）
一律在重标后的真实时效轴上进行。
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from .io_nc import FieldFile, check_pair
from .metrics import RMSEAccumulator, AnomalyCorrelationAccumulator, \
    ActivityAccumulator
from .specs import get_spec, apply_transform, VAR_SPECS
from .metrics.spectrum import zonal_spectrum, wavenumber_axis,\
    wavelength_km

_METRICS = ("rmse", "acc", "fa", "spectrum")


def _default_name(obs_path) -> str:
    """obs_20220830.nc → 20220830。"""
    m = re.search(r"(\d{8})", os.path.basename(str(obs_path)))
    return m.group(1) if m else os.path.splitext(os.path.basename(str(obs_path)))[0]


def _bbox_indices(lat, lon, bbox):
    """bbox = (lat0, lat1, lon0, lon1)，经度允许负值（自动 mod 360）。"""
    la0, la1, lo0, lo1 = bbox
    lat = np.asarray(lat)
    lon = np.mod(np.asarray(lon), 360.0)
    ilat = np.where((lat >= min(la0, la1)) & (lat <= max(la0, la1)))[0]
    ilon = np.where((lon >= min(lo0, lo1)) & (lon <= max(lo0, lo1)))[0]
    if ilat.size == 0 or ilon.size == 0:
        raise ValueError("bbox %s 内没有任何格点" % (bbox,))
    return ilat, ilon


def _grid_desc(lat, lon, lon_fixed) -> dict:
    dlat = float(np.diff(lat).mean())
    dlon = float(np.diff(lon).mean())
    return {
        "n_lat": int(lat.size), "n_lon": int(lon.size),
        "lat_range": [float(lat[0]), float(lat[-1])], "dlat_deg": round(dlat, 6),
        "lon_range": [float(lon[0]), float(lon[-1])], "dlon_deg": round(dlon, 6),
        "lon_fixed": bool(lon_fixed),
        "lon_note": ("文件头 lon 为已知错误(linspace(0,360,n))，已按上游确认"
                     "替换为标准 0.25° 网格" if lon_fixed else "使用文件头坐标"),
    }


def _plot_curves(df, path, title, dashed_cols=()):
    """RMSE/ACC/FA 通用曲线图（每列一个子图；dashed_cols 画虚线对照）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cols = list(df.columns)
    n = len(cols)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    x = df.index.values
    xdays = x / 24.0 if x.max() > 120 else x
    xlabel = "lead time (day)" if x.max() > 120 else "lead time (h)"
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             squeeze=False)
    for k, c in enumerate(cols):
        ax = axes[k // ncol][k % ncol]
        ax.plot(xdays, df[c].values, lw=1.8, color="#1f77b4",
                ls="--" if c in dashed_cols else "-")
        ax.set_title(c, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(True, alpha=0.3)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_spectrum(specs, path, title):
    """功率谱 log-log 图：每要素一个子图，pred vs obs。specs: {var: DataFrame}"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    vars_ = list(specs)
    n = len(vars_)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for k, v in enumerate(vars_):
        ax = axes[k // ncol][k % ncol]
        d = specs[v]
        ax.loglog(d.index, d["pred"], lw=1.6, color="#1f77b4", label="pred")
        ax.loglog(d.index, d["obs"], lw=1.6, color="#d62728", label="obs")
        ax.set_title(v, fontsize=11)
        ax.set_xlabel("zonal wavenumber k", fontsize=9)
        ax.set_ylabel("power", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def verify_pair(obs_path, pred_path, variables=None, metrics=("rmse",),
                outdir="results", name=None, engine="auto", fix_lon=True,
                lat_weighted=True, deacc=True, bbox=None, plot=True, block=16,
                pred_shift=0, lead_step=None, climo_path=None,
                climo_source="未标注", climo_window=15, acc_centered=False,
                spec_lead_range=None, verbose=True):
    """对一组 obs/pred 文件计算逐时效检验指标并落盘。

    Parameters
    ----------
    metrics : ("rmse",) | 含 "acc"/"fa"/"spectrum" 的组合
        rmse  纬度加权均方根误差；
        acc   距平相关系数（需气候态；默认 uncentered，FDP/WB2 口径）；
        fa    预报活跃度 = 距平加权标准差（QX 公式3；需气候态；含 obs 侧）；
        spectrum  纬向 FFT 功率谱（pred vs obs；QX 球谐口径见 spectrum.py 备忘）。
    pred_shift : int    pred 相对 obs 的时步错位数。本批数据 pred 不含 0 时次，
        取 1（pred[i]↔obs[i+1]）；obs 与预报完全对应的新数据取 0（默认）。
    lead_step : float | None   时效轴重标（小时）：本批 obs/pred 均不含 0 时次
        且文件头步长有误（0829 组 time 轴为 linspace、本意 12h），真实时效
        = (1..N)×lead_step（0829 用 12、0830 组用 24）；未来完整数据
        （含 0 时次、步长正确）传 None 按文件轴。
    climo_path / climo_source / climo_window : 气候态（acc/fa 必需），
        格式约定见 vfc/climo.py 模块注释。
    acc_centered : bool   ACC 是否用减加权平均的经典皮尔逊（默认 uncentered）。
    spec_lead_range : (h0, h1) | None   功率谱平均的时效窗（小时）；None=全部。

    Returns
    -------
    (results, meta)：results 为 {指标: DataFrame}（spectrum 另含 "spectrum_csv"）。
    """
    metrics = [m.lower() for m in metrics]
    bad = [m for m in metrics if m not in _METRICS]
    if bad:
        raise ValueError("未知指标 %s，可选 %s" % (bad, _METRICS))
    need_climo = bool({"acc", "fa"} & set(metrics))

    obs = FieldFile(obs_path, engine=engine, fix_lon=fix_lon)
    pred = FieldFile(pred_path, engine=engine, fix_lon=fix_lon)
    climo = None
    try:
        check_pair(obs, pred)
        common = [v for v in obs.levels if v in pred.levels]
        _explicit = variables is not None
        variables = list(variables) if _explicit else common
        missing = [v for v in variables if v not in common]
        if missing:
            raise ValueError("请求的要素 %s 不在 obs/pred 公共要素 %s 中"
                             % (missing, common))
        # 派生要素（风速 = sqrt(u^2+v^2)）：u/v 分量两侧都有时自动补进清单
        for _dv, _s in VAR_SPECS.items():
            if not _s.derived:
                continue
            if all(c in common for c in _s.derived):
                if (not _explicit) or (_dv in variables):
                    if _dv not in variables:
                        variables.append(_dv)
        if pred_shift < 0 or pred_shift >= obs.lead_hours.size:
            raise ValueError("pred_shift=%d 越界（时效步数 %d）"
                             % (pred_shift, obs.lead_hours.size))
        if need_climo:
            from .climo import DailyClimatology
            if not climo_path:
                raise ValueError(
                    "acc/fa 需要逐日气候态：请用 --climo 指定文件"
                    "（格式约定见 vfc/climo.py 或 README；QX 规范用 CRA40 "
                    "1991–2020 逐日，对表 FDP 建议 ERA5 同源逐日气候态），"
                    "并用 --climo-source 标注来源")
            if obs.init_date is None:
                raise ValueError("无法从 obs 的 time:units=%r 解析起报时刻，"
                                 "不能计算验证日期（acc/fa 需要）"
                                 % obs.time_units)
            climo = DailyClimatology(climo_path, engine=engine,
                                     window=climo_window, fix_lon=fix_lon,
                                     source=climo_source)
            climo.check_grid(obs)
            lack = [v for v in variables
                    if not climo.has(v) and not get_spec(v).derived]
            if lack:
                raise ValueError("气候态 %s 缺少要素 %s（有 %s）"
                                 % (climo_path, lack, climo.levels))

        name = name or _default_name(obs_path)
        os.makedirs(outdir, exist_ok=True)

        ilat = ilon = None
        if bbox is not None:
            ilat, ilon = _bbox_indices(obs.lat, obs.lon, bbox)
            lat_sub = obs.lat[ilat]
        else:
            lat_sub = obs.lat

        n_lead = obs.lead_hours.size
        sl_o = slice(pred_shift, n_lead)             # pred[i] ↔ obs[i+shift]
        sl_p = slice(0, n_lead - pred_shift)
        if lead_step:
            leads_real = np.arange(1, n_lead - pred_shift + 1,
                                   dtype="f8") * float(lead_step)
        else:
            leads_real = obs.lead_hours[sl_o]        # 真实时效（取 obs 轴）
        if pred_shift and verbose:
            print("[vfc] 对齐: pred 不含 0 时次，按 shift=%d 对齐 "
                  "(pred[i]↔obs[i+%d])，真实时效 %g…%g h，丢弃 pred 末 %d 步"
                  % (pred_shift, pred_shift, leads_real[0], leads_real[-1],
                     pred_shift))

        def _sub(a):
            return a[:, ilat][:, :, ilon] if ilat is not None else a

        def _run_acc(acc, o, p):
            """分块驱动一个 update(i0, pred, obs) 型累积器并 finalize。"""
            for i0 in range(0, o.shape[0], block):
                acc.update(i0, p[i0:i0 + block], o[i0:i0 + block])
            return acc.finalize()

        rmse_s, acc_s, fa_cols, spectra = {}, {}, {}, {}
        for v in variables:
            spec = get_spec(v)
            comps = spec.derived
            if comps:
                _ou = _sub(obs.read(comps[0])[sl_o])
                _ov = _sub(obs.read(comps[1])[sl_o])
                o_al = np.sqrt(_ou * _ou + _ov * _ov)
                _pu = _sub(pred.read(comps[0])[sl_p])
                _pv = _sub(pred.read(comps[1])[sl_p])
                p_al = np.sqrt(_pu * _pu + _pv * _pv)
            else:
                o_al = _sub(obs.read(v)[sl_o])
                p_al = _sub(pred.read(v)[sl_p])

            if "rmse" in metrics:
                o_tr, lh = apply_transform(spec, o_al, leads_real, deacc)
                p_tr, _ = apply_transform(spec, p_al, leads_real, deacc)
                rmse_s[v] = pd.Series(
                    _run_acc(RMSEAccumulator(o_tr.shape[0], lat_sub,
                                             weighted=lat_weighted),
                             o_tr, p_tr), index=lh, name=v)

            if "fa" in metrics:
                if spec.derived and not climo.has(v):
                    if verbose:
                        print("[vfc] 提示: 气候态无 %s，跳过 fa" % v)
                else:
                    o_tr, lh = apply_transform(spec, o_al, leads_real, deacc)
                    p_tr, _ = apply_transform(spec, p_al, leads_real, deacc)
                    fa = ActivityAccumulator(o_tr.shape[0], lat_sub,
                                             weighted=lat_weighted)
                    o_an = o_tr - climo.field_for(v, lh, obs.init_date)
                    p_an = p_tr - climo.field_for(v, lh, obs.init_date)
                    f_f, f_o = _run_acc(fa, o_an, p_an)
                    fa_cols[v + "_pred"] = pd.Series(f_f, index=lh)
                    fa_cols[v + "_obs"] = pd.Series(f_o, index=lh)
                    fa_cols[v + "_bias"] = pd.Series(f_f - f_o, index=lh)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fa_cols[v + "_ratio"] = pd.Series(f_f / f_o, index=lh)

            if "acc" in metrics:
                if spec.deaccumulate:
                    if verbose:
                        print("[vfc] 提示: 距平相关对累积量(%s)无意义，跳过" % v)
                elif spec.derived and not climo.has(v):
                    if verbose:
                        print("[vfc] 提示: 气候态无 %s，跳过 acc" % v)
                else:
                    o_an = o_al - climo.field_for(v, leads_real, obs.init_date)
                    p_an = p_al - climo.field_for(v, leads_real, obs.init_date)
                    acc = AnomalyCorrelationAccumulator(
                        o_al.shape[0], lat_sub, weighted=lat_weighted,
                        centered=acc_centered)
                    acc_s[v] = pd.Series(_run_acc(acc, o_an, p_an),
                                         index=leads_real, name=v)

            if "spectrum" in metrics:
                if spec.deaccumulate:
                    if verbose:
                        print("[vfc] 提示: 功率谱跳过累积量 %s" % v)
                else:
                    m = np.ones_like(leads_real, dtype=bool)
                    if spec_lead_range:
                        m &= ((leads_real >= spec_lead_range[0])
                              & (leads_real <= spec_lead_range[1]))
                    if not m.any():
                        raise ValueError("时效窗 %s 内没有谱计算时效"
                                         % (spec_lead_range,))
                    psd_p = zonal_spectrum(p_al[m], lat_sub,
                                           weighted=lat_weighted).mean(axis=0)
                    psd_o = zonal_spectrum(o_al[m], lat_sub,
                                           weighted=lat_weighted).mean(axis=0)
                    k = wavenumber_axis(obs.lon.size)
                    d = pd.DataFrame({"pred": psd_p, "obs": psd_o,
                                      "wavelength_km": wavelength_km(k)},
                                     index=pd.Index(k, name="wavenumber"))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        d["pred/obs"] = d["pred"] / d["obs"]
                    spectra[v] = d.iloc[1:]            # k=0 已置零，不输出

        # q700/q2m 统一按 g/kg 报告（内部 kg/kg；rmse/fa/谱 ×1000/×1e6）
        _Q_VARS = tuple(v for v, s in VAR_SPECS.items()
                        if s.unit == "kg kg-1")
        for _v in _Q_VARS:
            if _v in rmse_s:
                rmse_s[_v] = rmse_s[_v] * 1000.0
            for _suf in ("_pred", "_obs", "_bias"):
                _k = _v + _suf
                if _k in fa_cols:
                    fa_cols[_k] = fa_cols[_k] * 1000.0
            if _v in spectra:
                spectra[_v]["pred"] = spectra[_v]["pred"] * 1e6
                spectra[_v]["obs"] = spectra[_v]["obs"] * 1e6

        # -- 落盘 ---------------------------------------------------------------
        files = []
        results = {}

        def _write_table(table, tag, dashed=()):
            df = pd.DataFrame(table)
            csv = os.path.join(outdir, "%s_%s.csv" % (tag, name))
            df.to_csv(csv, float_format="%.6g", index_label=df.index.name
                      or "lead_hour")
            files.append(csv)
            results[tag] = df
            if plot:
                png = os.path.join(outdir, "%s_%s.png" % (tag, name))
                if _plot_curves(df, png, "%s  (%s)" % (tag.upper(), name),
                                dashed_cols=dashed):
                    files.append(png)
            return df

        if rmse_s:
            _write_table(rmse_s, "rmse")
        if acc_s:
            _write_table(acc_s, "acc")
        if fa_cols:
            _write_table(fa_cols, "fa",
                         dashed=tuple(c for c in fa_cols if c.endswith("_obs")))
        if spectra:
            for v, d in spectra.items():
                csv = os.path.join(outdir, "spectrum_%s_%s.csv" % (name, v))
                d.to_csv(csv, float_format="%.6g")
                files.append(csv)
            results["spectrum"] = pd.concat(
                {v: d[["pred", "obs"]] for v, d in spectra.items()}, axis=1)
            if plot:
                png = os.path.join(outdir, "spectrum_%s.png" % name)
                if _plot_spectrum(spectra, png,
                                  "power spectrum  (%s)" % name):
                    files.append(png)

        meta = {
            "name": name, "obs": str(obs_path), "pred": str(pred_path),
            "engine": obs.engine, "metrics": metrics,
            "variables": variables,
            "units": {v: get_spec(v).unit for v in variables},
            "n_leads": int(n_lead),
            "lead_hours": [float(x) for x in leads_real],
            "pred_shift": int(pred_shift),
            "pred_shift_note": ("pred 不含 0 时次，pred[i]↔obs[i+%d]，真实时效"
                                "取 obs 轴" % pred_shift) if pred_shift else
                               "obs/pred 时效逐点直配",
            "lead_step": lead_step,
            "lead_note": ("时效轴按不含 0 时次重标: (1..%d)×%gh"
                          % (n_lead - pred_shift, lead_step)) if lead_step
            else "时效轴取文件 time 轴",
            "time_units_obs": obs.time_units,
            "init_date": str(obs.init_date),
            "grid": _grid_desc(obs.lat, obs.lon, obs.lon_fixed),
            "options": {"lat_weighted": lat_weighted, "tp_deaccumulated": deacc,
                        "fix_lon": fix_lon, "bbox": list(bbox) if bbox else None,
                        "block": block, "acc_centered": acc_centered,
                        "spec_lead_range": list(spec_lead_range)
                        if spec_lead_range else None},
            "climatology": ({"path": climo_path, "source": climo_source,
                             "window_days": climo_window} if climo else None),
        }
        json_path = os.path.join(outdir, "%s_meta.json" % name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        files.append(json_path)

        if verbose:
            for tag in ("rmse", "acc", "fa"):
                if tag in results:
                    print("\n=== %s（%s，%s）===" % (
                        tag.upper(), name,
                        "cos(lat) 加权" if lat_weighted else "不加权"))
                    print(results[tag].round(4).to_string())
            if spectra:
                print("\n=== SPECTRUM（%s，纬向 FFT，时效窗 %s）=== 每要素"
                      "谱曲线见 CSV" % (name, spec_lead_range or "全部时效"))
            print("\n已写出:")
            for fp in files:
                print("        " + fp)
        return results, meta
    finally:
        obs.close()
        pred.close()
        if climo is not None:
            climo.close()
