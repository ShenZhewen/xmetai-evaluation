#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""降水分类评分 CLI：确定性 / 集合预报 → TS + 超越式 AROC/BS/BSS。

确定性预报（无 member 层）只算 TS 系列指标；
集合预报（member_* 子目录）同时算：
  * TS 系列指标（各成员**平均场**走与确定性一致的流程，超越式阈值）；
  * AROC / BS / BSS（各成员**逐成员**插值后，按**超越式**阈值
    ``x ≥ 阈值`` 计算；阈值独立于 TS，窗口由 ``--window-aroc`` 单独指定）。

输出：确定性写 ``ts_<名>.csv``；集合写 ``ts_<名>.csv`` + ``aroc_bss_<名>.csv``
（逐 (窗口, 时效, 阈值) 行 + ``AVG`` 综合行）。
BSS 参考：给 ``--ref`` 用外部逐 6h 气候概率，缺省回退样本气候频率。

用法：
    # 确定性（无 member 层，只算 TS）
    python run_categorical.py --root ts_sample/fuxi_single_eval_results \
        --station-dir ts_sample/r0/2025 --window-ts 6 24 \
        --station-list "ts_sample/r0/zd_sta_10285.dat"

    # BSS 用外部逐 6h 气候概率（ref/MMDDHH.000，北京时，站号+4阈值概率）
    python run_categorical.py --root ts_sample/fuxi_ens_output \
        --station-dir ts_sample/r0/2025 --station-list "ts_sample/r0/zd_sta_10285.dat" \
        --window-ts 24 --window-aroc 6 \
        --ref ts_sample/r0/ref

    python run_categorical.py --self-test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from vfc.reader_categorical import ClassDataset
from vfc.station_categorical import (HourlyStations, accumulate_window,
                                        scan_station_dir,
                                        load_station_whitelist,
                                        subset_stations, RefProb)
from vfc.metric_categorical import (TSContingency, ProbEventHistogram,
                                       interp_to_stations, DEFAULT_THRESHOLDS,
                                       DEFAULT_AROC_THRESHOLDS)


def _build_plans(windows, thr_table, step, lead_hours, label):
    """按窗口 × 时效构建 (W, lead, i_start, i_end) 计划。

    要求每个窗口在 thr_table 里有阈值表、且为 step 的整数倍；时效须 ≥W
    且为 W 的整数倍（窗口边界对齐文件边界）。
    """
    plans = []
    for W in windows:
        W = float(W)
        if W not in thr_table:
            raise ValueError("%s 窗口 %gh 无阈值表（有 %s）"
                             % (label, W, sorted(thr_table)))
        if abs(W / step - round(W / step)) > 1e-6:
            raise ValueError("%s 窗口 %gh 不是时效步长 %gh 的整数倍，"
                             "无法按文件边界切分" % (label, W, step))
        n_per = int(round(W / step))
        for lead in lead_hours:
            lead = round(float(lead), 6)
            if lead < W or round(lead % W, 6) != 0:
                continue
            i_end = int(round(lead / step))
            plans.append((W, lead, i_end - n_per, i_end))
    return plans


# ---------------------------------------------------------------- 主流程

def run_categorical(root, station_dir, window_ts=(6.0, 24.0),
                       window_aroc=(6.0,), step=6.0,
                       var="TP", init_hour=0.0, outdir="results/categorical",
                       name=None, interp="bilinear", tz_shift=8.0,
                       tp_scale=1.0, station_list=None, thresholds=None,
                       aroc_thresholds=None, ref=None, workers=1,
                       verbose=True):
    """主流程：返回 (results dict, meta)。

    window_ts: TS 累积窗（小时），默认 (6, 24)；
    window_aroc: AROC/BS/BSS 累积窗（小时），默认 (6,)，可传 24 开启；
    thresholds: {窗口: [(名,值), ...]}（TS 用，超越式）；
    aroc_thresholds: {窗口: [(名,值), ...]}（AROC/BS/BSS 用，超越式，
        独立于 TS 阈值；缺省 DEFAULT_AROC_THRESHOLDS，6h/24h 同一套）；
    ref: 逐 6h 气候概率参考目录（ref/MMDDHH.000，北京时，仅用于 6h AROC 窗）；
        缺省回退样本气候频率 BSS。
    """
    print("开始")
    t0 = time.time()
    thresholds = thresholds or DEFAULT_THRESHOLDS
    whitelist = load_station_whitelist(station_list) if station_list else None
    ds = ClassDataset(root, var=var, step=step, init_hour=init_hour,
                      workers=workers)
    if ds.lead_hours.size == 0:
        raise ValueError("数据集无时效文件")

    is_ensemble = ds.n_members > 1
    paths = scan_station_dir(station_dir)
    station_cache = {}

    def _hour(t):
        if t not in station_cache:
            if t not in paths:
                return None
            station_cache[t] = HourlyStations(paths[t])
        return station_cache[t]

    name = name or os.path.basename(os.path.normpath(str(root)))
    os.makedirs(outdir, exist_ok=True)

    # 构建 TS / AROC 各自的 (窗口, 时效) 计划（窗口分别指定、阈值独立）
    ts_plans = _build_plans(window_ts, thresholds, step, ds.lead_hours, "TS")
    aroc_thr = aroc_thresholds if aroc_thresholds is not None \
        else DEFAULT_AROC_THRESHOLDS
    aroc_plans = (_build_plans(window_aroc, aroc_thr, step, ds.lead_hours,
                               "AROC") if is_ensemble else [])

    ts_tables = {(W, lead): TSContingency(thresholds[W])
                 for W, lead, _, _ in ts_plans}
    ref_prob = RefProb(ref) if (is_ensemble and ref is not None) else None
    peh_tables = ({(W, lead): ProbEventHistogram(ds.n_members, aroc_thr[W])
                   for W, lead, _, _ in aroc_plans} if is_ensemble else None)
    ref_sums = {}        # (W, lead, ti) -> [Σ(p_clim−o)², n]，仅当 ref
    bs_sums = {}         # (W, lead, ti) -> [Σ(p_fc−o)², n]，仅当 ref（与 BS_ref 同批样本）

    ts_keys = {(W, lead) for W, lead, _, _ in ts_plans}
    aroc_keys = {(W, lead) for W, lead, _, _ in aroc_plans}
    all_plans = ts_plans + [p for p in aroc_plans
                            if (p[0], p[1]) not in ts_keys]
    if not all_plans:
        raise ValueError("无有效 (窗口, 时效) 组合（窗口/时效不匹配？）")
    stat = {(W, lead): {"skipped": 0} for W, lead, _, _ in all_plans}

    for init_date in ds.init_dates:
        if is_ensemble:
            members = ds.read_members(init_date)      # (M, n_lead, lat, lon)
            tp_mean = members.mean(axis=0).astype("f4")
        else:
            tp_mean = ds.read_mean(init_date)
            members = None

        for W, lead, i_start, i_end in all_plans:
            v_bjt = init_date + dt.timedelta(hours=lead + tz_shift)
            hours = [v_bjt - dt.timedelta(hours=k)
                     for k in range(int(W) - 1, -1, -1)]
            missing = [t for t in hours if t not in paths]
            if missing:
                stat[(W, lead)]["skipped"] += 1
                continue
            acc = accumulate_window({t: _hour(t) for t in hours},
                                    hours[0], hours[-1])
            if whitelist is not None:
                acc = subset_stations(acc, whitelist)

            # TS：平均场窗口求和 → 插值 → 列联表
            if (W, lead) in ts_keys:
                field_mean = np.sum(tp_mean[i_start:i_end], axis=0,
                                    dtype="f8")
                fcst = interp_to_stations(field_mean, ds.lat, ds.lon,
                                          acc["lat"], acc["lon"],
                                          method=interp) * tp_scale
                ts_tables[(W, lead)].update(fcst, acc["rain"])

            # 超越式 AROC/BS/BSS（仅集合，逐成员插值 → (M, n_stations)）
            if members is not None and (W, lead) in aroc_keys:
                fields_m = np.sum(members[:, i_start:i_end], axis=1,
                                  dtype="f8")          # (M, lat, lon)
                fcst_m = interp_to_stations(fields_m, ds.lat, ds.lon,
                                            acc["lat"], acc["lon"],
                                            method=interp) * tp_scale
                peh_tables[(W, lead)].update(fcst_m, acc["rain"])
                # BSS 外部参考：按验证时刻北京时查逐站概率，缺站跳过（BS 与 BS_ref 同批样本）
                if ref_prob is not None and W == 6.0:
                    probs = ref_prob.probabilities(v_bjt, acc["id"])   # (n, 4)
                    keep = (np.isfinite(probs).all(axis=1)
                            & np.isfinite(acc["rain"])
                            & np.all(np.isfinite(fcst_m), axis=0))
                    rain = acc["rain"][keep]
                    probs = probs[keep]
                    fcst_m_keep = fcst_m[:, keep]
                    for ti, (_nm, thr) in enumerate(aroc_thr[W]):
                        o = (rain >= thr).astype("f8")
                        p_fc = (fcst_m_keep >= thr).mean(axis=0)       # 预报概率 k/M
                        pc = probs[:, ti]
                        s, c = ref_sums.get((W, lead, ti), (0.0, 0))
                        ref_sums[(W, lead, ti)] = (
                            s + float(((pc - o) ** 2).sum()),
                            c + int(rain.size))
                        bs_s, bs_c = bs_sums.get((W, lead, ti), (0.0, 0))
                        bs_sums[(W, lead, ti)] = (
                            bs_s + float(((p_fc - o) ** 2).sum()),
                            bs_c + int(rain.size))

    # 输出 TS（确定性 / 集合均写）
    ts_rows = []
    for W, lead, _, _ in ts_plans:
        for r in ts_tables[(W, lead)].finalize():
            ts_rows.append(dict(window_h=W, lead_h=lead, **r))
    ts_df = pd.DataFrame(ts_rows)
    ts_csv = os.path.join(outdir, "ts_%s.csv" % name)
    ts_df.to_csv(ts_csv, index=False, float_format="%.6g")

    # 输出超越式 AROC / BS / BSS（仅集合；逐 (窗口, 时效, 阈值) + AVG 综合行）
    ab_df = None
    ab_csv = None
    if peh_tables is not None:
        per = {(W, lead): h.finalize() for (W, lead), h in peh_tables.items()}
        ab_rows = []
        for (W, lead) in sorted(per):
            thr_list = aroc_thr[W]
            a_vals, bs_vals, bss_vals = [], [], []
            for ti, (nm, thr) in enumerate(thr_list):
                r = per[(W, lead)][ti]
                a = r.get("AROC", np.nan)
                bs = r.get("BS", np.nan)
                if ref_prob is not None and W == 6.0:
                    s, c = ref_sums.get((W, lead, ti), (0.0, 0))
                    bs_ref = s / c if c else np.nan
                    bs_s, bs_c = bs_sums.get((W, lead, ti), (0.0, 0))
                    bs = bs_s / bs_c if bs_c else np.nan   # BS 改用 ref-covered 站，与 BS_ref 同批
                    bss = (1 - bs / bs_ref
                           if (np.isfinite(bs) and np.isfinite(bs_ref)
                               and bs_ref > 0) else np.nan)
                else:
                    bs_ref = np.nan
                    bss = r.get("BSS", np.nan)
                ab_rows.append(dict(
                    window_h=W, lead_h=lead, grade=nm, threshold_mm=thr,
                    AROC=a, BS=bs, BS_ref=bs_ref, BSS=bss,
                    base_rate=r.get("base_rate", np.nan),
                    n_points=r.get("n_points", 0.0)))
                if np.isfinite(a):
                    a_vals.append(a)
                if np.isfinite(bss):
                    bs_vals.append(bs)
                    bss_vals.append(bss)
            # AVG 综合行：AROC/BSS 各自对可算阈值求平均（无事件阈值跳过）
            ab_rows.append(dict(
                window_h=W, lead_h=lead, grade="AVG", threshold_mm=np.nan,
                AROC=float(np.mean(a_vals)) if a_vals else np.nan,
                BS=float(np.mean(bs_vals)) if bs_vals else np.nan,
                BS_ref=np.nan,
                BSS=float(np.mean(bss_vals)) if bss_vals else np.nan,
                base_rate=np.nan, n_points=np.nan, n_used=len(a_vals)))
        ab_df = pd.DataFrame(ab_rows)
        ab_csv = os.path.join(outdir, "aroc_bss_%s.csv" % name)
        ab_df.to_csv(ab_csv, index=False, float_format="%.6g")

    elapsed = time.time() - t0
    meta_windows = {}
    for W in sorted(set(w for w, _, _, _ in all_plans)):
        meta_windows["%g" % W] = {
            "thresholds": [(nm, v) for nm, v in thresholds.get(W, [])],
            "aroc_thresholds": [(nm, v) for nm, v in aroc_thr.get(W, [])],
            "leads": {
                "%g" % lead: stat[(W, lead)]
                for w2, lead, _, _ in all_plans if w2 == W
            },
        }
    meta = {
        "root": str(root), "var": var, "step_h": step,
        "init_hour": init_hour, "mode": "ensemble" if is_ensemble else "deterministic",
        "n_members": int(ds.n_members),
        "init_dates": [d.strftime("%Y%m%d") for d in ds.init_dates],
        "station_dir": str(station_dir), "tz_shift_h": tz_shift,
        "interp": interp, "tp_scale": tp_scale,
        "ts_windows": sorted(set(w for w, _, _, _ in ts_plans)),
        "aroc_windows": sorted(set(w for w, _, _, _ in aroc_plans)),
        "windows": meta_windows,
        "station_list": station_list,
        "n_station_list": len(whitelist) if whitelist else None,
        "ref": ref,
        "bss_reference": ("外部逐 6h 气候概率（--ref，逐站×阈值，缺站跳过）"
                          if ref_prob is not None
                          else "样本气候频率 r(1−r)"),
        "elapsed_s": round(elapsed, 3),
        "grid": {"n_lat": int(ds.lat.size), "n_lon": int(ds.lon.size)},
    }
    with open(os.path.join(outdir, "%s_meta.json" % name), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    results = {"ts": ts_df, "ts_csv": ts_csv}
    if ab_df is not None:
        results["aroc_bss"] = ab_df
        results["aroc_bss_csv"] = ab_csv

    if verbose:
        mode = "集合%d成员" % ds.n_members if is_ensemble else "确定性"
        print("\n=== TS（%s，%s，%d 起报日期，step %gh，%s 插值，tz+%gh）==="
              % (name, mode, len(ds.init_dates), step, interp, tz_shift))
        cols = ["window_h", "lead_h", "grade", "threshold_mm", "hits",
                "misses", "false_alarms", "n_pairs", "TS", "POD", "FAR",
                "漏报率", "BIAS"]
        print(ts_df[cols].round(4).to_string(index=False))
        if ab_df is not None:
            print("\n=== 超越式 AROC/BS/BSS（%s，逐时效×阈值，站点真值，参考=%s）==="
                  % (name, "外部气候概率" if ref_prob is not None
                     else "样本气候频率"))
            ab_cols = ["window_h", "lead_h", "grade", "threshold_mm",
                       "AROC", "BS", "BS_ref", "BSS", "base_rate",
                       "n_points", "n_used"]
            print(ab_df[ab_cols].round(4).to_string(index=False))
        print("\n已写出:")
        for fp in ([ts_csv] + ([ab_csv] if ab_csv else [])):
            print("        " + fp)
        print("\n总耗时: %.1f 秒（%.2f 分钟）" % (elapsed, elapsed / 60.0))
    ds.close()
    return results, meta


# ---------------------------------------------------------------- 合成辅助（self-test 用）

def _write_field_nc(path, field, lat, lon, var="TP"):
    """写单场 nc（每时效一个文件格式）：变量 var (lat, lon) + lat/lon 坐标。"""
    try:
        import netCDF4 as nc
        with nc.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("lat", lat.size)
            ds.createDimension("lon", lon.size)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            ds.createVariable(var, "f4", ("lat", "lon"))[:] = field
    except ImportError:
        import h5py
        with h5py.File(path, "w") as h:
            h["lat"] = lat
            h["lon"] = lon
            h[var] = field.astype("f4")


def _write_station(path, when, rows):
    """写合成 diamond3（GBK）。rows: [(站号, 经度, 纬度, 站高, 1h降水), ...]"""
    with open(path, "wb") as f:
        f.write("diamond 3 合成小时降水\n".encode("gbk"))
        f.write(("%d %d %d %d 1000 0 0 0 0 1 %d\n"
                 % (when.year % 100, when.month, when.day, when.hour,
                    len(rows))).encode("ascii"))
        for r in rows:
            f.write(("%d %.4f %.4f %.1f %.2f\n" % r).encode("ascii"))


# ---------------------------------------------------------------- 自检

def _self_test() -> int:
    ok = [0, 0]

    def check(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))

    # 1) ProbEventHistogram：完美区分（超越式）→ AROC=1、BSS=1
    ph = ProbEventHistogram(3, [("≥1", 1.0)])
    m = np.zeros((3, 4))
    o = np.zeros(4)
    m[:, :2] = 5.0                                   # 事件点：全成员超阈
    o[:2] = 5.0
    ph.update(m, o)
    r = ph.finalize()[0]
    check("ProbEventHistogram：完美概率预报 AROC=1、BSS=1",
          np.isclose(r["AROC"], 1.0) and np.isclose(r["BSS"], 1.0))

    # 2) ProbEventHistogram：无区分常数概率 → AROC=0.5、BSS=-1/9
    ph = ProbEventHistogram(3, [("≥1", 1.0)])
    m = np.zeros((3, 4))
    m[0] = 5.0                                       # 每点恰 1/3 成员超阈
    o = np.zeros(4)
    o[:2] = 5.0                                      # 事件率 0.5
    ph.update(m, o)
    r = ph.finalize()[0]
    check("ProbEventHistogram：无区分 AROC=0.5、BSS=-1/9",
          np.isclose(r["AROC"], 0.5) and np.isclose(r["BSS"], -1.0 / 9))

    # 3) ProbEventHistogram：NaN 显式剔除（观测 NaN 的站点不计入 n_points）
    ph = ProbEventHistogram(3, [("≥1", 1.0)])
    m = np.full((3, 5), 5.0)
    o = np.array([5.0, 5.0, np.nan, np.nan, np.nan])
    ph.update(m, o)
    r = ph.finalize()[0]
    check("ProbEventHistogram：观测 NaN 剔除（n_points=2）",
          np.isclose(r["n_points"], 2.0))

    # 4) AROC 默认阈值含 6h 与 24h（同一套）
    check("AROC 默认阈值含 6h 与 24h（同一套）",
          set(DEFAULT_AROC_THRESHOLDS) == {6.0, 24.0}
          and [v for _, v in DEFAULT_AROC_THRESHOLDS[6.0]]
          == [v for _, v in DEFAULT_AROC_THRESHOLDS[24.0]])

    # 5) 向量化插值 = 逐成员插值（逐位一致）
    g5lat = np.arange(-15.0, 16.0, 2.0)
    g5lon = np.arange(32) * 11.25
    s5lat = np.array([5.0, -3.0, 12.5])
    s5lon = np.array([5.0, 200.0, 350.0])
    rng = np.random.default_rng(0)
    f3 = rng.standard_normal((3, g5lat.size, g5lon.size))
    for mth in ("bilinear", "nearest"):
        vec = interp_to_stations(f3, g5lat, g5lon, s5lat, s5lon, mth)
        loop = np.stack([interp_to_stations(f3[m], g5lat, g5lon, s5lat,
                                            s5lon, mth) for m in range(3)])
        check("interp_to_stations：向量化与逐成员逐位一致（%s）" % mth,
              np.array_equal(vec, loop))

    # 4) 端到端：确定性 + 集合（3 成员相同 → 平均场=成员场）
    glat = np.arange(-15.0, 16.0, 2.0)
    glon = np.arange(32) * 11.25
    c = [1.0, 0.0, 3.0, 0.0]
    tmp = tempfile.mkdtemp(prefix="vfc_cls_selftest_")
    try:
        import shutil
        # 站点：站1=0.5/h，站2/3=0，站4 缺报（03/09/15/21 时）
        sdir = os.path.join(tmp, "stations")
        os.makedirs(sdir)
        stations = [(1, 5.0, 5.0, 10.0, 0.5), (2, 15.0, 5.0, 10.0, 0.0),
                    (3, 25.0, 5.0, 10.0, 0.0), (4, 35.0, 5.0, 10.0, 0.0)]
        t0 = dt.datetime(2025, 1, 1, 0)
        for k in range(49):
            t = t0 + dt.timedelta(hours=k)
            rows = [(i, lo, la, al, r) for (i, lo, la, al, r) in stations
                    if not (i == 4 and t.hour in (3, 9, 15, 21))]
            _write_station(os.path.join(sdir, t.strftime("%Y%m%d%H.000")),
                           t, rows)

        def make_root(kind):
            r = os.path.join(tmp, kind)
            for day in ("20250101", "20250102"):
                d = os.path.join(r, day)
                members = ["member_000", "member_001", "member_002"] \
                    if kind == "ens" else [None]
                for mem in members:
                    base = os.path.join(d, mem) if mem else d
                    os.makedirs(base)
                    for n, cn in enumerate(c, start=1):
                        field = np.full((glat.size, glon.size), cn, dtype="f4")
                        _write_field_nc(os.path.join(base, "%03d.nc" % n),
                                        field, glat, glon)

        make_root("det")
        make_root("ens")

        thr = {6.0: [("≥1", 1.0)], 24.0: [("≥1", 1.0)]}
        aroc_thr = {6.0: [("≥1", 1.0)], 24.0: [("≥1", 1.0)]}
        # 确定性：TS 解析解（W=6 lead=6: 单起报 h=1 m=0 f=2）
        res, meta = run_categorical(os.path.join(tmp, "det"), sdir,
                                       window_ts=[6.0], step=6.0,
                                       tz_shift=0.0, thresholds=thr,
                                       outdir=os.path.join(tmp, "out_det"),
                                       verbose=False)
        d6 = res["ts"].set_index("lead_h").loc[6.0]
        check("确定性：TS 只算（W=6 lead=6 h=2 m=0 f=4，跨 2 起报）",
              (d6["hits"], d6["misses"], d6["false_alarms"]) == (2, 0, 4)
              and meta["mode"] == "deterministic")

        # 集合：TS 6h，AROC 6h（窗口分别指定）
        res, meta = run_categorical(os.path.join(tmp, "ens"), sdir,
                                       window_ts=[6.0], window_aroc=[6.0],
                                       step=6.0, tz_shift=0.0, thresholds=thr,
                                       aroc_thresholds=aroc_thr,
                                       outdir=os.path.join(tmp, "out_ens"),
                                       verbose=False)
        e6 = res["ts"].set_index("lead_h").loc[6.0]
        check("集合：TS=平均场解析解（h=2 m=0 f=4，跨 2 起报）",
              (e6["hits"], e6["misses"], e6["false_alarms"]) == (2, 0, 4)
              and meta["mode"] == "ensemble" and meta["n_members"] == 3)
        check("集合：超越式 AROC/BSS 已输出（4 时效 × (1 阈值+AVG)=8 行）",
              "aroc_bss" in res and len(res["aroc_bss"]) == 8)
        # TS 测 6h+24h，AROC 只测 6h（window_aroc 只传 6）
        res2, _ = run_categorical(os.path.join(tmp, "ens"), sdir,
                                     window_ts=[6.0, 24.0], window_aroc=[6.0],
                                     step=6.0, tz_shift=0.0, thresholds=thr,
                                     aroc_thresholds=aroc_thr,
                                     outdir=os.path.join(tmp, "out_ens2"),
                                     verbose=False)
        check("集合：AROC 只测 6h（TS 测 6h+24h）",
              set(res2["aroc_bss"]["window_h"]) == {6.0}
              and set(res2["ts"]["window_h"]) == {6.0, 24.0})
        # AROC 也测 24h（window_aroc 传 6 24，同阈值）
        res4, _ = run_categorical(os.path.join(tmp, "ens"), sdir,
                                     window_ts=[6.0, 24.0],
                                     window_aroc=[6.0, 24.0],
                                     step=6.0, tz_shift=0.0, thresholds=thr,
                                     aroc_thresholds=aroc_thr,
                                     outdir=os.path.join(tmp, "out_ens4"),
                                     verbose=False)
        check("集合：AROC 可测 6h+24h（同阈值）",
              set(res4["aroc_bss"]["window_h"]) == {6.0, 24.0})
        # 外部 ref（逐 6h 气候概率，BS_ref=mean((p_clim−o)²)）
        refdir = os.path.join(tmp, "ref")
        os.makedirs(refdir)
        for hh in ("010106", "010112", "010118", "010200",
                   "010206", "010212", "010218", "010300"):
            with open(os.path.join(refdir, "%s.000" % hh), "w") as f:
                for sid in (1, 2, 3, 4):
                    f.write("%d 0.5 0.5 0.5 0.5\n" % sid)
        res3, _ = run_categorical(os.path.join(tmp, "ens"), sdir,
                                     window_ts=[6.0], window_aroc=[6.0],
                                     step=6.0, tz_shift=0.0, thresholds=thr,
                                     aroc_thresholds=aroc_thr,
                                     ref=refdir,
                                     outdir=os.path.join(tmp, "out_ens3"),
                                     verbose=False)
        bb = res3["aroc_bss"]
        check("集合：外部 ref BS_ref=mean((p_clim−o)²)（p_clim=0.5 → 0.25）",
              np.allclose(bb.loc[bb["grade"] != "AVG", "BS_ref"], 0.25))
        # 并行读（workers=2）与串行读（workers=1）逐位一致
        res_p, _ = run_categorical(os.path.join(tmp, "ens"), sdir,
                                      window_ts=[6.0], window_aroc=[6.0],
                                      step=6.0, tz_shift=0.0, thresholds=thr,
                                      aroc_thresholds=aroc_thr, workers=2,
                                      outdir=os.path.join(tmp, "out_par"),
                                      verbose=False)
        check("并行读：workers=2 与 workers=1 结果逐位一致",
              res_p["ts"].equals(res["ts"])
              and res_p["aroc_bss"].equals(res["aroc_bss"]))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n自检结果: %d/%d 通过" % (ok[0], ok[1]))
    return 0 if ok[0] == ok[1] else 1


# ---------------------------------------------------------------- CLI

def _parse_thresholds(s):
    return [(x.split(":")[0] if ":" in x else "≥%g" % float(x.split(":")[-1]),
             float(x.split(":")[-1]))
            for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="降水分类评分：确定性/集合 → TS + AROC/BS/BSS")
    ap.add_argument("--root", help="模型输出根目录（下为 YYYYMMDD 起报日期子目录）")
    ap.add_argument("--station-dir", default="/workspace/data/worm/r0/2025",
                    help="逐小时站点 diamond3 目录（文件名 YYYYMMDDHH.000）")
    ap.add_argument("--station-list", default="/workspace/data/worm/r0/zd_sta_10285.dat",
                    help="站点白名单（zd_sta_*.dat），仅用清单内站点评分")
    ap.add_argument("--step", type=float, default=6.0,
                    help="时效步长（小时），文件序号 n 对应 lead=n*step，默认 6")
    ap.add_argument("--var", default="TP", help="降水变量名，默认 TP")
    ap.add_argument("--init-hour", type=float, default=0.0,
                    help="起报时刻（UTC 小时），默认 0")
    ap.add_argument("--window-ts", nargs="+", type=float, default=[24.0],
                    help="TS 累积窗（小时），默认 6 与 24（须为 step 整数倍）")
    ap.add_argument("--window-aroc", nargs="+", type=float, default=[6.0],
                    help="AROC/BS/BSS 累积窗（小时），默认只 6（须为 step 整数倍）")
    ap.add_argument("--thr6", default=None, help="覆盖 6h TS 阈值，如 '0.1,13,25'")
    ap.add_argument("--thr24", default=None, help="覆盖 24h TS 阈值，如 '0.1,10,25,50,100'")
    ap.add_argument("--aroc-thr6", default=None,
                    help="覆盖 6h AROC 阈值，如 '0.1,4,13,25'")
    ap.add_argument("--aroc-thr24", default=None,
                    help="覆盖 24h AROC 阈值（默认与 6h 相同，如 '0.1,4,13,25'）")
    ap.add_argument("--ref", default='/workspace/data/worm/r0/ref',
                    help="BSS 外部逐 6h 气候概率目录（ref/MMDDHH.000，北京时，"
                    "站号+4阈值概率，缺站跳过）；缺省回退样本气候频率")
    ap.add_argument("--workers", type=int, default=32,
                    help="并行读文件线程数（默认 1 串行；集合数据建议 8）")
    ap.add_argument("--interp", default="bilinear", choices=["bilinear", "nearest"])
    ap.add_argument("--tz-shift", type=float, default=8.0,
                    help="pred(UTC)→站点文件时区的小时差（默认 8，北京时）")
    ap.add_argument("--tp-scale", type=float, default=1.0,
                    help="tp 单位换算到 mm 的系数")
    ap.add_argument("--outdir", default="results/categorical")
    ap.add_argument("--name", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.root or not args.station_dir:
        ap.error("需要 --root 与 --station-dir（或 --self-test）")
    thresholds = {}
    for w, s in ((6.0, args.thr6), (24.0, args.thr24)):
        if s:
            thresholds[w] = _parse_thresholds(s)
    aroc_thresholds = {}
    for w, s in ((6.0, args.aroc_thr6), (24.0, args.aroc_thr24)):
        if s:
            aroc_thresholds[w] = _parse_thresholds(s)
    run_categorical(args.root, args.station_dir,
                       window_ts=args.window_ts, window_aroc=args.window_aroc,
                       step=args.step, var=args.var, init_hour=args.init_hour,
                       outdir=args.outdir, name=args.name, interp=args.interp,
                       tz_shift=args.tz_shift, tp_scale=args.tp_scale,
                       station_list=args.station_list,
                       thresholds=thresholds or None,
                       aroc_thresholds=aroc_thresholds or None,
                       ref=args.ref, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
