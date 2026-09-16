#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""台风路径/强度预报误差检验 CLI：pred 场诊断台风中心 vs babj 实况。

用法：
    python run_tc.py --pred pred.nc --tcid 2205 --babj-dir /mnt/d/babj \
        --lead-step 6 --outdir results/tc
    python run_tc.py --self-test

流程：babj 时效 000 实况给出起报时刻的台风初始位置 → pred 的 msl 场
以前一时次位置为中心 ±3°（前 12h ±4°）方框 argmin 链式诊断中心 →
强度=当前中心 ±5° 框内 10m 风最大值 → 与 babj 实况（北京时，UTC+8）
逐时效配对：路径误差(km)、at/ct 分解、风速误差(m/s)、中心气压误差(hPa)。

集合模式（--forecast-type ens）默认按方案 B 汇总：先对成员位置求集合
平均，再与实况求误差（与 recompute_ens_track_error.py 同口径）；
--ens-agg A 可切换为方案 A（各成员误差的平均）。成员检验用进程池并行
（--workers N，默认 min(CPU核数, 成员数)）。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import json
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from vfc.io_nc import DataFileError, FieldFile
from vfc.typhoon import (TyphoonDataError, along_cross_track,
                         diagnose_track, great_circle_km, match_errors,
                         read_babj_analyses)


def _write_nc(path, data, levels, lat, lon, time, time_units):
    try:
        import netCDF4 as nc
        with nc.Dataset(path, "w", format="NETCDF4") as ds:
            for name, n in (("time", data.shape[0]), ("level", len(levels)),
                            ("lat", lat.size), ("lon", lon.size)):
                ds.createDimension(name, n)
            t = ds.createVariable("time", "f8", ("time",))
            t.units = time_units
            t[:] = time
            lv = ds.createVariable("level", str, ("level",))
            for i, s in enumerate(levels):
                lv[i] = s
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            ds.createVariable("data", "f4",
                              ("time", "level", "lat", "lon"))[:] = data
    except ImportError:
        import h5py
        with h5py.File(path, "w") as h:
            h["time"] = time
            h["time"].attrs["units"] = time_units
            h["level"] = np.array(levels, dtype=h5py.string_dtype())
            h["lat"] = lat
            h["lon"] = lon
            h["data"] = data.astype("f4")


def _write_babj(path, tcname, tcid, rows):
    """rows: [(北京时, 时效, lon, lat, pmin, vmax), ...]"""
    with open(path, "wb") as f:
        f.write(("diamond 7 %s台风路径\n" % tcname).encode("gbk"))
        f.write(("%s      %s   20            %d\n"
                 % (tcname, tcid, len(rows))).encode("gbk"))
        f.write(b"\n")
        for t, ff, lo, la, p, v in rows:
            f.write(("%4d %3d %3d %3d %3d %11.1f %11.1f %6d %5d"
                     "     0.0       0.0       0.0       0.0\n"
                     % (t.year, t.month, t.day, t.hour, ff, lo, la,
                        int(p), int(v))).encode("ascii"))


def _self_test() -> int:
    from vfc.typhoon import great_circle_km, along_cross_track
    from vfc.typhoon import find_center_by_slp, max_wind_in_box
    ok = [0, 0]

    def check(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))

    # 1) 球面距离：北京—上海 ≈ 1068 km；与参考 acos 公式一致
    d = great_circle_km(39.9, 116.4, 31.2, 121.5)
    acos = math_acos(39.9, 116.4, 31.2, 121.5)
    check("球面距离量级正确且与参考公式一致",
          1000 < d < 1130 and abs(d - acos) < 1e-6)

    # 2) at/ct：正东位移 → ct≈0、at≈距离；正北位移 → at≈0、ct≈距离
    at, ct = along_cross_track(20, 120, 20, 121, 20, 121)     # 预报=实况
    check("at/ct：完美预报 at=ct=0", abs(at) < 1e-9 and abs(ct) < 1e-9)
    at, ct = along_cross_track(20, 120, 20, 121, 20, 122)     # 预报偏东 1°
    d_east = float(great_circle_km(20, 122, 20, 121))
    check("at/ct：沿路径偏差 at≈球距、ct≈0",
          abs(at - d_east) < 0.5 and abs(ct) < 0.5)

    # 3) 方框搜索 + 经度回绕：中心在 359.9、涡旋在 0.4 处仍能找到
    glat = np.arange(30.0, -0.1, -0.25)
    glon = np.arange(0, 360, 0.25)
    LON, LAT = np.meshgrid(glon, glat)
    p = 100000 - 4000 * np.exp(-(((LAT - 10) ** 2
                                  + (((LON - 0.4 + 180) % 360 - 180) * np.cos(
                                      np.deg2rad(10))) ** 2) / 4.0))
    la, lo, pmin = find_center_by_slp(p, glat, glon, 10.0, 359.9, 3, 3)
    check("中心搜索：跨 0/360 缝回绕正确",
          abs(la - 10) <= 0.26 and abs(lo - 0.5) <= 0.3 and pmin < 96100)
    wind = 30 * np.exp(-(((LAT - 10) ** 2
                          + (((LON - 0.4 + 180) % 360 - 180) * np.cos(
                              np.deg2rad(10))) ** 2) / 25.0))
    check("强度：中心 ±5° 框内最大风 ≈ 30",
          abs(max_wind_in_box(wind, glat, glon, la, lo, 5, 5) - 30) < 0.5)

    # 4) babj 读取：只取时效 000，列位正确
    tmp = tempfile.mkdtemp(prefix="vfc_tc_selftest_")
    babj_path = os.path.join(tmp, "babj9901.dat")
    rows = []
    t0 = dt.datetime(2022, 9, 1, 8)                       # 北京时
    for k in range(4):                                    # 实况 逐 6h（只到 18h，
        rows.append((t0 + dt.timedelta(hours=6 * k), 0,   # 24h 时效将无实况）
                     130.0 + 0.6 * k, 10.0 + 0.3 * k, 950, 35))
    rows.append((t0, 12, 131.0, 10.5, 955, 36))           # 主观预报行(应被忽略)
    _write_babj(babj_path, "SYNTH", "9901", rows)
    name, tcid, analyses = read_babj_analyses(babj_path)
    check("babj：只取时效000（4 条）、字段正确",
          len(analyses) == 4 and analyses[t0]["lon"] == 130.0
          and analyses[t0]["pmin_hpa"] == 950
          and analyses[t0]["vmax_ms"] == 35)

    # 5) 端到端：合成移动涡旋 pred（msl+u10+v10）→ 诊断 → 配对误差
    T = 5                                                 # 0..24h 每 6h
    cy, cx0, vx, vy = 10.0, 130.0, 0.6, 0.3               # 每步向东北移动
    msl = np.empty((T, glat.size, glon.size))
    wspd = np.empty_like(msl)
    for k in range(T):
        cx = cx0 + vx * k
        d2 = ((LAT - (cy + vy * k)) ** 2
              + (((LON - cx + 180) % 360 - 180)
                 * np.cos(np.deg2rad(cy))) ** 2)
        msl[k] = 100000 - 5000 * np.exp(-d2 / 4.0)
        wspd[k] = 35 * np.exp(-d2 / 25.0)
    pred_path = os.path.join(tmp, "pred_tc.nc")
    _write_nc(pred_path, np.stack([msl, wspd, np.zeros_like(wspd)], axis=1),
              ["msl", "u10", "v10"], glat, glon,
              np.arange(T, dtype="f8") * 6.0,
              "hours since 2022-09-01 00:00:00")
    res, meta = run_tc(pred_path, babj_path, tcid="9901",
                       outdir=os.path.join(tmp, "out"), verbose=False,
                       u10_name="u10", v10_name="v10")
    d = res.dropna(subset=["track_err_km"])
    check("端到端：诊断位置贴合合成真值（≤30km）",
          d["track_err_km"].max() < 30)
    check("端到端：风速误差≈0、气压误差≈0",
          d["wind_err_ms"].abs().max() < 0.5
          and d["pmin_err_hpa"].abs().max() < 1.0)
    check("端到端：无实况时次为 NaN 而非外推",
          int(res["track_err_km"].isna().sum()) == 1 and len(d) == 4)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n自检结果: %d/%d 通过" % (ok[0], ok[1]))
    return 0 if ok[0] == ok[1] else 1


def math_acos(lat1, lon1, lat2, lon2):
    """参考脚本的 acos 距离公式（用于一致性对照）。"""
    k = (math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.cos(math.radians(lon1 - lon2))
         + math.sin(math.radians(lat1)) * math.sin(math.radians(lat2)))
    return 6371 * math.acos(max(-1.0, min(1.0, k)))


def _run_one_member(member_path, babj_path, tcid, outdir, engine, fix_lon,
                    lead_step, msl_name, u10_name, v10_name, tz_shift,
                    center_half, early_half, early_hours, intensity_half,
                    member_index):
    """单成员检验（供线程池并行调用）。

    返回 (member_index, member, frame|None, meta|None, err|None)；
    失败时 frame/meta 为 None、err 为错误信息。全部参数为基本类型，
    便于将来切换 ProcessPoolExecutor。
    """
    member = os.path.basename(member_path)
    try:
        member_df, member_meta = run_tc(
            member_path, babj_path, tcid, outdir=outdir,
            engine=engine, fix_lon=fix_lon, lead_step=lead_step,
            msl_name=msl_name, u10_name=u10_name,
            v10_name=v10_name, tz_shift=tz_shift,
            center_half=center_half, early_half=early_half,
            early_hours=early_hours, intensity_half=intensity_half,
            forecast_type="ens", verbose=False, write_output=False)
    except (DataFileError, TyphoonDataError, OSError, ValueError) as e:
        return member_index, member, None, None, str(e)
    frame = member_df.reset_index()
    frame.insert(0, "member", member)
    return member_index, member, frame, member_meta, None


def run_tc(pred_path, babj_path, tcid, outdir="results/tc", name=None,
           engine="auto", fix_lon=True, lead_step=None, msl_name="msl",
           u10_name="u10", v10_name="v10", tz_shift=8.0,
           center_half=3.0, early_half=4.0, early_hours=12.0,
           intensity_half=5.0, forecast_type="det", ens_agg="B",
           workers=None, verbose=True, write_output=True):
    """主流程：返回 (DataFrame, meta)。"""
    member_dirs = []
    if forecast_type == "ens" and os.path.isdir(pred_path):
        member_dirs = sorted(
            entry.path for entry in os.scandir(pred_path)
            if entry.is_dir() and entry.name.startswith("member_"))
    if member_dirs:
        # 成员间无共享状态（独立目录/独立 FieldFile）。用进程池而非线程池：
        # 本环境的 netCDF4/HDF5 构建非线程安全，多线程并发读不同文件会
        # 段错误（实测 Errno -101）；进程池各 worker 独立打开文件，
        # 绕开该问题且 numpy 计算真多核并行。
        n_workers = workers or min(os.cpu_count() or 1, len(member_dirs))
        results = {}
        failed_members = {}
        done = 0
        if verbose:
            print("[vfc-tc] 集合成员并行检验: %d 成员 × %d 进程"
                  % (len(member_dirs), n_workers), flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_info = {}
            for member_index, member_path in enumerate(member_dirs, start=1):
                fut = ex.submit(
                    _run_one_member, member_path, babj_path, tcid, outdir,
                    engine, fix_lon, lead_step, msl_name, u10_name, v10_name,
                    tz_shift, center_half, early_half, early_hours,
                    intensity_half, member_index)
                fut_info[fut] = (member_index, os.path.basename(member_path))
            for fut in as_completed(fut_info):
                idx, member = fut_info[fut]
                done += 1
                try:
                    idx, member, frame, member_meta, err = fut.result()
                except Exception as e:            # 进程级异常（如崩溃）
                    frame, member_meta, err = None, None, str(e)
                if err is not None or frame is None:
                    failed_members[member] = err or "成员进程异常退出"
                    if verbose:
                        print("[vfc-tc] 成员 %s 失败: %s"
                              % (member, failed_members[member]), flush=True)
                else:
                    results[idx] = (frame, member_meta)
                if verbose:
                    print("[vfc-tc] 完成 %d/%d: %s"
                          % (done, len(member_dirs), member), flush=True)
        # 按成员序号还原确定性顺序（与 member_dirs 排序一致）
        member_frames = [results[i][0] for i in sorted(results)]
        member_metas = [results[i][1] for i in sorted(results)]

        if not member_frames:
            detail = next(iter(failed_members.values()), "没有可用成员")
            raise TyphoonDataError("集合预报没有成功完成的成员: %s" % detail)

        members_df = pd.concat(member_frames, ignore_index=True)
        grouped = members_df.groupby("lead_h", sort=True)
        first_meta = member_metas[0]

        if ens_agg == "A":
            # 方案 A：先算各成员误差，再求平均（成员误差的平均）
            summary = grouped.agg(
                n_members=("member", "nunique"),
                n_valid_members=("track_err_km", "count"),
                track_err_mean_km=("track_err_km", "mean"),
                pmin_err_mean_hpa=("pmin_err_hpa", "mean"),
                at_mean_km=("at_km", "mean"),
                ct_mean_km=("ct_km", "mean"),
            )
        else:
            # 方案 B：先平均各成员位置得集合平均位置，再与实况求误差，
            # 与 recompute_ens_track_error.py 同口径。
            lead_order = sorted(grouped.groups)
            # 实况位置：同一时效各成员相同，取首个有实况的成员
            obs_by_lead = {}
            for lh in lead_order:
                g = grouped.get_group(lh).dropna(
                    subset=["obs_lat", "obs_lon"])
                if len(g):
                    obs_by_lead[lh] = (float(g["obs_lat"].iloc[0]),
                                       float(g["obs_lon"].iloc[0]))
            # 首时次 at/ct 基准：起报时刻实况（与 match_errors 口径一致）
            try:
                _, _, babj_ref = read_babj_analyses(babj_path)
                init_utc = pd.to_datetime(
                    first_meta["init_utc"]).to_pydatetime()
                ob0 = babj_ref.get(init_utc + dt.timedelta(hours=tz_shift))
                prev_obs = (ob0["lat"], ob0["lon"]) if ob0 is not None \
                    else None
            except (TyphoonDataError, ValueError, KeyError, TypeError):
                prev_obs = None
            if lead_order and lead_order[0] <= 0:
                prev_obs = None
            rows_b = []
            for lh in lead_order:
                g = grouped.get_group(lh)
                # 集合平均位置：各成员 (fcst_lat, fcst_lon) 算术平均
                # （与 recompute 脚本一致的朴素经度平均）
                mlat = float(g["fcst_lat"].mean())
                mlon = float(g["fcst_lon"].mean())
                ob = obs_by_lead.get(lh)
                if ob is not None:
                    err = float(great_circle_km(mlat, mlon, ob[0], ob[1]))
                    if prev_obs is None:
                        at = ct = float("nan")
                    else:
                        at, ct = along_cross_track(prev_obs[0], prev_obs[1],
                                                   ob[0], ob[1], mlat, mlon)
                    prev_obs = ob
                else:
                    err = at = ct = float("nan")
                rows_b.append({
                    "n_members": int(g["member"].nunique()),
                    "n_valid_members": int(g["track_err_km"].notna().sum()),
                    "track_err_mean_km": err,
                    "pmin_err_mean_hpa": float(g["pmin_err_hpa"].mean()),
                    "at_mean_km": at,
                    "ct_mean_km": ct,
                })
            summary = pd.DataFrame(
                rows_b, index=pd.Index(lead_order, name="lead_h"))
        result_name = name or "tc%s_%s" % (
            tcid, first_meta["init_utc"].replace("-", "").replace(":", "")
            .replace(" ", "")[:10])
        meta = dict(first_meta)
        meta.update({
            "pred": str(pred_path),
            "forecast_type": "ens",
            "n_members_found": len(member_dirs),
            "n_members_completed": len(member_frames),
            "n_workers": n_workers,
            "members": [os.path.basename(path) for path in member_dirs],
            "failed_members": failed_members,
            "aggregation": ("mean member error by lead_h" if ens_agg == "A"
                            else "ensemble mean position error by lead_h "
                                 "(先平均位置再求误差, 方案B)"),
        })
        if write_output:
            os.makedirs(outdir, exist_ok=True)
            members_csv = os.path.join(outdir, "%s_members.csv" % result_name)
            ensemble_csv = os.path.join(outdir, "%s_ensemble.csv" % result_name)
            members_df.to_csv(members_csv, index=False, float_format="%.6g")
            summary.to_csv(ensemble_csv, float_format="%.6g")
            with open(os.path.join(outdir, "%s_meta.json" % result_name), "w",
                      encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        if verbose:
            print("\n=== 集合台风检验（%s，完成 %d/%d 成员）==="
                  % (tcid, len(member_frames), len(member_dirs)))
            print(summary.round(2).to_string())
            if write_output:
                print("\n已写出: %s\n          %s"
                      % (members_csv, ensemble_csv))
        return summary, meta

    tcname, tcid_file, babj = read_babj_analyses(babj_path)
    if tcid_file != str(tcid):
        raise TyphoonDataError("babj 文件 %s 内编号 %s 与 --tcid %s 不符"
                               % (babj_path, tcid_file, tcid))
    pred = FieldFile(pred_path, engine=engine, fix_lon=fix_lon,
                     forecast_type=forecast_type)
    try:
        for v in (msl_name,):
            if v not in pred.levels:
                raise TyphoonDataError("%s 缺 %s（现有 %s）"
                                       % (pred_path, v, pred.levels))
        if pred.init_date is None:
            raise TyphoonDataError("无法解析 pred 起报时刻（time:units=%r）"
                                   % pred.time_units)
        init_bjt = pred.init_date + dt.timedelta(hours=tz_shift)
        ob0 = babj.get(init_bjt)
        if ob0 is None:
            near = min(babj, key=lambda t: abs((t - init_bjt).total_seconds()))
            raise TyphoonDataError(
                "babj 无 %s（北京时）的实况记录；最近记录 %s（差 %.1fh）"
                "——起报时刻须为 babj 分析时刻（00/12 UTC）"
                "常见原因：预报从 +6h 起编，babj 没有对应有效时间的分析场；可忽略并继续下一条。"
                % (init_bjt, near, abs((near - init_bjt).total_seconds()) / 3600))
        has_wind = u10_name in pred.levels and v10_name in pred.levels
        if not has_wind and verbose:
            print("[vfc-tc] 提示: pred 缺 %s/%s，强度项将为空"
                  "（现有 %s）" % (u10_name, v10_name, pred.levels))
        msl = pred.read(msl_name)
        wind = None
        if has_wind:
            u = pred.read(u10_name)
            v = pred.read(v10_name)
            wind = np.sqrt(u.astype("f8") ** 2 + v.astype("f8") ** 2)
        n = msl.shape[0]
        leads = (np.arange(1, n + 1, dtype="f8") * lead_step if lead_step
                 else pred.lead_hours)
        track = diagnose_track(msl, pred.lat, pred.lon, ob0["lat"],
                               ob0["lon"], leads, wind=wind,
                               center_half=center_half, early_half=early_half,
                               early_hours=early_hours,
                               intensity_half=intensity_half)
        rows = match_errors(track, leads, pred.init_date, babj,
                            tz_shift=tz_shift)
        df = pd.DataFrame(rows).set_index("lead_h")
        # ------ End of error evaluation -------
        name = name or "tc%s_%s" % (tcid, pred.init_date.strftime("%Y%m%d%H"))
        meta = {
            "pred": str(pred_path), "babj": str(babj_path),
            "tcid": str(tcid), "tcname": tcname,
            "init_utc": str(pred.init_date), "init_bjt": str(init_bjt),
            "init_pos": [ob0["lat"], ob0["lon"]], "engine": pred.engine,
            "forecast_type": pred.forecast_type,
            "lead_step": lead_step, "tz_shift_h": tz_shift,
            "search": {"center_half_deg": center_half,
                       "early_half_deg": early_half,
                       "early_hours": early_hours,
                       "intensity_half_deg": intensity_half,
                       "intensity_center": "当前诊断中心"},
            "has_wind": has_wind,
            "n_matched": int(df["track_err_km"].notna().sum()),
            "grid": {"n_lat": int(pred.lat.size),
                     "n_lon": int(pred.lon.size),
                     "lon_fixed": pred.lon_fixed},
        }
        if write_output:
            os.makedirs(outdir, exist_ok=True)
            csv = os.path.join(outdir, "%s.csv" % name)
            df.to_csv(csv, float_format="%.6g")
            with open(os.path.join(outdir, "%s_meta.json" % name), "w",
                      encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        if verbose:
            print("\n=== 台风检验（%s %s，起报 %s UTC）===" % (tcid, tcname,
                                                              pred.init_date))
            cols = ["fcst_lat", "fcst_lon", "obs_lat", "obs_lon",
                    "track_err_km", "at_km", "ct_km", "fcst_vmax_ms",
                    "obs_vmax_ms", "wind_err_ms", "fcst_pmin_hpa",
                    "obs_pmin_hpa", "pmin_err_hpa"]
            print(df[cols].round(2).to_string())
            if write_output:
                print("\n已写出: %s" % csv)
        return df, meta
    finally:
        pred.close()
        combined_path = os.path.join(pred_path, "combined.nc")
        if os.path.isdir(pred_path) and os.path.isfile(combined_path):
            os.remove(combined_path) 


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="台风路径/强度预报误差检验（pred 场诊断 vs babj 实况）")
    ap.add_argument("--pred", help="预报 nc（需含 msl；强度另需 u10/v10）")
    ap.add_argument("--tcid", help="台风编号（如 2205，对应 babj2205.dat）")
    ap.add_argument("--babj-dir", default="/mnt/d/babj",
                    help="babj 实况目录（默认 /mnt/d/babj）")
    ap.add_argument("--babj", default=None,
                    help="直接指定 babj 文件路径（优先于 --babj-dir）")
    ap.add_argument("--msl-name", default="msl", help="气压 level 名")
    ap.add_argument("--u10-name", default="u10", help="10m u 风 level 名")
    ap.add_argument("--v10-name", default="v10", help="10m v 风 level 名")
    ap.add_argument("--lead-step", type=float, default=None, metavar="H",
                    help="时效重标（小时）：不含 0 时次数据 (1..N)×H")
    ap.add_argument("--center-half", type=float, default=3.0,
                    help="中心搜索框半径（度，默认 3）")
    ap.add_argument("--early-half", type=float, default=4.0,
                    help="初期搜索框半径（度，默认 4）")
    ap.add_argument("--early-hours", type=float, default=12.0,
                    help="初期窗口（小时，默认 12）")
    ap.add_argument("--intensity-half", type=float, default=5.0,
                    help="强度取值框半径（度，默认 5）")
    ap.add_argument("--tz-shift", type=float, default=8.0,
                    help="pred(UTC)→babj(北京时) 时差（默认 8）")
    ap.add_argument("--outdir", default="results/tc")
    ap.add_argument("--name", default=None)
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "netcdf4", "h5py"])
    ap.add_argument("--forecast-type", default="det", choices=["det", "ens"],
                    help="预报类型：确定性 det（默认）或集合 ens")
    ap.add_argument("--ens-agg", default="B", choices=["A", "B"],
                    help="集合误差口径（仅 --forecast-type ens）："
                         "A=成员误差平均（先误差后平均）；"
                         "B=集合平均位置误差（先平均后误差，默认，"
                         "与 recompute_ens_track_error.py 同口径）")
    ap.add_argument("--workers", type=int, default=None, metavar="N",
                    help="集合成员并行进程数（默认=min(CPU核数, 成员数)）")
    ap.add_argument("--no-lon-fix", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.pred or not args.tcid:
        ap.error("需要 --pred 与 --tcid（或 --self-test）")
    babj = args.babj or os.path.join(args.babj_dir, "babj%s.dat" % args.tcid)
    run_tc(args.pred, babj, tcid=args.tcid, outdir=args.outdir,
           name=args.name, engine=args.engine,
           fix_lon=not args.no_lon_fix, lead_step=args.lead_step,
           forecast_type=args.forecast_type, ens_agg=args.ens_agg,
           workers=args.workers,
           msl_name=args.msl_name, u10_name=args.u10_name,
           v10_name=args.v10_name, tz_shift=args.tz_shift,
           center_half=args.center_half, early_half=args.early_half,
           early_hours=args.early_hours, intensity_half=args.intensity_half)
    return 0


if __name__ == "__main__":
    sys.exit(main())
