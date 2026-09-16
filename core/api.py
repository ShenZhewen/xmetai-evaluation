"""Thin adapters over the copied reference implementations."""
from __future__ import annotations

import importlib.util
import re
import sys
from datetime import time, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

#: 形如 20250610 的起报日目录名。pred 给到这一层就是单场次，给到上一层就是批量的根。
_DATE_DIR = re.compile(r"\d{8}")


def load_ref(filename: str):
    sys.path.insert(0, str(ROOT / "vfc"))
    try:
        spec = importlib.util.spec_from_file_location("ref_" + filename[:-3], ROOT / "core" / filename)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def run_categorical(cfg):
    print(f"[api] Loading categorical reference module", flush=True)
    m = load_ref("categorical_ref.py")
    opts = dict(cfg)
    root = opts.pop("root", None)
    if root is None:
        root = opts.pop("pred", None)
    if root is None:
        raise KeyError("categorical capability requires 'pred' or 'root'")
    station_dir = opts.pop("station_dir")
    opts.pop("outdir", None)
    opts.pop("windows", None)
    opts.pop("lead_step", None)
    print(f"[api] Starting categorical evaluation", flush=True)
    print(f"[api]   root: {root}", flush=True)
    print(f"[api]   station_dir: {station_dir}", flush=True)
    print(f"[api]   output: {cfg['outdir']}", flush=True)
    return m.run_categorical(root=root, station_dir=station_dir,
                             outdir=cfg["outdir"], verbose=True, **opts)


def run_ts_det(cfg):
    print(f"[api] run_ts_det: Deterministic TS evaluation (24h, thresholds 0.1/10/25/50/100/250mm)", flush=True)
    opts = dict(cfg)
    opts["root"] = opts.pop("pred")
    opts["window_ts"] = (24.0,)
    opts["window_aroc"] = ()
    # 档位与 grade 写法对齐参考结果 ts_fgvp_2025.csv：含 250mm 特大暴雨档，
    # grade 列写 "≥0.1"/"≥250"（超越式），不是裸数值。
    opts["thresholds"] = {24.0: [(f"≥{v}", float(v)) for v in (0.1, 10, 25, 50, 100, 250)]}
    return run_categorical(opts)


def run_ts_ens(cfg):
    print(f"[api] run_ts_ens: Ensemble TS evaluation (24h, thresholds 0.1/10/25/50/100/250mm)", flush=True)
    opts = dict(cfg)
    opts["root"] = opts.pop("pred")
    opts["window_ts"] = (24.0,)
    opts["window_aroc"] = ()
    opts["thresholds"] = {24.0: [(f"≥{v}", float(v)) for v in (0.1, 10, 25, 50, 100, 250)]}
    return run_categorical(opts)


def run_ts_ens_prob(cfg):
    print(f"[api] run_ts_ens_prob: Ensemble probability evaluation (6h, thresholds 0.1/4/13/25mm)", flush=True)
    opts = dict(cfg)
    opts["root"] = opts.pop("pred")
    opts["window_ts"] = ()
    opts["window_aroc"] = (6.0,)
    opts["aroc_thresholds"] = {6.0: [(str(v), float(v)) for v in (0.1, 4, 13, 25)]}
    return run_categorical(opts)


def run_field(cfg):
    print(f"[api] run_field: Deterministic field scores (RMSE, ACC, FA, spectra)", flush=True)
    sys.path.insert(0, str(ROOT))
    try:
        from vfc.regr_pair import verify_pair
        opts = dict(cfg)
        obs = opts.pop("obs")
        pred = opts.pop("pred")
        opts.pop("outdir", None)

        # Map config names to verify_pair parameter names
        if "climo" in opts:
            opts["climo_path"] = opts.pop("climo")

        print(f"[api]   obs: {obs}", flush=True)
        print(f"[api]   pred: {pred}", flush=True)
        return verify_pair(obs, pred, outdir=cfg["outdir"], plot=False, verbose=True, **opts)
    finally:
        sys.path.pop(0)


def run_ensemble(cfg, metrics):
    print(f"[api] run_ensemble: metrics={metrics}", flush=True)
    sys.path.insert(0, str(ROOT))
    try:
        from vfc.regr_ens import run_ensemble as ref_run
        opts = dict(cfg); opts.pop("outdir", None); pred = opts.pop("pred"); target = opts.pop("target")
        print(f"[api]   pred: {pred}", flush=True)
        print(f"[api]   target: {target}", flush=True)
        return ref_run(pred, target, metrics=metrics, outdir=cfg["outdir"], plot=False, verbose=True, **opts)
    finally:
        sys.path.pop(0)


def run_ens_crps(cfg):
    print(f"[api] run_ens_crps: Ensemble CRPS, spread, RMSE", flush=True)
    return run_ensemble(cfg, ["crps", "spread", "rmse"])


def run_ens_field(cfg):
    print(f"[api] run_ens_field: Ensemble field scores (RMSE, CRPS, ACC, FA, spectrum)", flush=True)
    return run_ensemble(cfg, ["rmse", "crps", "acc", "fa", "spectrum"])


def run_tc(cfg):
    """台风检验。pred/babj 指到哪一层决定跑一场还是跑一批：

    * ``pred`` 给一份 nc 文件或一个 ``<YYYYMMDD>`` 起报日目录 → 单场次（老行为）；
    * ``pred`` 给预报根目录（下面是一堆 ``<YYYYMMDD>``）→ 批量，逐日跑。

    ``babj`` 给单个 ``babj<编号>.dat`` → 只跑那一号台风；给目录 → 目录下全部
    ``babj*.dat`` 都跑。批量时 ``tcid`` 可省，给了就只跑该编号。
    """
    print(f"[api] run_tc: Typhoon track evaluation", flush=True)
    m = load_ref("tc_ref.py")
    opts = dict(cfg)
    pred = opts.pop("pred")
    babj = opts.pop("babj")
    tcid = opts.pop("tcid", None)
    opts.pop("outdir", None)

    # Remove config parameters that run_tc doesn't accept
    opts.pop("start_date", None)
    opts.pop("end_date", None)

    print(f"[api]   pred: {pred}", flush=True)
    print(f"[api]   babj: {babj}", flush=True)

    pred_dir = Path(pred)
    if not (pred_dir.is_dir() and not _DATE_DIR.fullmatch(pred_dir.name)):
        if not tcid:
            raise KeyError("单场次台风检验必须给 'tcid'（pred=%s）" % pred)
        print(f"[api]   tcid: {tcid}", flush=True)
        return m.run_tc(pred, babj, tcid, outdir=cfg["outdir"], verbose=True,
                        **opts)
    return _run_tc_batch(m, pred_dir, babj, tcid, cfg["outdir"], opts)


def _run_tc_batch(m, pred_root, babj, tcid, outdir, opts):
    """pred 给预报根目录时批量跑：babj 目录下每个文件 × 每个有预报的起报日。

    与参考实现的 ``run_tc_all.sh`` 同口径，但起报日不再拿 awk 从 babj 里猜，
    而是直接由实况时刻反推（``起报时刻 = 实况时刻 - tz_shift``），所以不会
    先跑一遍再靠报错筛掉。单个场次失败不影响后续场次。

    每场次仍各自写出 ``tc<编号>_<起报时刻>.csv`` 与 ``_meta.json``；
    返回值是各场次拼起来的汇总表（``runner.py`` 会写成 ``typhoon.csv``）。
    """
    tz_shift = float(opts.get("tz_shift", 8.0))
    babj_path = Path(babj)
    if babj_path.is_dir():
        babj_files = sorted(babj_path.glob("babj*.dat"))
        if not babj_files:
            raise FileNotFoundError("%s 下没有 babj*.dat" % babj_path)
    else:
        if not tcid:
            raise KeyError("babj 给单个文件时必须给 'tcid'")
        babj_files = [babj_path]
    print("[api]   批量模式: %d 个 babj 文件 × 预报根 %s"
          % (len(babj_files), pred_root), flush=True)

    frames = []
    n_ok = n_fail = n_skip = n_filtered = 0
    for f in babj_files:
        try:
            tcname, tcid_file, analyses = m.read_babj_analyses(str(f))
        except Exception as exc:
            n_fail += 1
            print("[api]   ✗ %s 读取失败: %s" % (f.name, exc), flush=True)
            continue
        if tcid and str(tcid) != str(tcid_file):
            n_filtered += 1
            continue
        # combined.nc 把起报时刻固定写成当天 00 UTC（_comb_netcdf4_fuxi 写死
        # "hours since <date> 00:00:00"），所以只有整点 00 UTC 的实况能对上
        # 一个 <YYYYMMDD> 预报目录；别的时刻对不上，跳过而不是等它报错。
        inits = sorted(t - timedelta(hours=tz_shift) for t in analyses
                       if (t - timedelta(hours=tz_shift)).time() == time(0, 0))
        print("[api] %s（%s %s）: %d 个起报时刻"
              % (f.name, tcid_file, tcname, len(inits)), flush=True)

        for init_utc in inits:
            date = init_utc.strftime("%Y%m%d")
            day_dir = pred_root / date
            if not day_dir.is_dir():
                n_skip += 1
                continue
            try:
                df, _meta = m.run_tc(str(day_dir), str(f), tcid_file,
                                     outdir=outdir, verbose=False, **opts)
            except Exception as exc:
                n_fail += 1
                print("[api]   ✗ %s %s: %s" % (tcid_file, date, exc), flush=True)
                continue
            n_ok += 1
            # 确定性场次叫 track_err_km，集合汇总表叫 track_err_mean_km
            col = ("track_err_km" if "track_err_km" in df.columns
                   else "track_err_mean_km")
            err = df[col].dropna()
            print("[api]   ✓ %s %s 起报 %s UTC  时效 %d，匹配 %d，首时效误差 %s km"
                  % (tcid_file, tcname, init_utc, len(df), len(err),
                     "%.1f" % err.iloc[0] if len(err) else "—"), flush=True)
            out = df.reset_index()
            out.insert(0, "tcid", str(tcid_file))
            out.insert(1, "tcname", tcname)
            out.insert(2, "init_utc", str(init_utc))
            frames.append(out)

    if n_filtered:
        print("[api]   （另有 %d 个 babj 文件被 config 里的 tcid=%s 过滤掉，"
              "删掉该行即全部跑）" % (n_filtered, tcid), flush=True)
    print("[api] 批量完成: 成功 %d，失败 %d，跳过（该日无预报目录）%d"
          % (n_ok, n_fail, n_skip), flush=True)
    if not frames:
        raise RuntimeError("没有一个场次成功（成功 0，失败 %d，跳过 %d）"
                           % (n_fail, n_skip))
    return pd.concat(frames, ignore_index=True)
