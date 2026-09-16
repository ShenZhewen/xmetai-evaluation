# -*- coding: utf-8 -*-
# [连续量/regression] 汇总层（批量 mean/summary 落盘）——属主：同 regr_pair
"""集合两个指标的跨日汇总：CRPS 与 离散度-误差比(spread/ensmean RMSE)。

输入：--root 为集合跑法的 outdir_root（含多个 YYYYMMDD 子目录），
每个日期目录内有 crps_<name>.csv / rmse_<name>_ensmean.csv / spread_<name>.csv
（及可选 spread_rmse_ratio_<name>.csv）。

输出（默认写回 root）：
  ens_summary_crps_mean.csv                 行=lead，列=变量，值=跨日均值
  ens_summary_spread_rmse_ratio_mean.csv    同上，spread ÷ ensmean RMSE
  ens_summary_overall.csv                   变量级总结：全时效均值 / 24h / 末尾时效
  ens_summary_crps.png / ens_summary_ratio.png  （matplotlib 可用时）

用法：
  python run_rmse.py --summarize-ens results/fuxi_ens [--out DIR] [--no-plot]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def _lead_at(df, h):
    s = df.index.to_numpy(dtype="f8")
    i = int(np.argmin(np.abs(s - h)))
    return s[i]


def _read_one(path):
    df = pd.read_csv(path, index_col=0)
    df.index.name = "lead_h"
    return df


def summarize_ens(root, out=None, plot=True, verbose=True):
    root = str(root)
    out = out or root
    if not os.path.isdir(root):
        raise ValueError("--root 目录不存在: %s" % root)
    os.makedirs(out, exist_ok=True)

    dates = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and re.fullmatch(r"\d{8}", d))
    if not dates:
        raise ValueError("%s 下没有 YYYYMMDD 子目录" % root)

    crps_frames, ratio_frames = [], []
    per_date = []
    for d in dates:
        dd = os.path.join(root, d)
        crps = sorted(glob.glob(os.path.join(dd, "crps_*.csv")))
        if not crps:
            continue
        name = os.path.basename(crps[0])[len("crps_"):-4]
        try:
            df_c = _read_one(crps[0])
        except Exception:
            continue
        rmse = os.path.join(dd, "rmse_%s_ensmean.csv" % name)
        spr = os.path.join(dd, "spread_%s.csv" % name)
        ratio_p = os.path.join(dd, "spread_rmse_ratio_%s.csv" % name)
        df_r = None
        if os.path.exists(ratio_p):
            try:
                df_r = _read_one(ratio_p)
            except Exception:
                df_r = None
        if df_r is None and os.path.exists(rmse) and os.path.exists(spr):
            try:
                a = _read_one(rmse)
                b = _read_one(spr)
                with np.errstate(divide="ignore", invalid="ignore"):
                    df_r = b / a
                df_r.index.name = "lead_h"
            except Exception:
                df_r = None
        if df_r is not None:
            ratio_frames.append(df_r)
        crps_frames.append(df_c)
        per_date.append(d)

    if not crps_frames:
        raise ValueError("%s 下没有任何含 crps_*.csv 的日期目录" % root)

    def _mean_of(frames):
        cols = sorted({c for f in frames for c in f.columns})
        if not cols:
            return pd.DataFrame()
        concat = pd.concat([f.reindex(columns=cols) for f in frames], axis=0)
        concat.index = pd.Index([str(x) for x in concat.index])
        mean = concat.groupby(level=0).mean()
        mean.index = pd.to_numeric(mean.index, errors="coerce")
        mean = mean.sort_index()
        mean.index.name = "lead_h"
        return mean

    crps_mean = _mean_of(crps_frames)
    ratio_mean = _mean_of(ratio_frames) if ratio_frames else pd.DataFrame()

    def _w(df, fn):
        df.to_csv(os.path.join(out, fn), float_format="%.6g")

    if crps_mean.size:
        _w(crps_mean, "ens_summary_crps_mean.csv")
    if ratio_mean.size:
        _w(ratio_mean, "ens_summary_spread_rmse_ratio_mean.csv")

    # 变量级总结
    rows = []
    for v in sorted(set(crps_mean.columns) | set(ratio_mean.columns)):
        row = {"var": v, "n_dates_crps": int(crps_mean[v].notna().sum())
               if v in crps_mean.columns else 0}
        if v in crps_mean.columns and crps_mean[v].notna().any():
            s = crps_mean[v]
            row["crps_all_lead_mean"] = float(s.mean())
            row["crps_lead24"] = float(s.loc[_lead_at(s, 24.0)])
            row["crps_last_lead"] = float(s.iloc[-1])
        if v in ratio_mean.columns and ratio_mean[v].notna().any():
            s = ratio_mean[v]
            row["ratio_all_lead_mean"] = float(s.mean())
            row["ratio_lead24"] = float(s.loc[_lead_at(s, 24.0)])
            row["ratio_last_lead"] = float(s.iloc[-1])
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, "ens_summary_overall.csv"),
                                  index=False, float_format="%.6g")

    meta = {"root": str(root), "out": str(out),
            "n_dates_total": len(dates), "n_dates_used_crps": len(crps_frames),
            "n_dates_used_ratio": len(ratio_frames), "dates": dates}
    with open(os.path.join(out, "ens_summary_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            plot = False
    if plot:
        _plot_summary(crps_mean, ratio_mean, out)

    if verbose:
        print("[ens-summary] 日期: %d（crps %d / ratio %d）"
              % (len(dates), len(crps_frames), len(ratio_frames)))
        print("  -> %s" % os.path.join(out, "ens_summary_crps_mean.csv"))
        print("  -> %s" % os.path.join(out,
                                       "ens_summary_spread_rmse_ratio_mean.csv"))
        print("  -> %s" % os.path.join(out, "ens_summary_overall.csv"))
    return 0


def _plot_summary(crps_mean, ratio_mean, out):
    import matplotlib.pyplot as plt

    def _fig(df, title, fn, ylog=False, hline=None):
        if df is None or not df.size:
            return
        vars_ = list(df.columns)
        fig, axes = plt.subplots(1, len(vars_), squeeze=False,
                                 figsize=(4.2 * len(vars_), 3.4))
        for ax, v in zip(axes[0], vars_):
            s = df[v]
            ax.plot(s.index.to_numpy(), s.to_numpy(), lw=1.8, marker=".")
            if hline is not None:
                ax.axhline(hline, color="0.6", ls="--", lw=1)
            ax.set_title(v, fontsize=11)
            ax.set_xlabel("lead (h)", fontsize=9)
            ax.grid(True, alpha=0.3)
            if ylog:
                ax.set_yscale("log")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out, fn), dpi=150)
        plt.close(fig)

    _fig(crps_mean, "CRPS (cross-date mean, ensemble)", "ens_summary_crps.png")
    _fig(ratio_mean, "spread/RMSE ratio (cross-date mean; ideal = 1)",
         "ens_summary_ratio.png", hline=1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="集合两指标汇总：CRPS + spread/RMSE 比")
    ap.add_argument("--summarize-ens", metavar="ROOT", default=None,
                    help="集合 outdir_root（含 YYYYMMDD 子目录）")
    ap.add_argument("--out", default=None, help="输出目录（默认=root）")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.summarize_ens:
        ap.error("需要 --summarize-ens <root>")
    summarize_ens(args.summarize_ens, out=args.out, plot=not args.no_plot,
                  verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


# ============================================================ deterministic single

def _mean_of_frames(frames):
    cols = sorted({c for f in frames for c in f.columns})
    if not cols:
        return pd.DataFrame()
    concat = pd.concat([f.reindex(columns=cols) for f in frames], axis=0)
    concat.index = pd.Index([str(x) for x in concat.index])
    m = concat.groupby(level=0).mean()
    m.index = pd.to_numeric(m.index, errors="coerce")
    return m.sort_index()


def _pick_metric_file(dd, stem, name):
    for cand in ("%s_%s_det.csv" % (stem, name), "%s_%s.csv" % (stem, name)):
        p = os.path.join(dd, cand)
        if os.path.exists(p):
            return p
    for fn in sorted(glob.glob(os.path.join(dd, stem + "_*"))):
        if fn.endswith("_ensmean.csv") or fn.endswith("_ens.csv"):
            continue
        return fn
    return None


def _lead_tables(df):
    df.index.name = "lead_h"
    return df


def summarize_det(root, out=None, plot=True, verbose=True):
    """确定性 single 结果汇总：RMSE / 功率谱 / ACC / FA。

    root 为 outdir_root（含 YYYYMMDD 子目录），每日期目录内有
    rmse_<date>_det.csv / acc_<date>_det.csv / fa_<date>_det.csv /
    spectrum_<date>_<var>.csv（也支持 run_rmse 单对的 rmse_<date>.csv 无 _det 命名）。
    """
    root = str(root)
    out = out or root
    if not os.path.isdir(root):
        raise ValueError("--summarize-det 目录不存在: %s" % root)
    os.makedirs(out, exist_ok=True)
    dates = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and re.fullmatch(r"\d{8}", d))
    if not dates:
        raise ValueError("%s 下没有 YYYYMMDD 子目录" % root)

    rmse_f, acc_f = [], []
    fa_kind = {"pred": [], "obs": [], "bias": [], "ratio": []}
    spec = {}
    used = []

    for d in dates:
        dd = os.path.join(root, d)
        hit = False
        for stem, bucket in (("rmse", rmse_f), ("acc", acc_f)):
            p = _pick_metric_file(dd, stem, d)
            if p:
                try:
                    bucket.append(_read_one(p))
                    hit = True
                except Exception:
                    pass
        p = _pick_metric_file(dd, "fa", d)
        if p and os.path.exists(p):
            try:
                fa = _read_one(p)
            except Exception:
                fa = None
            if fa is not None and fa.size:
                hit = True
                kinds = {}
                for c in fa.columns:
                    for suf in ("_pred", "_obs", "_bias", "_ratio"):
                        if c.endswith(suf):
                            kinds.setdefault(c[:-len(suf)], set()).add(suf)
                            break
                for v, sufs in kinds.items():
                    for kind in ("pred", "obs", "bias", "ratio"):
                        col = v + "_" + kind
                        if col in fa.columns:
                            fa_kind[kind].append(pd.DataFrame({v: fa[col]}))
                for kind, fn in (("bias", lambda a_, b_: a_ - b_),
                                 ("ratio", lambda a_, b_: a_ / b_.replace(0.0, np.nan))):
                    missing = [v for v, sufs in kinds.items()
                               if kind not in sufs and "pred" in sufs and "obs" in sufs]
                    if missing:
                        fa_kind[kind].append(pd.DataFrame(
                            {v: fn(fa[v + "_pred"], fa[v + "_obs"]) for v in missing}))
        for vp in sorted(glob.glob(os.path.join(dd, "spectrum_%s_*.csv" % d))):
            v = os.path.basename(vp)[len("spectrum_%s_" % d):-4]
            try:
                sp = _read_one(vp)
            except Exception:
                continue
            predc = next((c for c in sp.columns
                          if c in ("det_pred", "pred", "ensmean_pred")), None)
            obsc = next((c for c in sp.columns
                         if c in ("det_obs", "obs", "ensmean_obs")), None)
            if predc is None or obsc is None:
                continue
            spec.setdefault(v, []).append(
                (sp.index.to_numpy(), sp[predc], sp[obsc]))
            hit = True
        if hit:
            used.append(d)

    if not (rmse_f or acc_f or any(fa_kind.values()) or spec):
        raise ValueError("%s 下没有任何确定性结果文件" % root)

    def _w(df, fn):
        df.to_csv(os.path.join(out, fn), float_format="%.6g")

    lead_frames = {"rmse": rmse_f, "acc": acc_f}
    for kind, frames in fa_kind.items():
        lead_frames["fa_" + kind] = frames
    for tag, frames in lead_frames.items():
        m = _mean_of_frames(frames)
        if m.size:
            m.index.name = "lead_h"
            _w(m, "det_summary_%s_mean.csv" % tag)

    spec_vars = sorted(spec)
    for v in spec_vars:
        rows = spec[v]
        idx = rows[0][0]
        pr = pd.concat([pd.Series(r[1].to_numpy(), index=idx)
                        for r in rows], axis=1).mean(axis=1)
        ob = pd.concat([pd.Series(r[2].to_numpy(), index=idx)
                        for r in rows], axis=1).mean(axis=1)
        m = pd.DataFrame({"pred_mean": pr, "obs_mean": ob},
                         index=pd.Index(idx, name="wavenumber"))
        _w(m, "det_summary_spectrum_%s.csv" % v)

    def _mean_of_var(frames, v):
        vals = []
        for f in frames:
            if v in f.columns:
                vals.append(f[v])
        if not vals:
            return None
        return pd.concat([pd.Series(s.to_numpy(), index=s.index)
                          for s in vals], axis=1).mean(axis=1, skipna=True)

    all_vars = set()
    for frames in lead_frames.values():
        for f in frames:
            all_vars |= set(f.columns)
    all_vars |= set(spec_vars)
    rows_all = []
    for v in sorted(all_vars):
        row = {"var": v}
        for tag, frames in (("rmse", rmse_f), ("acc", acc_f)):
            s = _mean_of_var(frames, v)
            if s is not None and s.notna().any():
                row[tag + "_all_lead_mean"] = float(s.mean())
                i24 = int(np.argmin(np.abs(s.index.to_numpy(dtype="f8") - 24)))
                row[tag + "_lead24"] = float(s.iloc[i24])
                row[tag + "_last_lead"] = float(s.iloc[-1])
        for kind, frames in fa_kind.items():
            s = _mean_of_var(frames, v)
            if s is not None and s.notna().any():
                row["fa_" + kind + "_all_lead_mean"] = float(s.mean())
        if v in spec_vars:
            row["spectrum_has"] = 1
        rows_all.append(row)
    if rows_all:
        pd.DataFrame(rows_all).to_csv(
            os.path.join(out, "det_summary_overall.csv"),
            index=False, float_format="%.6g")

    meta = {"root": str(root), "out": str(out),
            "n_dates_total": len(dates), "n_dates_used": len(used),
            "variables": sorted(all_vars), "dates": dates}
    with open(os.path.join(out, "det_summary_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            plot = False
    if plot:
        _plot_det(lead_frames, spec, out)

    if verbose:
        print("[det-summary] 日期: %d（用了 %d）" % (len(dates), len(used)))
        for tag in lead_frames:
            print("  -> %s" % os.path.join(out, "det_summary_%s_mean.csv" % tag))
        print("  -> %s" % os.path.join(out, "det_summary_overall.csv"))
    return 0
def _plot_det(lead_frames, spec, out):
    import matplotlib.pyplot as plt

    for tag in ("rmse", "acc"):
        frames = lead_frames.get(tag, [])
        m = _mean_of_frames(frames)
        if not m.size:
            continue
        vars_ = list(m.columns)
        fig, axes = plt.subplots(1, len(vars_), squeeze=False,
                                 figsize=(4.2 * len(vars_), 3.4))
        for ax, v in zip(axes[0], vars_):
            s = m[v]
            ax.plot(s.index.to_numpy(), s.to_numpy(), lw=1.8, marker=".")
            ax.set_title(v, fontsize=11)
            ax.set_xlabel("lead (h)", fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle("%s (cross-date mean)" % tag.upper(), fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out, "det_summary_%s.png" % tag), dpi=150)
        plt.close(fig)

    m_fa = {kind: _mean_of_frames(frames)
            for kind, frames in lead_frames.items() if kind.startswith("fa_")}
    if m_fa.get("fa_pred", None) is not None and m_fa["fa_pred"].size:
        pred = m_fa["fa_pred"]
        obs = m_fa.get("fa_obs")
        vars_ = list(pred.columns)
        fig, axes = plt.subplots(1, len(vars_), squeeze=False,
                                 figsize=(4.2 * len(vars_), 3.4))
        for ax, v in zip(axes[0], vars_):
            ax.plot(pred[v].index.to_numpy(), pred[v].to_numpy(),
                    lw=1.6, label="FA_pred")
            if obs is not None and v in obs.columns:
                ax.plot(obs[v].index.to_numpy(), obs[v].to_numpy(),
                        lw=1.6, ls="--", label="FA_obs")
            ax.set_title(v, fontsize=11)
            ax.set_xlabel("lead (h)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("FA pred/obs (cross-date mean)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out, "det_summary_fa.png"), dpi=150)
        plt.close(fig)

    if spec:
        vars_ = sorted(spec)
        fig, axes = plt.subplots(1, len(vars_), squeeze=False,
                                 figsize=(4.6 * len(vars_), 3.6))
        for ax, v in zip(axes[0], vars_):
            rows = spec[v]
            idx = rows[0][0]
            pr = pd.concat([pd.Series(r[1].to_numpy(), index=idx)
                            for r in rows], axis=1).mean(axis=1)
            ob = pd.concat([pd.Series(r[2].to_numpy(), index=idx)
                            for r in rows], axis=1).mean(axis=1)
            ax.loglog(idx, pr, lw=1.6, label="pred")
            ax.loglog(idx, ob, lw=1.6, ls="--", label="obs")
            ax.set_title(v, fontsize=11)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("power spectrum (cross-date mean)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out, "det_summary_spectrum.png"), dpi=150)
        plt.close(fig)


def main_det(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="\u786e\u5b9a\u6027 single \u56db\u6307\u6807\u6c47\u603b\uff1aRMSE/\u8c31/ACC/FA")
    ap.add_argument("--summarize-det", metavar="ROOT", default=None,
                    help="\u786e\u5b9a\u6027 outdir_root\uff08\u542b YYYYMMDD \u5b50\u76ee\u5f55\uff09")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.summarize_det:
        ap.error("\u9700\u8981 --summarize-det <root>")
    summarize_det(args.summarize_det, out=args.out, plot=not args.no_plot,
                  verbose=not args.quiet)
    return 0
