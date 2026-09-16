"""fdp 参考实现适配层：config → 逐起报 argv → runpy 启动 fdp/ 原脚本。

fdp/ 下的 4 个 verifier 是从示范计划检验包原样拷入的，数据加载、单位
转换、多模型注册、绘图全部不动；本模块只做三件外围事：

1. 把 config 的日期区间展开成逐起报时刻（YYYYMMDDHH，默认每日 00 起报）；
2. 每个起报按 config 拼 argv，用 runpy 以 __main__ 跑一遍对应 verifier
   （单个起报失败只记日志，不中断整个区间；resume=True 时跳过已完成
   起报，靠 staging 下的 <date>.done 标记）；
3. 跑完把逐起报 CSV 拼成长表（加 init_date 列）返回，runner 落盘到
   outputs/results/<output_name>/<capability>.csv。

staging（fdp 原始逐日 CSV + PNG）放 outputs/.temp/<output_name>/，
与 batch 类评测同一套目录约定。
"""
from __future__ import annotations

import re
import runpy
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FDP_DIR = ROOT / "fdp"

#: 起报时刻文件名（fdp 原脚本用 10 位 YYYYMMDDHH 命名输出）
_INIT_RE = re.compile(r"_(\d{10})\.csv$")


def _expand_dates(start_date, end_date, init_hour="00"):
    """日期区间展开成逐起报时刻列表（含首尾）。"""
    if len(init_hour) != 2:
        raise ValueError("init_hour 必须是两位小时字符串，如 '00'/'12'，收到 %r" % init_hour)
    d0 = datetime.strptime(start_date, "%Y%m%d")
    d1 = datetime.strptime(end_date, "%Y%m%d")
    if d1 < d0:
        raise ValueError("end_date(%s) 早于 start_date(%s)" % (end_date, start_date))
    out = []
    d = d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d") + init_hour)
        d += timedelta(days=1)
    return out


def _run_script(script_name, argv):
    """以 __main__ 跑一遍 fdp 原脚本，argv 形同命令行参数。"""
    script = FDP_DIR / script_name
    old_argv = sys.argv
    sys.path.insert(0, str(FDP_DIR))  # tp_deterministic 里 from ensemble_verifier import ...
    sys.argv = [str(script)] + [str(a) for a in argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path.pop(0)


def _run_period(cfg, output_name, script_name, build_argv):
    """逐起报跑 script_name。返回 staging 目录（供后续合并用）。"""
    staging = ROOT / "outputs" / ".temp" / output_name
    staging.mkdir(parents=True, exist_ok=True)
    dates = _expand_dates(cfg["start_date"], cfg["end_date"], cfg.get("init_hour", "00"))
    resume = cfg.get("resume", False)

    n_ok = n_skip = n_fail = 0
    for i, date in enumerate(dates, 1):
        marker = staging / (date + ".done")
        if resume and marker.exists():
            n_skip += 1
            print("[fdp] [%d/%d] %s 已完成，跳过（resume）" % (i, len(dates), date), flush=True)
            continue
        argv = build_argv(date, staging)
        print("\n" + "=" * 60, flush=True)
        print("[fdp] [%d/%d] %s: python %s %s" % (i, len(dates), date, script_name,
                                                  " ".join(str(a) for a in argv[:6])) + " ...", flush=True)
        print("=" * 60, flush=True)
        try:
            _run_script(script_name, argv)
            n_ok += 1
            marker.write_text(date, encoding="utf-8")
        except Exception as exc:
            n_fail += 1
            print("[fdp] ✗ %s 失败: %s" % (date, exc), file=sys.stderr, flush=True)

    print("[fdp] 区间跑完: 成功 %d，跳过 %d，失败 %d（staging: %s）"
          % (n_ok, n_skip, n_fail, staging), flush=True)
    return staging


def _merge(files, extra_col=None):
    """逐起报 CSV 拼长表：init_date 取自文件名尾部的 YYYYMMDDHH。"""
    frames = []
    for fp in sorted(files):
        df = pd.read_csv(fp)
        m = _INIT_RE.search(fp.name)
        df.insert(0, "init_date", m.group(1) if m else "")
        if extra_col:
            df.insert(1, extra_col[0], extra_col[1])
        frames.append(df)
    if not frames:
        raise RuntimeError("staging 下没有任何逐起报 CSV，先看上面失败日志")
    return pd.concat(frames, ignore_index=True)


def _safe(model):
    return model.replace("-", "_").replace(".", "_")


# ======================================================================
# 四个能力：argv 映射 + 长表合并
# ======================================================================

def run_field_det(cfg):
    """确定性场检验（multi_model_verifier_fix.py）：z500/t2m/msl/u10/v10
    的 RMSE、Bias（全部）与 ACC（仅 z500，需气候态距平）。"""
    print("[fdp] run_field_det: RMSE/Bias/ACC, models=%s" % cfg.get("models"), flush=True)
    output_name = cfg.pop("output_name", "fdp_field_det")
    models = cfg["models"]
    hours = cfg.get("forecast_hours", list(range(6, 361, 6)))

    def build(date, staging):
        argv = ["--date", date, "--output-dir", staging,
                "--model-data-root", cfg["model_data_root"],
                "--cra-root", cfg["cra_root"], "--cli-root", cfg["cli_root"]]
        if models:
            argv += ["--models"] + list(models)
        argv += ["--forecast-hours"] + [str(h) for h in hours]
        if cfg.get("surface_only"):
            argv.append("--surface-only")
        if cfg.get("pressure_only"):
            argv.append("--pressure-only")
        return argv

    staging = _run_period(cfg, output_name, "multi_model_verifier_fix.py", build)

    files = []
    for date in _expand_dates(cfg["start_date"], cfg["end_date"], cfg.get("init_hour", "00")):
        for model in models:
            files += staging.glob("%s_*_verification_%s.csv" % (model, date))
    return _merge(files)


def run_ens(cfg):
    """集合检验（ensemble_verifier.py）：CRPS、Spread-Error Ratio、集合平均
    RMSE（全球 z500）+ BSS、AROC（中国区 6h 降水，0.1/4/13/25mm）。

    合并只收逐模型 CSV（ensemble_verification_<model>_<date>.csv），
    原脚本同时写的 multi_model 汇总表内容与之重复，不并进来。"""
    print("[fdp] run_ens: CRPS/SSR/BSS/AROC, models=%s" % cfg.get("models"), flush=True)
    output_name = cfg.pop("output_name", "fdp_ens")
    models = cfg["models"]
    hours = cfg.get("forecast_hours", list(range(6, 43, 6)))

    def build(date, staging):
        argv = ["--date", date, "--output-dir", staging,
                "--fcstdata-root", cfg["fcstdata_root"],
                "--cra-root", cfg["cra_root"],
                "--forecast-type", cfg.get("forecast_type", "ens"),
                "--accum-hours", str(cfg.get("accum_hours", 6))]
        if models:
            argv += ["--models"] + list(models)
        argv += ["--forecast-hours"] + [str(h) for h in hours]
        if cfg.get("forecast_root"):
            argv += ["--forecast-root", cfg["forecast_root"]]
        if cfg.get("obs_rain_root"):
            argv += ["--obs-rain-root", cfg["obs_rain_root"]]
        return argv

    staging = _run_period(cfg, output_name, "ensemble_verifier.py", build)

    files = []
    for date in _expand_dates(cfg["start_date"], cfg["end_date"], cfg.get("init_hour", "00")):
        for model in models:
            files.append(staging / ("ensemble_verification_%s_%s.csv" % (_safe(model), date)))
    return _merge(f for f in files if f.exists())


def run_activity_spectrum(cfg):
    """Z500 活跃度 + 功率谱（activity_spectrum_verifier.py）。

    返回 activity_ratio 长表（runner 落盘）；功率谱长表（行=
    model×时效×{forecast,observation}，列=P0…Pk）另写到同目录
    <output_name>_power_spectrum.csv。"""
    print("[fdp] run_activity_spectrum: activity ratio + power spectrum, models=%s"
          % cfg.get("models"), flush=True)
    output_name = cfg.pop("output_name", "fdp_activity_spectrum")
    models = cfg["models"]
    hours = cfg.get("forecast_hours", list(range(6, 361, 6)))

    def build(date, staging):
        argv = ["--date", date, "--output-dir", staging,
                "--model-data-root", cfg["model_data_root"],
                "--cra-root", cfg["cra_root"], "--cli-root", cfg["cli_root"]]
        if models:
            argv += ["--models"] + list(models)
        argv += ["--forecast-hours"] + [str(h) for h in hours]
        if cfg.get("spectrum_hours"):
            argv += ["--spectrum-hours"] + [str(h) for h in cfg["spectrum_hours"]]
        if cfg.get("debug"):
            argv.append("--debug")
        return argv

    staging = _run_period(cfg, output_name, "activity_spectrum_verifier.py", build)

    dates = _expand_dates(cfg["start_date"], cfg["end_date"], cfg.get("init_hour", "00"))
    ratio_files = [staging / ("activity_ratio_%s.csv" % d) for d in dates]
    spec_files = [staging / ("power_spectrum_%s.csv" % d) for d in dates]

    spec_df = _merge(f for f in spec_files if f.exists())
    spec_path = Path(cfg["outdir"]) / (output_name + "_power_spectrum.csv")
    spec_df.to_csv(spec_path, index=False)
    print("[fdp] 功率谱长表: %s" % spec_path, flush=True)
    return _merge(f for f in ratio_files if f.exists())


def run_tp_det(cfg):
    """确定性降水检验（tp_deterministic_verifier.py）：6h 档 TS/Bias 三档
    + FSS + 综合评分，24h 档 TS/Bias 五档 + 综合评分；站点实况（diamond 3）
    + CMPAS 网格实况（FSS 优先，CRA 回退）。"""
    print("[fdp] run_tp_det: TS/Bias/FSS, models=%s" % cfg.get("models"), flush=True)
    output_name = cfg.pop("output_name", "fdp_tp_det")
    models = cfg["models"]

    def build(date, staging):
        argv = ["--date", date, "--output-dir", staging,
                "--obs-root", cfg["obs_root"],
                "--fcstdata-root", cfg["fcstdata_root"],
                "--cra-root", cfg["cra_root"]]
        if models:
            argv += ["--models"] + list(models)
        if cfg.get("forecast_root"):
            argv += ["--forecast-root", cfg["forecast_root"]]
        if not cfg.get("skip_6h"):
            argv += ["--hours-6h"] + [str(h) for h in
                                      cfg.get("hours_6h", [6, 12, 18, 24, 30, 36, 42, 48])]
        else:
            argv.append("--no-6h")
        if not cfg.get("skip_24h"):
            argv += ["--hours-24h"] + [str(h) for h in
                                       cfg.get("hours_24h",
                                               [24, 48, 72, 96, 120, 144, 168, 192, 216, 240])]
        else:
            argv.append("--no-24h")
        if cfg.get("fss_windows"):
            argv += ["--fss-windows"] + [str(w) for w in cfg["fss_windows"]]
        if cfg.get("fss_threshold") is not None:
            argv += ["--fss-threshold", str(cfg["fss_threshold"])]
        return argv

    staging = _run_period(cfg, output_name, "tp_deterministic_verifier.py", build)

    dates = _expand_dates(cfg["start_date"], cfg["end_date"], cfg.get("init_hour", "00"))
    files_6h = [staging / ("tp_deterministic_6h_%s.csv" % d) for d in dates]
    files_24h = [staging / ("tp_deterministic_24h_%s.csv" % d) for d in dates]
    frames = ([_merge((f for f in files_6h if f.exists()), extra_col=("accum_hours", 6))]
              if any(f.exists() for f in files_6h) else [])
    frames += ([_merge((f for f in files_24h if f.exists()), extra_col=("accum_hours", 24))]
               if any(f.exists() for f in files_24h) else [])
    if not frames:
        raise RuntimeError("staging 下没有 6h/24h CSV，先看上面失败日志")
    return pd.concat(frames, ignore_index=True)
