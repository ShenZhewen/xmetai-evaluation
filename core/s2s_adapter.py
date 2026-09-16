"""s2s 确定性预测适配层：config → 逐步 argv → runpy 启动 s2s/ 原脚本。

s2s/ 下的 5 个 step 脚本（step1 观测距平 → step2 模式距平 → step3 周平均
→ step4 柱状图 TCC/RMSE → step5 空间图 TCC/p-value/RMSE）是从示范计划
"确定性预测" 检验包原样拷入的，气候态/距平/周平均/指标计算全部不动；
本模块只做外围三件事：

1. 按 config["steps"] 选步，逐步拼 argv，用 runpy 以 __main__ 跑一遍对应
   脚本（顺序固定 step1→5，后面的步骤依赖前面的产物）；
2. 各步骤自带断点续跑（输出已存在即跳过），气候态（step1 CLIM、step2
   CLIM）只依赖变量+基准年+数据源，算一次永久复用，换变量/换年才重算；
3. 跑完收集 step4/step5 的区域加权平均 CSV，拼成带 table 列的长表返回，
   runner 落盘到 outputs/results/<output_name>/<capability>.csv。

staging（step4/5 的 metrics/figures 等中间产物）放
outputs/.temp/<output_name>/step{4,5}_output/。
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S2S_DIR = ROOT / "s2s"

#: 5 个步骤的执行顺序（后面的步骤依赖前面的产物，不能乱序）
STEP_ORDER = ["step1", "step2", "step3", "step4", "step5"]


def _run_script(script_name, argv):
    """以 __main__ 跑一遍 s2s 原脚本，argv 形同命令行参数。"""
    script = S2S_DIR / script_name
    old_argv = sys.argv
    sys.path.insert(0, str(S2S_DIR))
    sys.argv = [str(script)] + [str(a) for a in argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path.pop(0)


def _ordered_steps(cfg):
    """steps 列表按固定顺序执行；未配置默认全流程。"""
    steps = cfg.get("steps") or list(STEP_ORDER)
    unknown = [s for s in steps if s not in STEP_ORDER]
    if unknown:
        raise ValueError("steps 含未知步骤 %s，可选 %s" % (unknown, STEP_ORDER))
    return [s for s in STEP_ORDER if s in steps]


def _build_argv(step, cfg, staging):
    """单个步骤的 argv（对齐原 step1_5_流程编排.py 的传参方式）。"""
    var = cfg["var"]
    year = cfg.get("year")

    if step == "step1":
        clim_years = cfg.get("clim_years", [2004, 2023])
        return [
            "--var", var,
            "--raw-dir", cfg["obs_raw"],
            "--clim-dir", cfg["obs_clim"],
            "--diff-dir", cfg["obs_diff"],
            "--interp-dir", cfg["obs_anom"],
            "--combine-dir", cfg["obs_combine"],
            "--clim-years", str(clim_years[0]), str(clim_years[1]),
            "--diff-years", *[str(y) for y in cfg.get("diff_years", [year])],
            "--lead-time", str(cfg.get("lead_time", 60)),
        ]

    if step == "step2":
        return [
            "--var", var,
            "--output-var", var,
            "--raw-dir", cfg["model_raw"],
            "--clim-dir", cfg["model_clim"],
            "--anom-dir", cfg["model_anom"],
            "--clim-start", str(cfg.get("clim_start", 2004)),
            "--clim-end", str(cfg.get("clim_end", 2023)),
            "--anom-year", str(cfg.get("anom_year", year)),
        ]

    if step == "step3":
        # step3 输入 = step1/step2 的输出：观测 combine 目录再进一层年份
        return [
            "--var", var,
            "--year", str(year),
            "--obs-input", "%s/%s" % (cfg["obs_combine"], year),
            "--obs-output", cfg["weekly_obs"],
            "--model-input", cfg["model_anom"],
            "--model-output", cfg["weekly_model"],
        ]

    step4_out = cfg.get("step4_output", str(staging / "step4_output"))
    step5_out = cfg.get("step5_output", str(staging / "step5_output"))

    if step == "step4":
        argv = [
            "--variables", var,
            "--year", str(year),
            "--output-dir", step4_out,
            "--model-sources", *cfg.get("model_sources", ["my_model", "other_model"]),
            "--reference-source", cfg.get("reference_source", "CRA_v1.5"),
            "--reference-path", cfg["reference_path"],
            "--my-model-path", cfg["my_model_path"],
            "--other-model-path", cfg["other_model_path"],
            "--init-start", cfg.get("init_start", "0101"),
            "--init-end", cfg.get("init_end", "1231"),
        ]
        return argv

    # step5
    return [
        "--vars", var,
        "--year-start", str(cfg.get("year_start", year)),
        "--year-end", str(cfg.get("year_end", year)),
        "--output-dir", step5_out,
        "--cra-path", cfg["reference_path"],
        "--my-model-path", cfg["my_model_path"],
        "--other-model-path", cfg["other_model_path"],
    ] + (["--cartopy-dir", cfg["cartopy_dir"]] if cfg.get("cartopy_dir") else [])


#: 步骤 → 脚本文件名
STEP_SCRIPTS = {
    "step1": "step1_obs_anomaly.py",
    "step2": "step2_model_anomaly.py",
    "step3": "step3_weekly_mean.py",
    "step4": "step4_tcc_rmse_bar.py",
    "step5": "step5_tcc_rmse_maps.py",
}


def run_det(cfg):
    """确定性预测（S2S 周平均距平 TCC/RMSE）检验。"""
    print("[s2s] run_det: steps=%s" % cfg.get("steps", "全流程"), flush=True)
    output_name = cfg.pop("output_name", "s2s_det")
    staging = ROOT / "outputs" / ".temp" / output_name
    staging.mkdir(parents=True, exist_ok=True)

    # 无显示环境（WSL/服务器）下 matplotlib 必须走 Agg，否则 step4/5 画图崩
    os.environ.setdefault("MPLBACKEND", "Agg")

    steps = _ordered_steps(cfg)
    for i, step in enumerate(steps, 1):
        argv = _build_argv(step, cfg, staging)
        print("\n" + "=" * 60, flush=True)
        print("[s2s] [%d/%d] %s: python %s %s" % (
            i, len(steps), step, STEP_SCRIPTS[step],
            " ".join(str(a) for a in argv[:6])) + " ...", flush=True)
        print("=" * 60, flush=True)
        _run_script(STEP_SCRIPTS[step], argv)

    return _collect_tables(cfg, steps, staging)


def _collect_tables(cfg, steps, staging):
    """收集 step4/step5 的区域加权平均 CSV，拼成带 table 列的长表。"""
    var = cfg["var"]
    year = cfg.get("year")
    frames = []

    if "step4" in steps:
        step4_out = Path(cfg.get("step4_output", staging / "step4_output"))
        for metric in ("tcc", "rmse"):
            fp = step4_out / "tables" / "%s_%s_regional_weighted_mean_%s.csv" % (var, metric, year)
            if fp.exists():
                df = pd.read_csv(fp)
                df.insert(0, "table", "step4_bar")
                frames.append(df)
                print("[s2s] 收集 step4 表: %s" % fp, flush=True)

    if "step5" in steps:
        step5_out = Path(cfg.get("step5_output", staging / "step5_output"))
        fp = step5_out / "regional" / var / "%s_regional_weighted_mean.csv" % var
        if fp.exists():
            df = pd.read_csv(fp)
            df.insert(0, "table", "step5_regional")
            frames.append(df)
            print("[s2s] 收集 step5 表: %s" % fp, flush=True)

    if not frames:
        # 只跑了前置步骤（气候态/距平/周平均），没有指标表可合并
        return {"steps_run": steps,
                "note": "无 step4/step5 指标表（前置数据处理步骤，产物在各自输出目录）"}

    return pd.concat(frames, ignore_index=True)
