#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
step1 ~ step5 流程编排器（集中配置版）
================================================================================

【用途】
  用户只需要修改本文件顶部的 CONFIG，变量名 / 年份 / 所有输入输出路径都集中在这里；
  然后本编排器把这些配置通过命令行参数传给 step1~step5，各脚本不再写死路径。

【目录逻辑】
  原始观测  step1_data/{year}/...                        → step1
  原始模式  step2_data/{year}{mmdd}.nc                   → step2
      ↓
  观测 60 天距平  {obs_combine}/{year}/{year}{mmdd}.nc
  模式 60 天距平  {model_anom}/{year}{mmdd}.nc
      ↓ step3（两个任务）
  观测周平均  {weekly_cra}
  模式周平均  {weekly_fengshun}
      ↓ step4 / step5
  柱状图 + 空间图（step4_output / step5_output）

【运行示例】
  python step1_5_流程编排.py
  python step1_5_流程编排.py --steps step1 step2 step3
  python step1_5_流程编排.py --var t2m --year 2024

作者：FDP Project（编排版）
日期：2026-08-28
================================================================================
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          USER  CONFIG（用户只改这里）                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PROJECT_ROOT = Path(__file__).resolve().parent

STEP1 = PROJECT_ROOT / "step1_观测数据处理.py"
STEP2 = PROJECT_ROOT / "step2_模型数据处理.py"
STEP3 = PROJECT_ROOT / "step3_模型和观测数据处理_step3（周平均）.py"
STEP4 = PROJECT_ROOT / "step4_绘图_tcc_rmse_bar_柱状图代码.py"
STEP5 = PROJECT_ROOT / "step5_绘图_tcc_rmse_空间图.py"

# ── 集中配置：变量、年份、所有输入输出路径 ────────────────────────────────────
CONFIG = {
    "var": "t2m",    
    "year": 2024,               #周平均/绘图用哪一年

    # step1 观测数据处理
    "obs_raw":     "/gpu/zhouchg/zhouchg/CRA/output/{var}",         ##输入观测数据路径##     # 观测原始数据根目录  step1从这里读，文件目录格式为 /gpu/zhouchg/zhouchg/CRA/output/{var}   {year}/{year}{mmdd}.nc

    "obs_clim":    "/gpu/zhouchg/liuzh/CRA_1.5/clim_1p5/{var}",              #观测气候态输出目录（0.25°，逐日，365 个文件）
    "obs_diff":    "/gpu/zhouchg/liuzh/CRA_1.5/diff_1p5/{var}",              #观测逐日距平输出目录（0.25°）
    "obs_anom":    "/gpu/zhouchg/liuzh/CRA_1.5/anom_1p5/{var}",              #观测插值到 1.5° 的逐日距平输出目录
    "obs_combine": "/gpu/zhouchg/liuzh/CRA_1.5/anom_combine_1p5/{var}",      #观测合并 60 天（lead_time=60）的距平输出目录；这是 step3 的输入
    "clim_years":  [2004, 2023],                                #观测气候态基准年 [起始, 结束]，当前 [2004, 2023]
    "diff_years":  [2024, 2025],                                #观测距平年份列表，当前 [2024, 2025]
    "lead_time":   60,                                          #合并的未来天数，当前 60

    # step2 模式数据处理(将模式数据里的member 平均（11 个成员平均成 1 个）
    "model_raw":   "/gpu/zhouchg/wangchp/FDP/Fengshun_v2.0/t2m",      ##输入我的模式数据路径##    #模式原始数据根目录，step2 从这里读 ，文件目录格式为.../gpu/zhouchg/wangchp/FDP/Fengshun_v2.0/t2m/   {year}{mmdd}.nc
    
    "model_clim":  "/gpu/zhouchg/liuzh/step2_output/clim_1p5/{var}",              #模式气候态输出目录（自带 lead_time=60）
    "model_anom":  "/gpu/zhouchg/liuzh/step2_output/anom_1p5/{var}",              #模式距平输出目录（自带 lead_time=60）；这是 step3 的输入
    "clim_start":  2004,                                        #模式气候态起止年
    "clim_end":    2023,                                        #模式气候态起止年
    "anom_year":   2024,                                        #模式距平年份，当前 2024

    # step3 周平均输出（= step4/5 的输入）
    "weekly_cra":      "/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/{var}",             #观测周平均输出目录，同时是 step4/5 的观测输入
    "weekly_my_model": "/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/{var}",        #模式周平均输出目录，同时是 step4/5 的模式输入

    # step4 柱状图     
    "step4_model_sources": ["my_model", "other_model"],
    "step4_reference":     "CRA_v1.5",  
    "step4_reference_path": "/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
    "step4_my_model_path":  "/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
    "step4_other_model_path": "/gpu/zhouchg/liuzh/module1_other_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",       ##输入被比较的模式数据路径##   
    "step4_output":         "/gpu/zhouchg/liuzh/step4_output",

    # step5 空间图          
    "step5_cra_path":      "/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",                                    #CRA 周平均的路径模板（含 {var} 和 {yyyymmdd}）
    "step5_my_model_path": "/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",                                    #my_model 周平均的路径模板
    "step5_other_model_path":"/gpu/zhouchg/liuzh/module1_other_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",      ##输入被比较的模式数据路径##                  
    "step5_output":        "/gpu/zhouchg/liuzh/step5_output",                                                                                           #step5 输出目录（空间图、统计 NetCDF、CSV）
}

DEFAULT_STEPS = ["step1", "step2", "step3", "step4", "step5"]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          INTERNAL（业务逻辑代码）                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def log(msg: str) -> None:
    """统一日志输出，带时间戳。"""
    from datetime import datetime as _dt
    print(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(script: Path, *args: str) -> None:
    """用当前解释器运行一个脚本，并把配置作为命令行参数传入。"""
    cmd = [sys.executable, str(script), *[str(a) for a in args]]
    log(f"▶ 运行: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    """解析编排器自身参数。"""
    parser = argparse.ArgumentParser(
        description="集中配置并顺序运行 step1~step5。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step1_5_流程编排.py
  python step1_5_流程编排.py --steps step1 step2 step3
  python step1_5_流程编排.py --var t2m --year 2024
        """,
    )
    parser.add_argument(
        "--steps", type=str, nargs="+", default=DEFAULT_STEPS,
        choices=["step1", "step2", "step3", "step4", "step5"],
        help=f"执行步骤（默认: {DEFAULT_STEPS}）",
    )
    parser.add_argument("--var", type=str, default=None, help="覆盖 CONFIG 中的变量名")
    parser.add_argument("--year", type=int, default=None, help="覆盖 CONFIG 中的年份")
    return parser.parse_args()


def main() -> None:
    """主入口：读取 CONFIG，逐脚本传参运行。"""
    args = parse_args()
    c = CONFIG
    var = args.var if args.var is not None else c["var"]
    year = args.year if args.year is not None else c["year"]

    log("=" * 60)
    log("step1 ~ step5 流程编排（集中配置版）")
    log(f"  变量: {var}, 年份: {year}")
    log(f"  步骤: {args.steps}")
    log("=" * 60)

    if "step1" in args.steps:
        run(
            STEP1,
            "--var", var,
            "--raw-dir", c["obs_raw"].format(var=var),
            "--clim-dir", c["obs_clim"].format(var=var),
            "--diff-dir", c["obs_diff"].format(var=var),
            "--interp-dir", c["obs_anom"].format(var=var),
            "--combine-dir", c["obs_combine"].format(var=var),
            "--clim-years", str(c["clim_years"][0]), str(c["clim_years"][1]),
            "--diff-years", *[str(y) for y in c["diff_years"]],
            "--lead-time", str(c["lead_time"]),
        )

    if "step2" in args.steps:
        run(
            STEP2,
            "--var", var,
            "--output-var", var,
            "--raw-dir", c["model_raw"],
            "--clim-dir", c["model_clim"].format(var=var),
            "--anom-dir", c["model_anom"].format(var=var),
            "--clim-start", str(c["clim_start"]),
            "--clim-end", str(c["clim_end"]),
            "--anom-year", str(c["anom_year"]),
        )

    if "step3" in args.steps:
        run(
            STEP3,
            "--var", var,
            "--year", str(year),
            "--obs-input", f"{c['obs_combine'].format(var=var)}/{year}",
            "--obs-output", c["weekly_cra"].format(var=var),
            "--model-input", c["model_anom"].format(var=var),
            "--model-output", c["weekly_my_model"].format(var=var),
        )

    if "step4" in args.steps:
        run(
            STEP4,
            "--variables", var,
            "--year", str(year),
            "--output-dir", c["step4_output"],
            "--model-sources", *c["step4_model_sources"],
            "--reference-source", c["step4_reference"],
            "--reference-path", c["step4_reference_path"],
            "--my-model-path", c["step4_my_model_path"],
            "--other-model-path", c["step4_other_model_path"],
        )

    if "step5" in args.steps:
        run(
            STEP5,
            "--vars", var,
            "--year-start", str(year),
            "--year-end", str(year),
            "--output-dir", c["step5_output"],
            "--cra-path", c["step5_cra_path"],
            "--my-model-path", c["step5_my_model_path"],
            "--other-model-path", c["step5_other_model_path"],
        )

    log("=" * 60)
    log("全部完成。")
    log("=" * 60)


if __name__ == "__main__":
    main()
