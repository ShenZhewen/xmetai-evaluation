#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个批次归档目录 → 纬向 FFT 与球谐谱补充分析报告（确定性单成员）。

用法::

    python skills/xmetai-evaluation/scripts/generate_wave_report.py \\
      --model AIFS=outputs/results/weather_rmse_single_aifs/single_aifs \\
      --model FengQing=outputs/results/weather_rmse_single_fengqing/d2d283c6a2618455 \\
      --model FGVP=outputs/results/weather_rmse_single_fgvp/single_fgvp \\
      --model FGVP_pinpu=outputs/results/weather_rmse_single_fgvp_pinpu/dc849c75feb7d393 \\
      --model FuXi=outputs/results/weather_rmse_single_fuxi/single_fuxi \\
      --out report/wave_5models

``--model NAME=目录`` 可重复，至少两个。目录是 ``vfc/regr_ens.py`` 批量跑出来的
**批次归档目录**（``summary.csv`` + ``batch_meta.json`` + ``<YYYYMMDD>/``），
与 ``generate_det_report.py`` 吃的是同一种归档——两条支线的差别只在报告口径。

其余选项：

    --out DIR       报告输出目录（必给，不存在会建）
    --archive TEXT  归档名，写进报告抬头与附录的「归档文件」

**与 `_single` 支线的差别**：纬向谱 log-RMS 这里用 **ln**（骨架 §2.1 的定义），
`generate_det_report.py` 用 **log10**，两者差 ×2.3026，不要互相对数。

**球谐带（§5.1–5.3）的数据来自归档里的 `spherical_bands_<日期>_<变量>.csv`**
（`vfc/regr_ens.py` 跑批量时逐 lead 落盘，列名契约见
`vfc.metrics.spectrum.band_power_frame`）。**老归档没有这两类文件**，
此时报告保留节位并明确标注「本批未出」，不静默省略、也不留空表。

**本脚本只做入口**：解析参数 → 把仓库根加进 ``sys.path`` → 调
``visualization.wave_report.build_wave_report`` → 打印产物清单。
加载、统计、渲染的逻辑**全在** ``visualization/`` 里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: scripts/ -> xmetai-evaluation/ -> skills/ -> 仓库根
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_model(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--model 要写成 NAME=目录，收到的是 {value!r}")
    name, directory = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--model 的模型名不能是空的: {value!r}")
    return name, Path(directory.strip())


def main(argv=None) -> int:
    # Windows 控制台默认 GBK，报告正文和日志里有大量中文。控制台代码页不是
    # UTF-8 时，print 会抛 UnicodeEncodeError——报告和图其实都已经写完了，
    # 却在最后一步非 0 退出。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="按多个批次归档目录出图并写纬向 FFT / 球谐谱补充分析报告",
        epilog="示例: python skills/xmetai-evaluation/scripts/generate_wave_report.py "
        "--model FGVP=outputs/results/weather_rmse_single_fgvp/single_fgvp "
        "--model FGVP_pinpu=outputs/results/weather_rmse_single_fgvp_pinpu/dc849c75feb7d393 "
        "--out report/wave_2models",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="模型的批次归档目录（可重复，至少两个）",
    )
    parser.add_argument("--out", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--archive", default=None,
                        help="归档名，写进报告抬头与附录的「归档文件」")
    args = parser.parse_args(argv)

    models = []
    for raw in args.model:
        try:
            models.append(parse_model(raw))
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    if len(models) < 2:
        parser.error(f"多模型报告至少要两个 --model，现在只有 {len(models)} 个")
    names = [name for name, _ in models]
    if len(set(names)) != len(names):
        parser.error(f"模型名重复了: {names}")

    sys.path.insert(0, str(REPO_ROOT))
    from visualization.det_report import load_archive
    from visualization.wave_report import build_wave_report

    archives = [load_archive(name, directory) for name, directory in models]
    artifacts = build_wave_report(archives, args.out, archive=args.archive)

    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
