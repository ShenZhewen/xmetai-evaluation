#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个集合批次归档目录 → 集合预报多模型评估检验报告。

用法::

    python skills/xmetai-evaluation/scripts/generate_ens_report.py \\
      --model FuXi=outputs/results/weather_rmse_ens_fuxi/ens_fuxi \\
      --model AIFS=outputs/results/weather_rmse_ens_aifs/ens_aifs \\
      --out outputs/.temp/ens_2models

``--model NAME=目录`` 可重复，至少两个。目录是 ``--summarize-ens`` 口径跑出来的
**批次归档目录**（``summary.csv`` + ``batch_meta.json`` + ``<YYYYMMDD>/``）。

其余选项：

    --out DIR       报告输出目录（必给，不存在会建）
    --archive TEXT  归档名，写进报告抬头

**与 `_single` 支线的差别（选错入口会静默算错）**：

- 逐日 ``rmse`` / ``acc`` / ``fa`` 读的是 ``*_ensmean.csv``，即**集合平均场**，
  不是单个成员；``crps`` / ``spread`` / ``spread_rmse_ratio`` 是集合特有指标。
- 纬向谱偏差用 **ln**，``generate_det_report.py`` 用 **log10**，两者差 ×2.3026，
  不要互相对数。
- 集合归档也能被 ``generate_det_report.py`` 读进来（它认 ``_ensmean`` 后缀），
  但那份报告的章节口径不是集合的，**不要拿它当集合报告**。

**共同日期不足 2 天会直接报错**，并在错误里点名说清各模型的日期跨度与两两交集
——三个模型各覆盖不同时段时，硬出一份报告只会得到一组没有意义的数。

**本脚本只做入口**：解析参数 → 把仓库根加进 ``sys.path`` → 调
``visualization.ens_report.build_ens_report`` → 打印产物清单。
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
        description="按多个集合批次归档目录出图并写集合预报评估检验报告",
        epilog="示例: python skills/xmetai-evaluation/scripts/generate_ens_report.py "
        "--model FuXi=outputs/results/weather_rmse_ens_fuxi/ens_fuxi "
        "--model AIFS=outputs/results/weather_rmse_ens_aifs/ens_aifs "
        "--out outputs/.temp/ens_2models",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="集合批次归档目录（可重复，至少两个）",
    )
    parser.add_argument("--out", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--archive", default=None, help="归档名，写进报告抬头")
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
    from visualization.ens_report import build_ens_report

    archives = [load_archive(name, directory) for name, directory in models]
    artifacts = build_ens_report(archives, args.out, archive=args.archive)

    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
