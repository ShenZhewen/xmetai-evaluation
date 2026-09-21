#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个批次归档目录 → 多模型对比图 + Markdown 报告（确定性单成员）。

用法::

    python skills/xmetai-evaluation/scripts/generate_det_report.py \\
      --model FGVP=/workspace/_XMETAI_test_results/single_fgvp \\
      --model AIFS=/workspace/_XMETAI_test_results/single_aifs \\
      --model FuXi=/workspace/_XMETAI_test_results/single_fuxi \\
      --model FengQing=/workspace/_XMETAI_test_results/single_fengqing \\
      --out reports/det_4models

``--model NAME=目录`` 可重复，至少两个。目录是 ``vfc/regr_ens.py`` 批量跑出来的
**批次归档目录**（``summary.csv`` + ``batch_meta.json`` + ``<YYYYMMDD>/``），
不是 ``weather_field_scores`` 的产物目录——那条链路走 ``generate_report.py``。

其余选项：

    --out DIR                   报告输出目录（必给，不存在会建）
    --title TEXT                覆盖报告标题那一行
    --change TEXT               本次改动说明，写进第一节
    --archive TEXT              归档名，写进报告抬头的「归档：」
    --declared NAME=v1,v2,...   给第 6.2 节用的声明变量清单，可重复；
                                不给则该节「声明变量数」一律记 —

**报告口径**：正文与附录统一使用 N 个模型**共同的日期**，逐日结果按 lead 平均后比较；
因此各表的样本量一致、可直接横向比较。派生指标的定义见
``visualization/det_report.py`` 的模块 docstring。

**本脚本只做入口**：解析参数 → 把仓库根加进 ``sys.path`` → 调
``visualization.det_report.build_det_report`` → 打印产物清单。
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


def parse_declared(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--declared 要写成 NAME=v1,v2,...，收到的是 {value!r}"
        )
    name, variables = value.split("=", 1)
    return name.strip(), [v.strip() for v in variables.split(",") if v.strip()]


def main(argv=None) -> int:
    # Windows 控制台默认 GBK，报告正文和日志里有大量中文。控制台代码页不是
    # UTF-8 时（GBK 有中文，但 cp437 / cp1252 这类没有），print 会抛
    # UnicodeEncodeError——报告和图其实都已经写完了，却在最后一步非 0 退出。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="按多个批次归档目录出图并写多模型确定性评估报告",
        epilog="示例: python skills/xmetai-evaluation/scripts/generate_det_report.py "
        "--model FGVP=evaluation_results/weather_rmse_single_fgvp "
        "--model FuXi=evaluation_results/weather_rmse_single_fuxi --out reports/det_2models",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="模型的批次归档目录（可重复，至少两个）",
    )
    parser.add_argument("--out", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--title", default=None, help="覆盖报告标题那一行")
    parser.add_argument("--change", default=None, help="本次改动说明，写进报告第一节")
    parser.add_argument("--archive", default=None, help="归档名，写进报告抬头的「归档：」")
    parser.add_argument(
        "--declared", action="append", default=None, metavar="NAME=v1,v2,...",
        help="声明变量清单（可重复），只给第 6.2 节用",
    )
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

    declared = {}
    for raw in args.declared or []:
        try:
            name, variables = parse_declared(raw)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        declared[name] = variables

    sys.path.insert(0, str(REPO_ROOT))
    from visualization.det_report import build_det_report, load_archive

    archives = [load_archive(name, directory) for name, directory in models]
    artifacts = build_det_report(
        archives,
        args.out,
        title=args.title,
        change=args.change,
        archive=args.archive,
        declared=declared or None,
    )
    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
