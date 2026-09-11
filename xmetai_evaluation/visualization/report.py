#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""产物目录 → 报告：认能力 + 派发。

一句话用法::

    python -m xmetai_evaluation.visualization.report <产物目录> [--out 输出目录]

**能力身份从产物自己认，不靠目录名**：``manifest.json`` 里的
``resolved_config.pipeline`` 是权威判据（``runner`` 无条件写它）。目录名与
``run_id`` 都不可靠——配置名（``weather_field_scores_era5_fuxi``）与流程名
（``weather_field_scores``）本来就不绑定。

没有 ``resolved_config`` 的老产物走退路判据：``protocol_id`` + ``manifest["metrics"]``
的组合。注意 ``manifest["metrics"]`` 是**原始**指标名（``activity``），
不是长表里的 ``activity_ratio``。

认得出的能力如果还没有报告模板，**明确报错**，不静默回落到别的模板上——
拿错模板比报错危险得多。

能力先归到**报告家族**（``IMPLEMENTED`` 的值），再按家族派发：同一家族的流程
写出的产物表结构相同，共用一套模板与图表。目前两个家族——``field``
（连续场，吃 ``scores.csv``）与 ``ts``（降水分类检验，吃
``diagnostics/categorical_wide.csv``）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from xmetai_evaluation.visualization.field_report import manifest_info_from

#: 已经有报告模板的能力：流程名 -> 所属**报告家族**。
#: 同一家族的流程共用一套模板与图表，因为它们写出的产物表结构相同
#: （三条 TS 流程都写 ``diagnostics/categorical_wide.csv``）。
IMPLEMENTED: Dict[str, str] = {
    "weather_field_scores": "field",
    "weather_ts_det": "ts",
    "weather_ts_ens": "ts",
    "fdp_precip_ts": "ts",
}

#: 家族 -> 一句话说明，只用于日志与报错信息。
FAMILY_LABELS: Dict[str, str] = {
    "field": "确定性连续场检验（RMSE / ACC / 活跃度 / 纬向谱）",
    "ts": "站点降水分类检验（TS / POD / FAR / 频率偏差）",
}

#: 退路判据 ``(流程名, protocol_id, 指标名集合)``，只在 manifest 里没有
#: ``resolved_config.pipeline`` 时启用；指标集合要求**完全相等**，
#: 多一个少一个都不算（``weather_ens_field_scores`` 就是多一个 ``crps``）。
_FALLBACK_RULES = (
    (
        "weather_field_scores",
        "grid_valid_time",
        frozenset({"rmse", "acc", "activity", "zonal_spectrum"}),
    ),
    # det / ens / fdp 三类 TS 产物在这组判据下区分不开，但三者同属 "ts" 家族、
    # 共用一套模板，所以落到哪个名字上都不影响出什么报告。
    ("weather_ts_det", "station_valid_time", frozenset({"ts"})),
)

DETAILS_RELATIVE_PATH = Path("diagnostics") / "scores_detail.csv"
CATEGORICAL_WIDE_RELATIVE_PATH = Path("diagnostics") / "categorical_wide.csv"


def parse_compare_specs(specs: Optional[Sequence[str]]) -> Dict[str, Path]:
    """把 ``["FuXi=a.csv", "AIFS=b.csv"]`` 解析成 ``{名字: 路径}``。

    对比模型**只能是裸结果表**，不能是产物目录——它们没有 manifest，
    认不出能力，名字也没法从 ``run_id`` 推断（``ts_multi_*`` 那几个目录就是这种）。
    名字由调用方显式给出，不从目录名猜：这个仓库的规矩是身份不靠目录名。

    Raises:
        ValueError: 条目没有 ``=``，或文件不存在。
    """
    parsed: Dict[str, Path] = {}
    for spec in specs or []:
        name, separator, raw_path = str(spec).partition("=")
        name, raw_path = name.strip(), raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(
                f"--compare 的写法是「名字=CSV路径」，收到的是 {spec!r}。"
                f"例：--compare \"FuXi=evaluation_results/ts_multi_fuxi/ts_fuxi_2025.csv\""
            )
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"--compare 里的 {name} 指向的 {path} 不存在")
        parsed[name] = path
    return parsed


def load_manifest(output_dir: Path) -> Dict[str, object]:
    """读产物目录里的 ``manifest.json``。

    Raises:
        ValueError: 文件不存在或不是合法 JSON。
    """
    path = Path(output_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(
            f"{path} 不存在：这里看着不像评测产物目录。"
            f"该给的是 ``--config`` 的 output_dir（里面有 scores.csv 和 manifest.json）。"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} 不是合法 JSON：{error}") from error


def detect_capability(output_dir: Path) -> str:
    """认出产物目录属于哪条流程。

    Raises:
        ValueError: 目录里没有 manifest，或认不出 / 认得但没有报告模板。
    """
    manifest = load_manifest(Path(output_dir))
    resolved = manifest.get("resolved_config")
    pipeline = resolved.get("pipeline") if isinstance(resolved, dict) else None
    if pipeline:
        if str(pipeline) in IMPLEMENTED:
            return str(pipeline)
        raise ValueError(
            f"流程 '{pipeline}' 的报告模板还没做"
            f"（已有模板的流程：{'、'.join(sorted(IMPLEMENTED))}）。"
            f"产物本身是完整的，只是没有对应的出图与报告代码。"
        )

    protocol = str(manifest.get("protocol_id") or "")
    metrics = frozenset(str(name) for name in (manifest.get("metrics") or []))
    for name, want_protocol, want_metrics in _FALLBACK_RULES:
        if protocol == want_protocol and metrics == want_metrics:
            return name
    raise ValueError(
        f"认不出这个产物属于哪条流程：manifest 里既没有 `resolved_config.pipeline`，"
        f"`protocol_id={protocol!r}` + `metrics={sorted(metrics)}` 也不匹配任何已知判据。"
        f"已有模板的流程：{'、'.join(sorted(IMPLEMENTED))}。"
    )


def _default_output_dir(output_dir: Path, info: Dict[str, object]) -> Path:
    run_id = str(info.get("run_id") or Path(output_dir).name)
    return Path("reports") / run_id


def build_report(
    output_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    spectrum_variable: Optional[str] = None,
    spectrum_lead: Optional[float] = None,
    change_description: Optional[str] = None,
    compare: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """按产物目录里的能力出图并写报告，返回 ``{名称: 路径}``。

    Args:
        output_dir: ``--config`` 里的 output_dir，里面应有 ``manifest.json``
            以及该能力对应的结果表——连续场是 ``scores.csv``，
            TS 系列是 ``diagnostics/categorical_wide.csv``。
        out_dir: 报告输出目录；默认 ``reports/<run_id>``（相对当前目录）。
        model_name: 报告标题里的模型名；默认取 ``run_id``。**不要用 manifest 的
            ``forecast_source``**——那是 reader 类型（如 ``"fuxi"``），不是模型名，
            拿它当标题会张冠李戴。
        compare: 对比模型，``["名字=CSV路径", …]``，只在 TS 系列上有意义。
            给出来的模型会和主模型一起进箱线图、差值图和对比表。
        spectrum_variable / spectrum_lead: 只在连续场链路、且要看逐波数谱曲线时才给——
            给了才会去读很大的 ``diagnostics/scores_detail.csv``。
            ``spectrum_lead`` 省略时取该变量的最大时效。

    Raises:
        ValueError: 该能力要的结果表不存在，或认不出能力，或 ``compare`` 不合法。
        ImportError: 没装 matplotlib（``pip install -e .[viz]``）。
    """
    output_dir = Path(output_dir)
    capability = detect_capability(output_dir)
    family = IMPLEMENTED[capability]
    info = manifest_info_from(load_manifest(output_dir))
    target = Path(out_dir) if out_dir is not None else _default_output_dir(output_dir, info)
    full_model_name = model_name or str(info.get("run_id") or output_dir.name)

    # 先把 compare 校验干净再往下走：任何错误都必须在 create_report 建目录之前抛出来，
    # 否则会留下一个只有半张图的空报告目录。
    comparisons = parse_compare_specs(compare)
    if comparisons and family != "ts":
        raise ValueError(
            f"只有 TS 系列支持 --compare（对比对象要的是同量级、同阈值的宽表）；"
            f"当前产物是 {capability}（{FAMILY_LABELS[family]}）。"
        )
    if full_model_name in comparisons:
        raise ValueError(
            f"--compare 里的名字 {full_model_name!r} 和主模型重名了，"
            f"会把主模型的来源路径覆盖掉。换个名字，或用 --model 改主模型名。"
        )

    print(f"能力：{capability} —— {FAMILY_LABELS[family]}")
    if family == "ts":
        return _render_ts(output_dir, target, full_model_name, change_description, comparisons)
    return _render_field(
        output_dir,
        target,
        full_model_name,
        info,
        spectrum_variable,
        spectrum_lead,
        change_description,
    )


def _render_ts(
    output_dir: Path,
    target: Path,
    model_name: str,
    change_description: Optional[str],
    comparisons: Optional[Dict[str, Path]] = None,
) -> Dict[str, Path]:
    """TS 系列：宽表 → 曲线图 + 报告。``comparisons`` 是 ``{模型名: 对比结果表路径}``。"""
    wide_path = output_dir / CATEGORICAL_WIDE_RELATIVE_PATH
    if not wide_path.is_file():
        raise ValueError(
            f"{wide_path} 不存在，没有可出报告的结果表。TS 系列的图表吃的是宽表，"
            f"这里**不做长表→宽表转换**（转出来的口径和评测流程自己写的那份未必一致）。"
            f'在配置里加上 writers=["csv_long", "categorical_wide"] 重跑评测即可产出。'
        )
    try:
        import pandas as pd

        from xmetai_evaluation.visualization.precipitation_plots import PrecipitationPlotter
    except ImportError as error:  # pragma: no cover - 取决于环境
        raise ImportError(
            "出图需要 matplotlib（可选依赖），先装：pip install -e .[viz]。"
            f"原始错误：{error}"
        ) from error

    comparisons = comparisons or {}
    # 来源要连对比模型一起写进报告头，否则每个对比模型都标成"未提供"
    sources = {model_name: str(wide_path)}
    sources.update({name: str(path) for name, path in comparisons.items()})
    return PrecipitationPlotter().create_report(
        pd.read_csv(wide_path),
        output_dir=target,
        model_name=model_name,
        baselines={name: pd.read_csv(path) for name, path in comparisons.items()} or None,
        sources=sources,
        change_description=change_description,
    )


def _render_field(
    output_dir: Path,
    target: Path,
    model_name: str,
    info: Dict[str, object],
    spectrum_variable: Optional[str],
    spectrum_lead: Optional[float],
    change_description: Optional[str],
) -> Dict[str, Path]:
    """连续场：长表 → 曲线图 + 报告。"""
    scores_path = output_dir / "scores.csv"
    if not scores_path.is_file():
        raise ValueError(f"{scores_path} 不存在，没有可出报告的结果表")
    try:
        import pandas as pd

        from xmetai_evaluation.visualization.field_plots import FieldScorePlotter
    except ImportError as error:  # pragma: no cover - 取决于环境
        raise ImportError(
            "出图需要 matplotlib（可选依赖），先装：pip install -e .[viz]。"
            f"原始错误：{error}"
        ) from error

    details_path = output_dir / DETAILS_RELATIVE_PATH
    return FieldScorePlotter().create_report(
        pd.read_csv(scores_path),
        output_dir=target,
        details_path=details_path if details_path.is_file() else None,
        spectrum_variable=spectrum_variable,
        spectrum_lead=spectrum_lead,
        model_name=model_name,
        sources={"scores": str(scores_path)},
        manifest_info=info,
        change_description=change_description,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="按评测产物目录自动选模板，出图并写 Markdown 报告",
        epilog="示例: python -m xmetai_evaluation.visualization.report "
        "evaluation_results/weather_ts_det_fgvp --out reports/weather_ts_det_fgvp",
    )
    parser.add_argument("output_dir", type=Path, help="评测产物目录（含 scores.csv 与 manifest.json）")
    parser.add_argument("--out", type=Path, default=None, help="报告输出目录（默认 reports/<run_id>）")
    parser.add_argument("--model", default=None, help="模型名（默认取 manifest 的 run_id）")
    parser.add_argument(
        "--spectrum-variable", default=None, help="要看谱曲线的变量（需变量真的算过纬向谱）"
    )
    parser.add_argument(
        "--spectrum-lead", type=float, default=None, help="谱曲线的时效(h)，省略取该变量最大时效"
    )
    parser.add_argument("--change", default=None, help="本次改动说明，写进报告第一节")
    parser.add_argument(
        "--compare", action="append", default=None, metavar="名字=CSV路径",
        help="对比模型的结果表（可重复），只在 TS 系列上有意义；"
        "给了就一起进箱线图、差值图与对比表",
    )
    args = parser.parse_args(argv)

    artifacts = build_report(
        args.output_dir,
        out_dir=args.out,
        model_name=args.model,
        spectrum_variable=args.spectrum_variable,
        spectrum_lead=args.spectrum_lead,
        change_description=args.change,
        compare=args.compare,
    )
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
