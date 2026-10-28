# -*- coding: utf-8 -*-
"""暴雨过程个例检验的出图器（多模型 × 多过程 × 双口径）。

只画图，统计量与正文在 :mod:`xmetai_evaluation.visualization.rainstorm_report`；
图用的是它算好的同一批统计量。四类图都多模型并排：过程总览分组条形（两种口径
各一张）、单过程逐日 24h TS 折线（按等级分面）、单过程单起报逐日汇总 TS 条形、
A/B 口径散点（带 ``y = x`` 参考线，看名次会不会翻转）。

不出降水空间分布图：本模块的输入只有站点列联表的聚合统计，没有逐格降水场；
空间图是动图，另见 ``rainstorm_spatial_anim.py``（读抽取脚本聚合的 tp 场，
色标与小图实现放在 ``rainstorm_spatial.py``）。
"""

from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase
from typing import Dict, Optional, Sequence

import numpy as np
from matplotlib.ticker import MaxNLocator

from xmetai_evaluation.visualization.style import (
    BASE_WIDTH,
    LEGEND_FRAME,
    NEUTRAL,
    apply_style,
    font_sizes,
    model_colors,
    snap_width,
)

#: 折线/条形里"没有有效取值"的图元直接跳过，而不是画成 0——
#: TS 的 NaN 是无有效配对或无事件，与"技巧为 0"是两回事（SKILL.md 硬性规则 4）。
_MISSING = "—"

#: 报告图的字号**不随图宽放大**。这些图在屏幕上宽 12–16 in，按 ``font_sizes``
#: 的等比规则会把刻度放大到 20 pt 以上，整张图被字压住；钉在 183 mm 基准档
#: 再明确放大 1.45 倍（刻度约 14.5 pt、轴标签 17.4 pt），屏幕上和 A4 打印都够大。
_REPORT_FONT_SCALE = 1.45
_REPORT_FONTS = font_sizes(BASE_WIDTH * _REPORT_FONT_SCALE)

#: TS 纵轴的档距：0、0.2、0.4……
_TS_STEP = 0.2


def _ts_top(values) -> float:
    """TS 纵轴上界：数据最大值向上取到 0.2 的整数倍，至少 0.2。

    不用固定 1.0——TS ≥50mm 档常年在 0.4–0.6，固定到顶会把曲线压成一条平线。
    """
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    top = max(valid) if valid else _TS_STEP
    top = np.ceil(max(top, _TS_STEP) / _TS_STEP - 1e-9) * _TS_STEP
    return float(round(top, 10))


def _apply_ts_axis(ax, top: float) -> None:
    """0 起、0.2 一档的 TS 纵轴（条形图用，基线必须是 0）。图内标注一律英文。"""
    ax.set_ylim(0.0, top)
    ax.set_yticks(np.round(np.arange(0.0, top + 1e-9, _TS_STEP), 10))


def _fit_ts_panel(ax, values) -> None:
    """折线子图各用各的纵轴：取本格数据的最小/最大值，两端各留一点。

    三个等级的 TS 量级差得多（≥50mm 常在 0–0.4、≥10mm 常在 0.3–0.8），共享
    一根 0 起步的轴会把低等级那格压成一条平线；这里只在本格内把数据拉开，
    刻度由 ``MaxNLocator`` 在 1/2/2.5/5 系列里挑整齐的值。
    """
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not valid:
        ax.set_ylim(0.0, 0.1)
        return
    lo, hi = min(valid), max(valid)
    span = hi - lo
    pad = 0.12 * span if span > 0 else 0.05
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


def _bottom_legend(fig, ax, ncols: int) -> None:
    """图例排成一行放在整张图最底下，加灰色描边框（不填底、不投影）。"""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=ncols,
        fontsize=_REPORT_FONTS["legend"],
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.6,
        handlelength=1.8,
        shadow=False,
        **LEGEND_FRAME,
    )


class RainstormPlotter:
    """暴雨过程个例检验的绘图器。

    Args:
        style: 可选 matplotlib 样式名；默认不套样式（样式会覆盖中文字体设置）。
    """

    def __init__(self, style: Optional[str] = None):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.font = apply_style()
        if style:
            plt.style.use(style)
            self.font = apply_style()
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"

    # ------------------------------------------------------------------ 私有

    def _save(self, fig, save_path: Optional[Path]) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"已保存: {save_path}")
        self._plt.close(fig)

    # ------------------------------------------------------------------ 图

    def plot_process_overview(
        self,
        codes: Sequence[str],
        values: Dict[str, Dict[str, float]],
        *,
        threshold: float,
        save_path: Optional[Path] = None,
        order: Optional[Sequence[str]] = None,
        metric_label: str = "Process cumulative",
    ) -> None:
        """43 个过程的分组条形，按 ``order``（默认第一个模型）降序。

        Args:
            codes: 过程编号，决定横轴顺序。
            values: ``{模型名: {编号: TS}}``；缺的过程画成空位（不补 0）。
            threshold: 图题里的阈值，只影响文字。
            order: 排序依据的模型名；缺省取 ``values`` 的第一个模型。
            metric_label: 纵轴里的口径名（图内标注统一英文），如 ``Process cumulative`` / ``Daily 24-h pooled``；
                **只影响文字**，数值由调用方决定用哪个口径。
        """
        plt = self._plt
        names = list(values)
        ranked_by = order or (names[0] if names else None)
        ordered = sorted(
            codes,
            key=lambda code: (
                values.get(ranked_by, {}).get(code) is None,
                -(values.get(ranked_by, {}).get(code) or 0.0),
            ),
        )
        colors = model_colors(names)
        positions = np.arange(len(ordered), dtype=float)
        # 柱宽和 figure 宽度分开命名：同名会被后一行覆盖成图宽量级。
        bar_width = 0.8 / max(len(names), 1)

        width = snap_width(max(10.0, 0.34 * len(ordered)))
        sizes = _REPORT_FONTS
        fig, ax = plt.subplots(figsize=(width, 5.4))
        for index, name in enumerate(names):
            series = [values.get(name, {}).get(code) for code in ordered]
            xs = [x for x, value in zip(positions, series) if value is not None]
            ys = [value for value in series if value is not None]
            ax.bar(
                np.array(xs) + (index - (len(names) - 1) / 2) * bar_width,
                ys,
                width=bar_width,
                label=name,
                color=colors[name],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered, rotation=90, fontsize=sizes["tick"])
        # 英文标签比中文长，单行在 5.4 in 高的画布里会顶到上下沿；从 "TS" 前折行。
        ax.set_ylabel(
            f"{metric_label}\nTS (≥{threshold:g} mm)", fontsize=sizes["label"]
        )
        ax.tick_params(axis="y", labelsize=sizes["tick"])
        # 图题里的「按 X 降序」写在图注里；图上不放大标题。
        _apply_ts_axis(
            ax,
            _ts_top(
                values.get(name, {}).get(code)
                for name in names
                for code in ordered
            ),
        )
        _bottom_legend(fig, ax, len(names))
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        self._save(fig, save_path)

    def plot_daily_ts(
        self,
        days: Sequence[str],
        series: Dict[str, Dict[str, Dict[float, Optional[float]]]],
        thresholds: Sequence[float],
        *,
        code: str,
        span: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """单个过程的逐日 24h TS 折线，一个阈值一个子图。

        图上去掉总标题（过程编号与时段写在图注里），子图标题只留 ``(a) ≥10mm``
        这样的分面标注；**每个子图各用各的纵轴**（不共享），按本格数据的
        最小/最大值上下各留 12% 的余量，尽量把曲线拉开；图例一行放最底下。

        Args:
            days: 起报日，按顺序决定横轴。
            series: ``{模型名: {起报日: {阈值: TS}}}``。
            thresholds: 要分面的阈值（升序）。
        """
        plt = self._plt
        names = list(series)
        colors = model_colors(names)
        short = [_short_day(day) for day in days]
        sizes = _REPORT_FONTS

        width = snap_width(4.2 * max(len(thresholds), 1))
        # 子图上下拉长：原来 3.8 in 的扁条在 16 in 宽下几乎看不出逐日起伏。
        height = max(4.8, 0.36 * width)
        # 各子图各用各的纵轴（不 sharey）：≥10 与 ≥50 的量级差得多。
        fig, axes = plt.subplots(
            1, max(len(thresholds), 1), figsize=(width, height),
            squeeze=False,
        )
        for column, threshold in enumerate(thresholds):
            ax = axes[0][column]
            column_values = []
            for name in names:
                values = [
                    series.get(name, {}).get(day, {}).get(threshold) for day in days
                ]
                column_values.extend(value for value in values if value is not None)
                xs = [x for x, value in zip(short, values) if value is not None]
                ys = [value for value in values if value is not None]
                if xs:
                    ax.plot(xs, ys, marker="o", markersize=5, label=name, color=colors[name])
            ax.set_title(
                f"({ascii_lowercase[column]}) ≥{threshold:g} mm",
                fontsize=sizes["panel"], loc="left", pad=4,
            )
            _fit_ts_panel(ax, column_values)
            ax.tick_params(labelsize=sizes["tick"])
        axes[0][0].set_ylabel("TS", fontsize=sizes["label"])
        _bottom_legend(fig, axes[0][0], len(names))
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
        self._save(fig, save_path)

    def plot_threshold_bars(
        self,
        thresholds: Sequence[float],
        values: Dict[str, Dict[float, Optional[float]]],
        *,
        code: str,
        span: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """单个过程的累积 TS 按阈值分组的条形，每个模型一组。

        与 :meth:`plot_daily_ts` 同一套版式：不写总标题，纵轴 0.2 一档且上界
        取最大值，图例一行放最底下。
        """
        plt = self._plt
        names = list(values)
        colors = model_colors(names)
        labels = [f"≥{threshold:g}" for threshold in thresholds]
        positions = np.arange(len(thresholds), dtype=float)

        # 柱宽按"一组里每根柱子"算；figure 宽度是另一个量，别再同名（以前同名
        # 互相覆盖，柱子宽度被撑到整个横轴的量级，刻度全糊在一起）。
        bar_width = 0.8 / max(len(names), 1)

        width = snap_width(max(6.0, 1.3 * len(thresholds)))
        height = max(4.8, 0.36 * width)
        sizes = _REPORT_FONTS
        fig, ax = plt.subplots(figsize=(width, height))
        for index, name in enumerate(names):
            series = [values.get(name, {}).get(threshold) for threshold in thresholds]
            xs = [x for x, value in zip(positions, series) if value is not None]
            ys = [value for value in series if value is not None]
            ax.bar(
                np.array(xs) + (index - (len(names) - 1) / 2) * bar_width,
                ys,
                width=bar_width,
                label=name,
                color=colors[name],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=sizes["tick"])
        ax.tick_params(axis="y", labelsize=sizes["tick"])
        ax.set_xlabel("Rainfall threshold (mm)", fontsize=sizes["label"])
        ax.set_ylabel("Process cumulative TS", fontsize=sizes["label"])
        _apply_ts_axis(
            ax,
            _ts_top(
                values.get(name, {}).get(threshold)
                for name in names
                for threshold in thresholds
            ),
        )
        _bottom_legend(fig, ax, len(names))
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
        self._save(fig, save_path)

    def plot_caliber_scatter(
        self,
        points: Dict[str, Sequence[tuple]],
        *,
        threshold: float,
        save_path: Optional[Path] = None,
    ) -> None:
        """逐过程散点：横轴口径 A 起报日汇总、纵轴口径 B 累积，带 ``y = x`` 参考线。

        落在这条线**上方**的过程，说明它的累积 TS 高于按逐日平均外推的水平；
        偏离越大，两个口径给出的"这个过程好不好预报"差得越远。点都是有效
        配对（两个口径都有数）——缺一样的那个过程整点不画。

        Args:
            points: ``{模型名: [(口径A 汇总 TS, 口径B 累积 TS), …]}``。
        """
        plt = self._plt
        names = list(points)
        colors = model_colors(names)

        width = snap_width(5.6)
        sizes = _REPORT_FONTS
        fig, ax = plt.subplots(figsize=(width, 5.2))
        for name in names:
            pairs = [(x, y) for x, y in points.get(name, []) if x is not None and y is not None]
            if not pairs:
                continue
            xs = [pair[0] for pair in pairs]
            ys = [pair[1] for pair in pairs]
            ax.scatter(xs, ys, s=26, label=name,
                       color=colors[name], alpha=0.8, edgecolors="none")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9,
                color=NEUTRAL["reference"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ticks = np.round(np.arange(0.0, 1.0 + 1e-9, _TS_STEP), 10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel(
            f"Caliber A: daily 24-h pooled TS (≥{threshold:g} mm)",
            fontsize=sizes["label"],
        )
        # 英文标签比中文长得多，横排会顶出画布上下沿；折成两行再转 90°。
        ax.set_ylabel(
            f"Caliber B: process\ncumulative TS (≥{threshold:g} mm)",
            fontsize=sizes["label"],
        )
        ax.tick_params(labelsize=sizes["tick"])
        _bottom_legend(fig, ax, len(names))
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
        self._save(fig, save_path)


def _short_day(day: str) -> str:
    """``20250311`` → ``0311``，折线横轴用。不是八位数字就原样返回。"""
    text = str(day)
    return text[4:] if len(text) == 8 and text.isdigit() else text
