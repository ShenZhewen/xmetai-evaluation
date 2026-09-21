#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性多模型对比出图（RMSE / ACC / FA / 纬向谱）。

与 ``field_plots`` / ``precipitation_plots`` 的分工：那两份画的是**单模型**的产物，
这一份画的是**同一指标、多个模型**叠在一起的对比图——
报告附录那四张图（RMSE、ACC、FA ratio、功率谱）就是它出的。

配色复用 :data:`visualization.precipitation_plots.MODEL_COLORS`，
按"模型在字典里的次序"分配，所以只要调用方每次传进来的次序一致，
同一个模型在四张图里就是同一个颜色。

约定：``series_by_model`` 是 ``{模型名: DataFrame}``，DataFrame 的
index 是预报时效（h），columns 是变量名——和逐日 ``rmse_<date>_det.csv`` 的
布局一致（那边 index 就是 lead_h）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xmetai_evaluation.visualization.precipitation_plots import MODEL_COLORS, setup_chinese_font


def model_colors(names: Sequence[str]) -> Dict[str, str]:
    """按模型出现次序取色；超过调色板长度就循环。"""
    return {str(name): MODEL_COLORS[index % len(MODEL_COLORS)] for index, name in enumerate(names)}


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    """把非数值列/行清掉，剩下纯 float 的 lead × 变量表。"""
    data = frame.copy()
    if not isinstance(data.index, pd.Index):
        raise ValueError("series_by_model 的值得是 DataFrame（index=lead_h，columns=变量）")
    data.index = pd.to_numeric(data.index, errors="coerce")
    data = data[data.index.notna()]
    data = data.apply(pd.to_numeric, errors="coerce")
    return data.sort_index()


class DetPlotter:
    """多模型确定性指标对比图的绘图器。"""

    def __init__(self, style: Optional[str] = None):
        """
        Args:
            style: 可选 matplotlib 样式名；默认不套样式，
                避免样式把中文字体设置覆盖掉（降水那边就栽在这）。
        """
        self.font = setup_chinese_font()
        if style:
            plt.style.use(style)
            setup_chinese_font()
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"

    # ------------------------------------------------------------------ #
    def _save(self, fig, save_path: Optional[Path]) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

    # ------------------------------------------------------------------ #
    def plot_model_panels(
        self,
        series_by_model: Mapping[str, pd.DataFrame],
        variables: Sequence[str],
        *,
        ylabel: str,
        ref_line: Optional[float] = None,
        ncols: int = 4,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """一个变量一个子图、一个模型一条线。RMSE / ACC / FA 三张图共用。

        Args:
            series_by_model: ``{模型名: DataFrame}``，DataFrame index=lead_h、columns=变量。
            variables: 要画的变量，决定子图顺序与数量。
            ylabel: 纵轴标签（各子图共用）。
            ref_line: 画一条水平参考线（FA 的 ratio=1 用）。
            ncols: 一行几个子图。
            suptitle: 总标题。
            save_path: PNG 落点。

        Raises:
            ValueError: 没有可画的变量，或所有模型在该变量上都是空的。
        """
        variables = [str(v) for v in variables]
        if not variables:
            raise ValueError("variables 是空的，没东西可画")

        frames = {str(name): _finite(frame) for name, frame in series_by_model.items()}
        colors = model_colors(list(frames))

        ncols = max(1, min(int(ncols), len(variables)))
        nrows = int(np.ceil(len(variables) / ncols))
        figure, axes = plt.subplots(
            nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False
        )
        axes = axes.ravel()

        for index, variable in enumerate(variables):
            axis = axes[index]
            plotted = 0
            for name, frame in frames.items():
                if variable not in frame.columns:
                    continue
                series = frame[variable].dropna()
                if series.empty:
                    continue
                axis.plot(
                    series.index,
                    series.values,
                    color=colors[name],
                    linewidth=1.6,
                    marker="o",
                    markersize=3,
                    label=name,
                )
                plotted += 1
            if plotted == 0:
                raise ValueError(f"所有模型都没有变量 {variable!r} 的数据")
            if ref_line is not None:
                axis.axhline(ref_line, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
            axis.set_title(variable, fontsize=11)
            axis.set_xlabel("预报时效 (h)", fontsize=9)
            axis.set_ylabel(ylabel, fontsize=9)
            axis.grid(alpha=0.3, linewidth=0.6)
            axis.tick_params(labelsize=8)

        for axis in axes[len(variables):]:  # 子图数不整除时，多余的格子收起来
            axis.set_visible(False)

        # 图例放整张图底部，省得每个子图都重复一遍四行图例把曲线盖住。
        handles, labels = axes[0].get_legend_handles_labels()
        if ref_line is not None:
            handles.append(plt.Line2D([], [], color="#666666", linestyle="--", linewidth=1.0))
            labels.append(f"参考线 = {ref_line:g}")
        if handles:
            figure.legend(
                handles, labels, loc="lower center", ncols=min(len(labels), 6),
                fontsize=10, frameon=False,
            )
        if suptitle:
            figure.suptitle(suptitle, fontsize=13)
        figure.tight_layout(rect=(0, 0.04, 1, 0.97 if suptitle else 1.0))
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_group_bars(
        self,
        table: pd.DataFrame,
        *,
        ylabel: str,
        ref_line: Optional[float] = None,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """一行一个分组、组里一个模型一根柱。

        「分组」是纬度带这类**同一指标、不同统计范围**的切法（分带综合相对 RMSE 就是
        这个形状）。和 :meth:`plot_model_panels` 的分工：那个是「一个分组一张子图、
        组内画随时效的曲线」，这个是「所有分组并排、组内比大小」——分带只有几个，
        并排放才看得出带间差距。

        Args:
            table: index=分组名，columns=模型名。
            ylabel: 纵轴标签。
            ref_line: 参考线（相对指标画 100）。
            suptitle: 总标题。
            save_path: PNG 落点。

        Raises:
            ValueError: 表是空的。
        """
        data = table.apply(pd.to_numeric, errors="coerce")
        if data.empty or not len(data.columns):
            raise ValueError("分组对比表是空的，没东西可画")

        names = [str(column) for column in data.columns]
        groups = [str(index) for index in data.index]
        colors = model_colors(names)

        positions = np.arange(len(groups), dtype=float)
        width = 0.8 / max(len(names), 1)
        figure, axis = plt.subplots(figsize=(max(6.0, 1.8 * len(groups)), 4.6))
        for offset, name in enumerate(names):
            axis.bar(
                positions + offset * width - 0.4 + width / 2,
                data[name].values,
                width=width,
                color=colors[name],
                label=name,
            )
        if ref_line is not None:
            axis.axhline(ref_line, color="#666666", linestyle="--", linewidth=1.0,
                         label=f"参考线 = {ref_line:g}")
        axis.set_xticks(positions)
        axis.set_xticklabels(groups, fontsize=9)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.grid(alpha=0.3, axis="y", linewidth=0.6)
        axis.legend(fontsize=10, frameon=False)
        if suptitle:
            axis.set_title(suptitle, fontsize=12)
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_model_spectrum(
        self,
        spectra_by_model: Mapping[str, pd.DataFrame],
        variable: str,
        *,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """双对数坐标：多模型的预测谱 + 一条共享的观测谱。

        Args:
            spectra_by_model: ``{模型名: DataFrame}``，index=波数 k，
                columns 含 ``pred`` 与 ``obs``。``obs`` 是所有模型共用的实况谱，
                只画第一条非空的（同一个变量、同一批日期，观测谱本就该一致）。
            variable: 变量名，只用来写标题。
            suptitle: 总标题。
            save_path: PNG 落点。

        Raises:
            ValueError: 没有任何模型的谱数据，或所有模型都只有 k=0。
        """
        figure, axis = plt.subplots(figsize=(9, 6))
        colors = model_colors(list(spectra_by_model))
        observation = None
        plotted = 0

        for name, raw in spectra_by_model.items():
            frame = _finite(raw)
            if "pred" not in frame.columns:
                raise ValueError(f"{name} 的谱表没有 pred 列；现有列: {list(frame.columns)}")
            curve = frame["pred"].dropna()
            curve = curve[curve.index > 0]  # k=0 的功率恒为 0，log 轴画不了
            curve = curve[curve > 0]
            if curve.empty:
                continue
            axis.plot(curve.index, curve.values, color=colors[name], linewidth=1.6, label=name)
            plotted += 1
            if observation is None and "obs" in frame.columns:
                reference = frame["obs"].dropna()
                reference = reference[(reference.index > 0) & (reference > 0)]
                if not reference.empty:
                    observation = reference

        if plotted == 0:
            raise ValueError(f"没有任何模型给出 {variable} 的可画谱线（k>0 且功率>0）")

        if observation is not None:
            axis.plot(
                observation.index, observation.values,
                color="#000000", linestyle="--", linewidth=1.4, label="观测",
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("纬向波数 k", fontsize=10)
        axis.set_ylabel("功率（双对数）", fontsize=10)
        axis.grid(alpha=0.3, which="both", linewidth=0.6)
        axis.legend(loc="best", fontsize=10)
        axis.set_title(suptitle or f"{variable} 纬向功率谱（多模型对比）", fontsize=12)
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_rmse_heatmap(
        self,
        panels: Mapping[str, pd.DataFrame],
        variables: Sequence[str],
        *,
        leads: Sequence[float],
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """变量 × 时效 的热力图，一个模型一幅，**共用一根色标**。

        Args:
            panels: ``{模型名: DataFrame}``，index=变量、columns=lead_h。
                格子里放的是**已经归一化过的值**（惯例是「对该模型首时效的 RMSE
                倍数」）——各变量单位不同，不归一化就没法共用一根色标。
            variables: 纵轴变量顺序（自上而下）。
            leads: 横轴时效顺序。
            suptitle: 总标题。
            save_path: PNG 落点。

        Raises:
            ValueError: 没有任何模型有可画的格子。
        """
        variables = [str(v) for v in variables]
        leads = [float(v) for v in leads]
        if not variables or not leads:
            raise ValueError("variables 或 leads 是空的，没东西可画")

        names = [str(name) for name in panels]
        matrices: Dict[str, np.ndarray] = {}
        for name in names:
            frame = _finite(panels[name])
            matrix = np.full((len(variables), len(leads)), np.nan, dtype=float)
            for row, variable in enumerate(variables):
                if variable not in frame.columns:
                    continue
                for column, lead in enumerate(leads):
                    if lead in frame.index:
                        matrix[row, column] = float(frame.loc[lead, variable])
            matrices[name] = matrix

        stacked = np.concatenate([m.ravel() for m in matrices.values()])
        finite = stacked[np.isfinite(stacked)]
        if finite.size == 0:
            raise ValueError("所有模型都没有可画的格子")
        low, high = float(finite.min()), float(finite.max())
        if high <= low:                      # 全部同值：给个非退化区间，免得色标爆掉
            high = low + 1e-9

        columns = max(1, len(names))
        figure, axes = plt.subplots(
            1, columns, figsize=(2.6 * columns + 2.4, 0.42 * len(variables) * 2 + 2.6),
            squeeze=False,
        )
        axes = axes.ravel()
        image = None
        for axis, name in zip(axes, names):
            matrix = matrices[name]
            image = axis.imshow(
                matrix, aspect="auto", cmap="viridis", vmin=low, vmax=high,
                origin="upper", interpolation="nearest",
            )
            # 时效多到几十个时全标会挤成一片，隔几个标一个，密度控制在 20 个以内。
            stride = max(1, int(np.ceil(len(leads) / 20)))
            axis.set_xticks(range(0, len(leads), stride))
            axis.set_xticklabels(
                [f"{leads[index]:g}" for index in range(0, len(leads), stride)],
                fontsize=8, rotation=90,
            )
            axis.set_yticks(range(len(variables)))
            axis.set_yticklabels(variables, fontsize=8)
            axis.set_xlabel("预报时效 (h)", fontsize=9)
            axis.set_title(name, fontsize=11)
            # 标数只在小表上做：列一多，每个格子摊不到半个字宽，标了也是糊的，
            # 读数交给色标。判据是**列数**不是格子总数——宽表正是糊得最狠的那种。
            if matrix.shape[1] <= 14 and matrix.size <= 400:
                for row in range(matrix.shape[0]):
                    for column in range(matrix.shape[1]):
                        value = matrix[row, column]
                        if not np.isfinite(value):
                            continue
                        shade = (value - low) / (high - low)
                        axis.text(
                            column, row, f"{value:.1f}", ha="center", va="center",
                            fontsize=6, color="white" if shade < 0.6 else "black",
                        )
        for axis in axes[len(names):]:
            axis.set_visible(False)
        if image is not None:
            bar = figure.colorbar(image, ax=list(axes[: len(names)]), fraction=0.03, pad=0.02)
            bar.ax.tick_params(labelsize=8)
        if suptitle:
            figure.suptitle(suptitle, fontsize=13)
        self._save(figure, save_path)
        return figure
