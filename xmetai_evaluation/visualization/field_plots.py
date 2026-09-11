#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连续场评分（RMSE / ACC / 活跃度 / 纬向谱）的可视化。

输入：``weather_field_scores`` 流程产出的 ``scores.csv`` 长表，列至少包含
``variable / metric / lead_h / value``（``unit`` 与 ``status`` 有则用、没有则兜底）。

输出：若干 PNG，以及一份 Markdown 报告（报告正文见 :mod:`field_report`）。

三件与"一张图一根轴画到底"不同的做法：

* **RMSE 按单位分面**。同一张图里 z500 是 ``m^2/s^2``、t2m 是 ``K``、msl 是 ``Pa``，
  混在一根纵轴上比大小是错的，所以一个单位一个子图；
* **热力图用"相对首时效的倍数"**。同理，跨变量的原始值不可比；除过首时效之后
  无量纲，行与行才可比，行标签里同时写上首时效的绝对值，尺度不丢；
* **参考线只在有理想值时画**（语义表里的 ``ref``）：比值类画 ``y=1``，ACC 画 ``y=1``，
  RMSE 没有理想值就不画。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xmetai_evaluation.visualization.field_report import (
    SKILL_METRICS,
    build_field_report,
    metric_label,
    metric_ref,
    requested_from,
)
from xmetai_evaluation.visualization.precipitation_plots import setup_chinese_font

REQUIRED_COLUMNS = ("lead_h", "variable", "metric", "value")

#: ``details`` 里逐波数谱的 group 形如 ``k=0`` / ``k=720``；``summary`` 是总功率行。
_WAVENUMBER_PATTERN = r"^k=\d+$"

#: 读大明细表时只取这几列（实测 90 万行，全读没必要）。
_DETAIL_COLUMNS = ("variable", "lead_h", "group", "field", "value")


def prepare_field_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化连续场长表。

    保留 ``status != success`` 的行（报告要数它们），出图时由各方法自行过滤。

    Raises:
        ValueError: 缺少必需列，或数据为空 / 没有可用的 ``lead_h``。
    """
    data = df.copy()
    data.columns = [str(column).strip() for column in data.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"连续场长表缺少必需列 {missing}；现有列: {list(data.columns)}")
    if data.empty:
        raise ValueError("连续场长表是空的")

    for column in ("lead_h", "value"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["lead_h"])
    if data.empty:
        raise ValueError("连续场长表里没有可用的 lead_h 数值")

    data["variable"] = data["variable"].astype(str)
    data["metric"] = data["metric"].astype(str)
    if "unit" not in data.columns:
        data["unit"] = ""
    data["unit"] = data["unit"].fillna("").astype(str)
    if "status" not in data.columns:
        data["status"] = "success"
    data["status"] = data["status"].fillna("success").astype(str)
    return data.sort_values(["metric", "variable", "lead_h"])


def _usable(data: pd.DataFrame) -> pd.DataFrame:
    """只留 ``status=success`` 且 ``value`` 非空的行。"""
    return data[(data["status"] == "success") & data["value"].notna()]


def _unit_of(chunk: pd.DataFrame) -> str:
    texts = [str(unit) for unit in chunk["unit"].dropna().unique() if str(unit)]
    return texts[0] if texts else ""


def _axis_label(metric: str, unit: str) -> str:
    """纵轴标签：无量纲的 ``1`` 不写出来，其余原样带上方括号。"""
    label = metric_label(metric)
    if not unit or unit == "1":
        return label
    return f"{label} [{unit}]"


def load_spectrum_details(
    path: Path, variable: Optional[str] = None, lead_h: Optional[float] = None
) -> pd.DataFrame:
    """读 ``diagnostics/scores_detail.csv`` 里的逐波数谱，按变量/时效过滤。

    明细表很大（本仓库实测 90 万行），所以只取用得上的几列再过滤。

    Raises:
        ValueError: 过滤后没有任何 ``k=<波数>`` 行。
    """
    frame = pd.read_csv(path, usecols=lambda name: name in _DETAIL_COLUMNS)
    for column in _DETAIL_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"明细表 {path} 缺少列 {column}；现有列: {list(frame.columns)}")
    frame["group"] = frame["group"].astype(str)
    frame = frame[frame["group"].str.match(_WAVENUMBER_PATTERN)]
    frame["lead_h"] = pd.to_numeric(frame["lead_h"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if variable is not None:
        frame = frame[frame["variable"].astype(str) == str(variable)]
    if lead_h is not None:
        frame = frame[frame["lead_h"] == float(lead_h)]
    if frame.empty:
        hint = ""
        if variable is not None:
            hint = "；该变量可能没算纬向谱（逐变量路由的结果）"
        raise ValueError(
            f"{path} 里没有 variable={variable!r} lead_h={lead_h!r} 的逐波数谱{hint}"
        )
    return frame


class FieldScorePlotter:
    """连续场评分结果的绘图器。"""

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

    def _subset(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        data = _usable(prepare_field_dataframe(df))
        subset = data[data["metric"] == str(metric)]
        if subset.empty:
            raise ValueError(f"长表里没有 metric={metric!r} 的可用数据")
        return subset

    # ------------------------------------------------------------------ #
    def plot_metric_vs_lead(
        self,
        df: pd.DataFrame,
        metric: str,
        *,
        variables: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """一个指标一条线一个变量；**按单位分面**，一个单位一个子图。"""
        subset = self._subset(df, metric)
        if variables is not None:
            subset = subset[subset["variable"].isin([str(name) for name in variables])]
            if subset.empty:
                raise ValueError(f"指定的 variables 在 metric={metric!r} 里一个都没有")

        grouped: Dict[str, List[str]] = {}
        for variable, chunk in subset.groupby("variable", observed=True):
            grouped.setdefault(_unit_of(chunk), []).append(str(variable))
        units = sorted(grouped)

        ref = metric_ref(metric)
        figure, axes = plt.subplots(
            len(units), 1, figsize=(12, max(3.4 * len(units), 3.6)), sharex=True, squeeze=False
        )
        axes = [row[0] for row in axes]
        for axis, unit in zip(axes, units):
            for variable in sorted(grouped[unit]):
                chunk = subset[subset["variable"] == variable].sort_values("lead_h")
                axis.plot(
                    chunk["lead_h"],
                    chunk["value"],
                    marker="o" if len(chunk) <= 20 else None,
                    markersize=3.5,
                    linewidth=1.6,
                    label=variable,
                )
            axis.set_ylabel(_axis_label(metric, unit), fontsize=10)
            axis.grid(alpha=0.3, linewidth=0.6)
            axis.legend(loc="best", fontsize=9, ncol=2, framealpha=0.85)
            if ref is not None:
                axis.axhline(ref, color="#888888", linestyle="--", linewidth=1)
                axis.annotate(
                    f"参考 {ref:g}",
                    xy=(0.995, ref),
                    xycoords=("axes fraction", "data"),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
        axes[-1].set_xlabel("预报时效 (h)", fontsize=10)
        figure.suptitle(title or f"{metric_label(metric)} 随预报时效变化", fontsize=13)
        if len(units) > 1:
            figure.text(
                0.01,
                0.005,
                "注：按单位分面，各面纵轴不同，面与面之间不可直接比大小。",
                fontsize=8,
                color="#555555",
            )
        figure.tight_layout(rect=(0, 0.02, 1, 0.96))
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_metric_heatmap(
        self,
        df: pd.DataFrame,
        metric: str,
        *,
        variables: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """变量 × 时效热力图，填色是**相对首时效的倍数**（单位无关，行间可比）。"""
        subset = self._subset(df, metric)
        if variables is not None:
            subset = subset[subset["variable"].isin([str(name) for name in variables])]
            if subset.empty:
                raise ValueError(f"指定的 variables 在 metric={metric!r} 里一个都没有")

        pivot = subset.pivot_table(
            index="variable", columns="lead_h", values="value", observed=True
        ).sort_index(axis=1)
        if pivot.shape[1] < 2:
            raise ValueError(f"metric={metric!r} 只有 1 个时效，画不了热力图")
        base = pivot.iloc[:, 0].replace(0.0, np.nan)
        ratios = pivot.divide(base, axis=0)
        order = ratios.iloc[:, -1].sort_values(ascending=False).index
        ratios = ratios.loc[order]
        base = base.loc[order]

        labels = []
        for variable in order:
            unit = _unit_of(subset[subset["variable"] == variable])
            suffix = f" {unit}" if unit and unit != "1" else ""
            labels.append(f"{variable}（首时效 {base[variable]:.4g}{suffix}）")

        leads = [float(value) for value in ratios.columns]
        figure, axis = plt.subplots(figsize=(max(10, 0.22 * len(leads) + 4), 0.42 * len(order) + 3))
        mesh = axis.pcolormesh(
            np.asarray(leads),
            np.arange(len(order)),
            ratios.values,
            cmap="viridis",
            shading="nearest",
        )
        axis.set_yticks(np.arange(len(order)))
        axis.set_yticklabels(labels, fontsize=9)
        axis.invert_yaxis()
        axis.set_xlabel("预报时效 (h)", fontsize=10)
        axis.set_title(
            title or f"{metric_label(metric)} 相对首时效的倍数（行标签括号内为首时效绝对值）",
            fontsize=12,
        )
        bar = figure.colorbar(mesh, ax=axis, pad=0.02)
        bar.set_label(f"{metric_label(metric)} 相对首时效的倍数", fontsize=9)
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_spectrum_curve(
        self,
        details: pd.DataFrame,
        variable: str,
        lead_h: float,
        *,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """波数 vs 功率，预报/实况两条线 + 总功率比标注（log 纵轴）。"""
        frame = details.copy()
        frame["group"] = frame["group"].astype(str)
        frame = frame[frame["group"].str.match(_WAVENUMBER_PATTERN)]
        wide = frame.pivot_table(index="group", columns="field", values="value")
        for field in ("power_forecast", "power_observation"):
            if field not in wide.columns:
                raise ValueError(f"明细表里没有 {field} 字段；现有字段: {list(wide.columns)}")
        wide["wavenumber"] = [
            int(str(index).split("=")[1]) for index in wide.index
        ]
        wide = wide.sort_values("wavenumber")
        wide = wide[wide["wavenumber"] > 0]  # k=0 的功率恒为 0，log 轴画不了
        if wide.empty:
            raise ValueError(f"{variable} @ {lead_h:g}h 的谱线只有 k=0，没有可画的波数")

        forecast_total = float(wide["power_forecast"].sum())
        observation_total = float(wide["power_observation"].sum())
        ratio = forecast_total / observation_total if observation_total else float("nan")

        figure, axis = plt.subplots(figsize=(11, 6))
        axis.plot(wide["wavenumber"], wide["power_forecast"], linewidth=1.6, label="预报")
        axis.plot(wide["wavenumber"], wide["power_observation"], linewidth=1.6, label="实况")
        axis.set_yscale("log")
        axis.set_xlabel("纬向波数 k", fontsize=10)
        axis.set_ylabel("功率（对数轴）", fontsize=10)
        axis.grid(alpha=0.3, which="both", linewidth=0.6)
        axis.legend(loc="best", fontsize=10)
        axis.set_title(
            title
            or f"{variable} 纬向谱 @ {lead_h:g}h（总功率比 预报/实况 = {ratio:.3f}）",
            fontsize=12,
        )
        axis.annotate(
            f"总功率比 = {ratio:.3f}\n（全波数求和后相除，与长表里的 "
            f"spectrum_power_ratio 同口径）",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#333333",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def create_report(
        self,
        scores_df: pd.DataFrame,
        output_dir: Path,
        *,
        details_path: Optional[Path] = None,
        spectrum_variable: Optional[str] = None,
        spectrum_lead: Optional[float] = None,
        model_name: str = "model",
        sources: Optional[Dict[str, str]] = None,
        manifest_info: Optional[Dict[str, object]] = None,
        change_description: Optional[str] = None,
    ) -> Dict[str, Path]:
        """出图 + 写 Markdown 报告，返回 ``{名称: 路径}``。

        ``details`` 只在点名要看谱曲线时才读（单变量单时效就有 721 个波数）。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = prepare_field_dataframe(scores_df)
        present = set(_usable(data)["metric"])
        print(f"\n生成 {model_name} 连续场评测报告（输出 {output_dir}）")
        artifacts: Dict[str, Path] = {}

        for metric in SKILL_METRICS:
            if metric not in present:
                print(f"  跳过 {metric}：长表里没有这个指标")
                continue
            path = output_dir / f"{metric}_vs_lead.png"
            self.plot_metric_vs_lead(
                data,
                metric,
                title=f"{model_name} {metric_label(metric)} 随预报时效变化",
                save_path=path,
            )
            plt.close()
            artifacts[f"{metric}_vs_lead"] = path

        if "rmse" in present:
            path = output_dir / "rmse_heatmap.png"
            self.plot_metric_heatmap(
                data,
                "rmse",
                title=f"{model_name} RMSE 相对首时效的倍数（行标签括号内为首时效绝对值）",
                save_path=path,
            )
            plt.close()
            artifacts["rmse_heatmap"] = path

        spectrum_note = None
        if spectrum_variable and details_path:
            details = load_spectrum_details(
                Path(details_path), variable=spectrum_variable, lead_h=spectrum_lead
            )
            lead = (
                float(spectrum_lead)
                if spectrum_lead is not None
                else float(details["lead_h"].max())
            )
            details = details[details["lead_h"] == lead]
            path = output_dir / f"spectrum_curve_{spectrum_variable}_{lead:g}h.png"
            self.plot_spectrum_curve(
                details,
                spectrum_variable,
                lead,
                title=f"{model_name} {spectrum_variable} 纬向谱 @ {lead:g}h",
                save_path=path,
            )
            plt.close()
            artifacts[f"spectrum_curve_{spectrum_variable}_{lead:g}h"] = path
            spectrum_note = (
                f"谱曲线：`{path.name}`（{spectrum_variable} @ {lead:g}h）—— 纵轴对数，"
                f"两条线分别是预报与实况的功率谱，总功率比见右上角标注。"
            )
        elif spectrum_variable and not details_path:
            spectrum_note = (
                "要求了谱曲线但没给明细表路径，未出图"
                "（逐波数谱在 `diagnostics/scores_detail.csv` 里）。"
            )

        report_path = output_dir / "REPORT.md"
        build_field_report(
            data,
            report_path,
            model_name=model_name,
            requested=requested_from(manifest_info or {}),
            manifest_info=manifest_info,
            sources=sources,
            change_description=change_description,
            artifacts=artifacts,
            spectrum_note=spectrum_note,
        )
        artifacts["report"] = report_path
        print(f"✓ 报告完成：{report_path}（{len(artifacts) - 1} 张图 + 1 份报告）")
        if self.font is None:
            print("⚠ 未找到中文字体，图中的中文可能显示为方块；请安装 Noto Sans CJK 或 SimHei")
        return artifacts


