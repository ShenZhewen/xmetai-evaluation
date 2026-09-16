#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连续场评分（RMSE / ACC / 活跃度 / 纬向谱）的可视化。

输入：``weather_field_scores`` 流程产出的 ``scores.csv`` 长表，列至少包含
``variable / metric / lead_h / value``（``unit`` 与 ``status`` 有则用、没有则兜底）。

输出：若干 PNG，以及一份 Markdown 报告（报告正文见 :mod:`field_report`）。

五件与"一张图一根轴画到底"不同的做法：

* **折线画的是跨起报均值，不是每一条起报**。长表一行一个起报（本仓库实测 184 个），
  把 ``value`` 按 ``lead_h`` 排序直接连线，画出来是 184 条曲线首尾相接地穿插，
  看着像"带误差棒的粗线"，其实线的走向由起报的先后顺序决定、和均值无关——
  同一个时效里的 184 个点被按原顺序连成一串竖线，再斜着连到下一个时效。
  真正的形态（比如 RMSE 到底单不单调）只能从跨起报均值上看。离散程度改画分位带；
* **RMSE 按单位分面**。同一张图里 z500 是 ``m^2/s^2``、t2m 是 ``K``、msl 是 ``Pa``，
  混在一根纵轴上比大小是错的，所以一个单位一个子图；
* **热力图用"相对首时效的倍数"**。同理，跨变量的原始值不可比；除过首时效之后
  无量纲，行与行才可比，行标签里同时写上首时效的绝对值，尺度不丢；
* **谱曲线优先读 ``summary.csv``**（``scope=wavenumber`` 档），那是一张已经跨起报
  平均好的逐波数表，一个变量一次画全。只有要看**指定时效**的谱时才去读
  ``diagnostics/scores_detail.csv``——本仓库那份是 492MB / 1.67 亿行，为一张图读它不划算；
* **参考线只在有理想值时画**（语义表里的 ``ref_result``）：比值类画 ``y=1``，ACC 画 ``y=1``，
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

from visualization.field_report import (
    SKILL_METRICS,
    build_field_report,
    metric_label,
    metric_ref,
    requested_from,
)
from visualization.precipitation_plots import setup_chinese_font

REQUIRED_COLUMNS = ("lead_h", "variable", "metric", "value")

#: ``details`` 里逐波数谱的 group 形如 ``k=0`` / ``k=720``；``summary`` 是总功率行。
_WAVENUMBER_PATTERN = r"^k=\d+$"

#: 读大明细表时只取这几列（实测 90 万行，全读没必要）。
_DETAIL_COLUMNS = ("variable", "lead_h", "group", "field", "value")

#: 折线图里跨起报离散带取的分位。用 10–90 而不是 min–max：后者由单次极端起报
#: 决定上下沿，带子会宽得看不出形态。
_SPREAD_QUANTILES = (0.10, 0.90)

#: ``summary.csv`` 的逐波数档只取这几列。
_SUMMARY_SPECTRUM_COLUMNS = ("scope", "variable", "wavenumber", "field", "value")

#: 赤道周长（km）。纬向波数 k 的波长按 ``40075 / k`` 算，与参考实现的
#: ``mean_spectrum_<var>.csv`` 里 ``wavelength_km`` 列同口径（k=1 → 40075km）。
EQUATOR_CIRCUMFERENCE_KM = 40075.0


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


def lead_aggregate(chunk: pd.DataFrame) -> pd.DataFrame:
    """把「逐起报」的行压成「逐时效」的均值 / 分位带 / 起报数。

    口径就是 ``summary.csv`` 里 ``scope=lead`` 那一档（跨起报算术平均），
    所以图上的线能和落盘的数字对上。

    Args:
        chunk: 同一 ``变量 × 指标`` 的可用行，要有 ``lead_h`` 与 ``value``。

    Returns:
        以 ``lead_h`` 升序为索引的 DataFrame，列为
        ``mean`` / ``low`` / ``high`` / ``n``（``n`` 是该时效的起报数）。
    """
    grouped = chunk.groupby("lead_h", observed=True)["value"]
    stats = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "low": grouped.quantile(_SPREAD_QUANTILES[0]),
            "high": grouped.quantile(_SPREAD_QUANTILES[1]),
            "n": grouped.size(),
        }
    )
    return stats.sort_index()


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


def load_wavenumber_spectra(path: Path) -> pd.DataFrame:
    """读 ``summary.csv`` 里 ``scope=wavenumber`` 的逐波数谱曲线。

    这是**已经跨起报（并跨时效）平均**好的一张表，跟参考实现的
    ``mean_spectrum_<var>.csv`` 同口径，所以一个变量一张谱、一次读全，
    不必为了看一眼谱去啃几百 MB 的 ``diagnostics/scores_detail.csv``。

    两条路的差别要说清楚：这里的谱是**全时段平均**的，看不出谱随预报时效怎么退化；
    要比某个具体时效的谱，用 :func:`load_spectrum_details` 那条路。

    Args:
        path: ``summary.csv`` 的路径。

    Returns:
        长表，列为 ``variable / wavenumber / field / value``，
        ``field`` ∈ {``power_forecast``, ``power_observation``}。

    Raises:
        ValueError: 表里没有 ``scope=wavenumber`` 的行，或缺少必需列。
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} 不存在")
    frame = pd.read_csv(path, usecols=lambda name: name in _SUMMARY_SPECTRUM_COLUMNS)
    missing = [name for name in _SUMMARY_SPECTRUM_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} 缺少列 {missing}；现有列: {list(frame.columns)}")
    frame = frame[frame["scope"].astype(str) == "wavenumber"].copy()
    if frame.empty:
        raise ValueError(
            f"{path} 里没有 scope=wavenumber 的行：这条流程的配置没写 field_summary，"
            f"或者纬度方向的格点数不足以做纬向谱"
        )
    frame["variable"] = frame["variable"].astype(str)
    frame["field"] = frame["field"].astype(str)
    for column in ("wavenumber", "value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["wavenumber", "value"]).sort_values(["variable", "wavenumber"])


def spectrum_series(spectra: pd.DataFrame, variable: str) -> pd.DataFrame:
    """把一个变量的谱帧整成 ``wavenumber × {power_forecast, power_observation}``。

    Raises:
        ValueError: 该变量缺预报或实况其中一条谱线。
    """
    subset = spectra[spectra["variable"].astype(str) == str(variable)]
    wide = subset.pivot_table(
        index="wavenumber", columns="field", values="value", aggfunc="mean", observed=True
    ).sort_index()
    for field in ("power_forecast", "power_observation"):
        if field not in wide.columns:
            raise ValueError(
                f"{variable} 的谱只有 {list(wide.columns)}，缺 {field}，画不了两条线"
            )
    return wide


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
        """一个变量一条线；**按单位分面**，一个单位一个子图。

        线是**跨起报均值**（口径同 ``summary.csv`` 的 ``scope=lead``），
        阴影是跨起报的 10–90 分位。每个时效只有 1 个起报时不画阴影——
        带子宽度恒为 0，画了只会让人以为"离散很小"。
        """
        subset = self._subset(df, metric)
        if variables is not None:
            subset = subset[subset["variable"].isin([str(name) for name in variables])]
            if subset.empty:
                raise ValueError(f"指定的 variables 在 metric={metric!r} 里一个都没有")

        grouped: Dict[str, List[str]] = {}
        for variable, chunk in subset.groupby("variable", observed=True):
            grouped.setdefault(_unit_of(chunk), []).append(str(variable))
        units = sorted(grouped)

        cycle = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"]))
        ref = metric_ref(metric)
        spread_drawn = False
        most_inits = 0
        figure, axes = plt.subplots(
            len(units), 1, figsize=(12, max(3.4 * len(units), 3.6)), sharex=True, squeeze=False
        )
        axes = [row[0] for row in axes]
        for axis, unit in zip(axes, units):
            for index, variable in enumerate(sorted(grouped[unit])):
                stats = lead_aggregate(subset[subset["variable"] == variable])
                color = cycle[index % len(cycle)]
                most_inits = max(most_inits, int(stats["n"].max()))
                if bool((stats["n"] > 1).any()):
                    axis.fill_between(
                        stats.index,
                        stats["low"],
                        stats["high"],
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )
                    spread_drawn = True
                axis.plot(
                    stats.index,
                    stats["mean"],
                    color=color,
                    marker="o" if len(stats) <= 20 else None,
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
        notes = []
        if spread_drawn:
            notes.append(
                f"实线为跨起报均值，阴影为跨起报 10–90 分位（每个时效最多 {most_inits} 个起报）。"
            )
        else:
            notes.append("每个时效只有 1 个起报，只有均值曲线，没有分位带。")
        if len(units) > 1:
            notes.append("按单位分面，各面纵轴不同，面与面之间不可直接比大小。")
        figure.text(
            0.01, 0.005, "注：" + "".join(notes), fontsize=8, color="#555555"
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
        """变量 × 时效热力图，填色是**相对首时效的倍数**（单位无关，行间可比）。

        格子里是**跨起报均值**（与折线图同一口径），否则一个格子要装 184 个数。
        """
        subset = self._subset(df, metric)
        if variables is not None:
            subset = subset[subset["variable"].isin([str(name) for name in variables])]
            if subset.empty:
                raise ValueError(f"指定的 variables 在 metric={metric!r} 里一个都没有")

        pivot = subset.pivot_table(
            index="variable", columns="lead_h", values="value", aggfunc="mean", observed=True
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
    def plot_spectrum_panel(
        self,
        spectra: pd.DataFrame,
        *,
        variables: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """一张多面板图：一个变量一个子图，预报/实况两条谱线（log 纵轴）。

        输入是 :func:`load_wavenumber_spectra` 读出来的逐波数表（跨起报平均），
        每个子图右上角标该变量的**全波数总功率比**——它和长表里的
        ``spectrum_power_ratio`` 同口径，所以图上的线和表里的数字对得上。
        """
        names = sorted(
            str(name) for name in (variables if variables is not None else spectra["variable"].unique())
        )
        if not names:
            raise ValueError("谱表里没有任何变量")

        columns = min(3, len(names))
        rows = -(-len(names) // columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(5.0 * columns, 3.6 * rows), squeeze=False
        )
        flat = [axis for row in axes for axis in row]
        for axis, variable in zip(flat, names):
            wide = spectrum_series(spectra, variable)
            wide = wide[wide.index > 0]  # k=0 的功率恒为 0，log 轴画不了
            forecast_total = float(wide["power_forecast"].sum())
            observation_total = float(wide["power_observation"].sum())
            ratio = forecast_total / observation_total if observation_total else float("nan")
            axis.plot(wide.index, wide["power_forecast"], linewidth=1.4, label="预报")
            axis.plot(wide.index, wide["power_observation"], linewidth=1.4, label="实况")
            axis.set_yscale("log")
            axis.set_title(f"{variable}（总功率比 {ratio:.3f}）", fontsize=11)
            axis.grid(alpha=0.3, which="both", linewidth=0.6)
        for axis in flat[len(names):]:
            axis.axis("off")
        for axis in flat[: len(names)][-columns:]:
            axis.set_xlabel("纬向波数 k", fontsize=9)
        for index, axis in enumerate(flat[: len(names)]):
            if index % columns == 0:
                axis.set_ylabel("功率（对数轴）", fontsize=9)
        flat[0].legend(loc="lower left", fontsize=9)
        figure.suptitle(title or "纬向功率谱（跨起报平均）", fontsize=13)
        figure.text(
            0.01,
            0.005,
            "注：口径同 `mean_spectrum_<var>.csv`——跨起报、跨时效平均后的功率谱，"
            "看不出谱随预报时效怎么退化；要指定时效的谱请读 scores_detail.csv。"
            "标题里的总功率比是**全波数求和后**再相除，它正常不代表分布正常。",
            fontsize=8,
            color="#555555",
        )
        figure.tight_layout(rect=(0, 0.03, 1, 0.96))
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def create_report(
        self,
        scores_df: pd.DataFrame,
        output_dir: Path,
        *,
        summary_path: Optional[Path] = None,
        details_path: Optional[Path] = None,
        spectrum_variable: Optional[str] = None,
        spectrum_lead: Optional[float] = None,
        model_name: str = "model",
        sources: Optional[Dict[str, str]] = None,
        manifest_info: Optional[Dict[str, object]] = None,
        change_description: Optional[str] = None,
    ) -> Dict[str, Path]:
        """出图 + 写 Markdown 报告，返回 ``{名称: 路径}``。

        ``summary_path`` 给了就出一张全变量的纬向谱面板，并让报告多一节分波段
        功率比；``details_path`` 只在点名要看**指定时效**的谱曲线时才读
        （单变量单时效就有 721 个波数，本仓库那份明细表是 492MB）。
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

        spectra = None
        if summary_path:
            try:
                spectra = load_wavenumber_spectra(Path(summary_path))
            except ValueError as error:
                print(f"  跳过纬向谱面板：{error}")
        if spectra is not None and not spectra.empty:
            path = output_dir / "spectrum_panel.png"
            self.plot_spectrum_panel(
                spectra,
                title=f"{model_name} 纬向功率谱（跨起报平均）",
                save_path=path,
            )
            plt.close()
            artifacts["spectrum_panel"] = path

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
            spectra=spectra,
        )
        artifacts["report"] = report_path
        # 别在这两行用 ✓ / ⚠：Windows 控制台默认 GBK，编码不了这两个字符。报告和图
        # 这时**已经写完了**，却会在最后一行抛 UnicodeEncodeError、进程非 0 退出——
        # 看着像失败，其实什么都没坏。中文字符本身 GBK 是有的，只有这些符号会炸。
        print(f"报告完成：{report_path}（{len(artifacts) - 1} 张图 + 1 份报告）")
        if self.font is None:
            print("未找到中文字体，图中的中文可能显示为方块；请安装 Noto Sans CJK 或 SimHei")
        return artifacts


