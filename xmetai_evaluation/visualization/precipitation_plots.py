#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""降水评测结果可视化模块

提供专业的气象评测报告图表，包括：
- 时效-指标曲线
- 阈值-指标柱状图
- 性能图（Performance Diagram）
- 热力图
- Taylor 图
- 多模型对比图

所有图表遵循气象可视化最佳实践。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.ticker import MultipleLocator

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 专业配色方案（色盲友好）
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
}

THRESHOLD_COLORS = {
    '≥0.1': COLORS['blue'],
    '≥10': COLORS['green'],
    '≥25': COLORS['orange'],
    '≥50': COLORS['red'],
    '≥100': COLORS['purple'],
    '≥250': COLORS['brown'],
}


class PrecipitationPlotter:
    """降水评测可视化工具类"""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Args:
            style: matplotlib 样式（'seaborn-v0_8-darkgrid', 'bmh', 'ggplot'）
        """
        plt.style.use(style)

    def plot_metric_vs_lead(
        self,
        df: pd.DataFrame,
        metric: str = 'TS',
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """绘制指标随时效变化曲线

        Args:
            df: 评测结果 DataFrame
            metric: 指标名称（'TS', 'POD', 'FAR', 'BIAS'）
            figsize: 图片尺寸
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize)

        for grade in sorted(df['grade'].unique()):
            subset = df[df['grade'] == grade].sort_values('lead_h')
            ax.plot(
                subset['lead_h'],
                subset[metric],
                marker='o',
                linewidth=2,
                markersize=6,
                label=grade,
                color=THRESHOLD_COLORS.get(grade, 'gray'),
            )

        ax.set_xlabel('预报时效 (小时)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(
            title or f'{metric} 随预报时效变化',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )
        ax.legend(title='降水等级', loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)

        # 根据指标类型设置 Y 轴范围
        if metric in ['TS', 'POD', 'FAR', '漏报率']:
            ax.set_ylim(0, 1)
        elif metric == 'BIAS':
            ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='理想值')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

        return fig

    def plot_metric_by_threshold(
        self,
        df: pd.DataFrame,
        lead_h: int = 24,
        metrics: List[str] = ['TS', 'POD', 'FAR'],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """绘制不同阈值的指标柱状对比图

        Args:
            df: 评测结果 DataFrame
            lead_h: 固定时效
            metrics: 要展示的指标列表
            figsize: 图片尺寸
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        subset = df[df['lead_h'] == lead_h].sort_values('threshold_mm')

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(subset))
        width = 0.25

        for i, metric in enumerate(metrics):
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                subset[metric],
                width,
                label=metric,
                alpha=0.8,
            )

        ax.set_xlabel('降水阈值 (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('指标值', fontsize=12, fontweight='bold')
        ax.set_title(
            f'预报时效 {lead_h}h 不同阈值的指标对比',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(subset['grade'])
        ax.legend(loc='upper right', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

        return fig

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        metric: str = 'TS',
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'RdYlGn',
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """绘制时效-阈值热力图

        Args:
            df: 评测结果 DataFrame
            metric: 指标名称
            figsize: 图片尺寸
            cmap: 色图名称
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        # 构建透视表
        pivot = df.pivot(index='grade', columns='lead_h', values=metric)

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            cbar_kws={'label': metric},
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            vmin=0,
            vmax=1 if metric != 'BIAS' else 2,
        )

        ax.set_xlabel('预报时效 (小时)', fontsize=12, fontweight='bold')
        ax.set_ylabel('降水等级', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{metric} 热力图 (时效 × 阈值)',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

        return fig

    def plot_performance_diagram(
        self,
        df: pd.DataFrame,
        lead_h: int = 24,
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """绘制性能图（Performance Diagram）

        X轴: 1-FAR (Success Ratio)
        Y轴: POD
        等值线: TS 和 BIAS

        Args:
            df: 评测结果 DataFrame
            lead_h: 固定时效
            figsize: 图片尺寸
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        subset = df[df['lead_h'] == lead_h]

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制 TS 等值线
        sr_range = np.linspace(0, 1, 100)
        pod_range = np.linspace(0, 1, 100)
        SR, POD = np.meshgrid(sr_range, pod_range)

        # TS = 1 / (1/POD + 1/SR - 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            TS = 1.0 / (1.0 / POD + 1.0 / SR - 1.0)
            TS = np.where((POD > 0) & (SR > 0), TS, np.nan)

        # 绘制 TS 等值线
        cs_ts = ax.contour(
            SR, POD, TS,
            levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            colors='gray',
            linewidths=1,
            linestyles='--',
            alpha=0.5,
        )
        ax.clabel(cs_ts, inline=True, fontsize=8, fmt='TS=%.1f')

        # 绘制 BIAS 等值线
        bias_levels = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        for bias in bias_levels:
            pod_bias = np.linspace(0, 1, 100)
            sr_bias = pod_bias / bias
            sr_bias = np.clip(sr_bias, 0, 1)
            linestyle = '-' if bias == 1.0 else ':'
            linewidth = 2 if bias == 1.0 else 1
            ax.plot(
                sr_bias, pod_bias,
                color='blue',
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.4,
            )
            # 标注 BIAS 值
            if bias <= 1:
                idx = int(len(sr_bias) * 0.95)
            else:
                idx = int(len(sr_bias) * 0.7)
            ax.text(
                sr_bias[idx], pod_bias[idx],
                f'{bias}',
                fontsize=8,
                color='blue',
                alpha=0.6,
            )

        # 绘制数据点
        for _, row in subset.iterrows():
            sr = 1 - row['FAR']
            pod = row['POD']
            ax.scatter(
                sr, pod,
                s=150,
                color=THRESHOLD_COLORS.get(row['grade'], 'gray'),
                edgecolors='black',
                linewidths=1.5,
                label=row['grade'],
                zorder=10,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=12, fontweight='bold')
        ax.set_ylabel('POD (命中率)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'性能图 (预报时效 {lead_h}h)',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )
        ax.legend(title='降水等级', loc='lower left', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

        # 添加说明文字
        ax.text(
            0.98, 0.02,
            'TS 等值线 (灰色虚线)\nBIAS 等值线 (蓝色虚线)\nBIAS=1 为理想',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

        return fig

    def plot_multi_model_comparison(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        metric: str = 'TS',
        lead_h: int = 24,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """绘制多模型对比图

        Args:
            model_dfs: 模型名称 -> DataFrame 字典
            metric: 指标名称
            lead_h: 固定时效
            figsize: 图片尺寸
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 获取所有模型在该时效的数据
        all_grades = None
        model_data = {}

        for model_name, df in model_dfs.items():
            subset = df[df['lead_h'] == lead_h].sort_values('threshold_mm')
            if all_grades is None:
                all_grades = subset['grade'].values
            model_data[model_name] = subset[metric].values

        x = np.arange(len(all_grades))
        width = 0.8 / len(model_dfs)

        for i, (model_name, values) in enumerate(model_data.items()):
            offset = (i - len(model_dfs) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=model_name,
                alpha=0.8,
            )

        ax.set_xlabel('降水阈值 (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(
            f'多模型 {metric} 对比 (预报时效 {lead_h}h)',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_grades)
        ax.legend(title='模型', loc='upper right', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1 if metric != 'BIAS' else 2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

        return fig

    def create_report(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        model_name: str = 'FuXi',
    ) -> None:
        """生成完整评测报告（所有图表）

        Args:
            df: 评测结果 DataFrame
            output_dir: 输出目录
            model_name: 模型名称
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n生成 {model_name} 评测报告...")
        print("=" * 60)

        # 1. TS 时效曲线
        print("1/6 绘制 TS 时效曲线...")
        self.plot_metric_vs_lead(
            df, metric='TS',
            title=f'{model_name} TS 评分随预报时效变化',
            save_path=output_dir / f'{model_name}_TS_vs_lead.png',
        )
        plt.close()

        # 2. POD 时效曲线
        print("2/6 绘制 POD 时效曲线...")
        self.plot_metric_vs_lead(
            df, metric='POD',
            title=f'{model_name} 命中率(POD)随预报时效变化',
            save_path=output_dir / f'{model_name}_POD_vs_lead.png',
        )
        plt.close()

        # 3. FAR 时效曲线
        print("3/6 绘制 FAR 时效曲线...")
        self.plot_metric_vs_lead(
            df, metric='FAR',
            title=f'{model_name} 空报率(FAR)随预报时效变化',
            save_path=output_dir / f'{model_name}_FAR_vs_lead.png',
        )
        plt.close()

        # 4. 24h 阈值对比
        print("4/6 绘制 24h 阈值对比...")
        self.plot_metric_by_threshold(
            df, lead_h=24,
            save_path=output_dir / f'{model_name}_threshold_comparison_24h.png',
        )
        plt.close()

        # 5. TS 热力图
        print("5/6 绘制 TS 热力图...")
        self.plot_heatmap(
            df, metric='TS',
            save_path=output_dir / f'{model_name}_TS_heatmap.png',
        )
        plt.close()

        # 6. 性能图
        print("6/6 绘制性能图...")
        self.plot_performance_diagram(
            df, lead_h=24,
            save_path=output_dir / f'{model_name}_performance_diagram_24h.png',
        )
        plt.close()

        print("=" * 60)
        print(f"✓ 报告生成完成，输出目录: {output_dir}")
        print(f"  - 共生成 6 张图表")


def example_usage():
    """示例用法"""
    # 读取评测结果
    df = pd.read_csv("output/ts_fuxi.csv")

    # 创建绘图对象
    plotter = PrecipitationPlotter()

    # 生成完整报告
    plotter.create_report(
        df,
        output_dir=Path("output/figures"),
        model_name="FuXi",
    )

    # 或者单独绘制某一张图
    plotter.plot_metric_vs_lead(df, metric='TS', save_path="ts_curve.png")

    # 多模型对比
    fuxi_df = pd.read_csv("output/ts_fuxi.csv")
    fgvp_df = pd.read_csv("output/ts_fgvp.csv")

    plotter.plot_multi_model_comparison(
        model_dfs={'FuXi': fuxi_df, 'FGVP': fgvp_df},
        metric='TS',
        lead_h=24,
        save_path="model_comparison.png",
    )


if __name__ == "__main__":
    example_usage()
