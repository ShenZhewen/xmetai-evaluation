#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成降水评测报告图表

使用方法：
    python generate_plots.py --csv output/ts_fuxi.csv --output figures/ --model FuXi

支持的图表类型：
    - TS/POD/FAR 时效曲线
    - 阈值对比柱状图
    - TS 热力图
    - 性能图（Performance Diagram）
    - 多模型对比
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from xmetai_evaluation.visualization import PrecipitationPlotter


def main():
    parser = argparse.ArgumentParser(description="生成降水评测报告图表")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="评测结果 CSV 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures",
        help="图表输出目录（默认: figures/）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FuXi",
        help="模型名称（默认: FuXi）",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=['all', 'ts', 'pod', 'far', 'threshold', 'heatmap', 'performance'],
        default='all',
        help="图表类型（默认: all 生成所有图表）",
    )
    parser.add_argument(
        "--lead",
        type=int,
        default=24,
        help="固定时效（用于阈值对比和性能图，默认: 24）",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs='+',
        help="多模型对比：提供多个 CSV 文件路径（格式: 模型名:路径）",
    )

    args = parser.parse_args()

    # 读取数据
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"错误: 文件不存在 {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    print(f"已读取: {csv_path}")
    print(f"  - 数据行数: {len(df)}")
    print(f"  - 时效范围: {df['lead_h'].min()}h ~ {df['lead_h'].max()}h")
    print(f"  - 降水等级: {', '.join(df['grade'].unique())}")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建绘图对象
    plotter = PrecipitationPlotter()

    # 根据类型生成图表
    if args.type == 'all':
        print("\n生成完整报告...")
        plotter.create_report(df, output_dir, model_name=args.model)

    elif args.type == 'ts':
        print("\n生成 TS 时效曲线...")
        plotter.plot_metric_vs_lead(
            df,
            metric='TS',
            title=f'{args.model} TS 评分随预报时效变化',
            save_path=output_dir / f'{args.model}_TS_vs_lead.png',
        )

    elif args.type == 'pod':
        print("\n生成 POD 时效曲线...")
        plotter.plot_metric_vs_lead(
            df,
            metric='POD',
            title=f'{args.model} 命中率(POD)随预报时效变化',
            save_path=output_dir / f'{args.model}_POD_vs_lead.png',
        )

    elif args.type == 'far':
        print("\n生成 FAR 时效曲线...")
        plotter.plot_metric_vs_lead(
            df,
            metric='FAR',
            title=f'{args.model} 空报率(FAR)随预报时效变化',
            save_path=output_dir / f'{args.model}_FAR_vs_lead.png',
        )

    elif args.type == 'threshold':
        print(f"\n生成阈值对比图 (时效 {args.lead}h)...")
        plotter.plot_metric_by_threshold(
            df,
            lead_h=args.lead,
            save_path=output_dir / f'{args.model}_threshold_comparison_{args.lead}h.png',
        )

    elif args.type == 'heatmap':
        print("\n生成 TS 热力图...")
        plotter.plot_heatmap(
            df,
            metric='TS',
            save_path=output_dir / f'{args.model}_TS_heatmap.png',
        )

    elif args.type == 'performance':
        print(f"\n生成性能图 (时效 {args.lead}h)...")
        plotter.plot_performance_diagram(
            df,
            lead_h=args.lead,
            save_path=output_dir / f'{args.model}_performance_diagram_{args.lead}h.png',
        )

    # 多模型对比
    if args.compare:
        print("\n生成多模型对比图...")
        model_dfs = {}
        for item in args.compare:
            if ':' not in item:
                print(f"警告: 跳过格式错误的参数 '{item}'（应为 '模型名:路径'）")
                continue
            model_name, csv_path_str = item.split(':', 1)
            csv_path = Path(csv_path_str)
            if not csv_path.exists():
                print(f"警告: 跳过不存在的文件 {csv_path}")
                continue
            model_dfs[model_name] = pd.read_csv(csv_path)
            print(f"  - 已加载: {model_name} ({csv_path})")

        if len(model_dfs) > 1:
            plotter.plot_multi_model_comparison(
                model_dfs,
                metric='TS',
                lead_h=args.lead,
                save_path=output_dir / f'model_comparison_TS_{args.lead}h.png',
            )
            print(f"✓ 多模型对比图已保存")
        else:
            print("需要至少 2 个模型才能进行对比")

    print(f"\n✓ 完成！图表已保存到: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
