---
name: precipitation-report
description: 生成降水模型对比评估报告，包含深度分析和改进建议
---

# 降水模型评估报告生成 Skill

## 功能

为新训练的降水预报模型生成完整的智能评估报告。

**输入**：模型评估结果目录（包含 CSV）  
**输出**：
- 10+ 张优化图表（刻度自动调整）
- 完整 Markdown 报告（含图表汇总）
- 智能分析报告（自动诊断 + 改进建议）

---

## 使用方法

### 基本用法

```
/precipitation-report
```

然后回答以下问题：
1. 结果目录在哪？（默认：`ref/tiqnqi/xmetai_model_verification/results`）
2. 基准模型是哪个？（默认：`FuXi`）
3. 输出目录？（默认：`reports/comparison`）

### 高级用法（直接指定参数）

```
/precipitation-report \
  --results-dir ref/tiqnqi/xmetai_model_verification/results \
  --baseline FuXi \
  --output reports/fgvp_v2
```

---

## 执行步骤

当用户调用此 skill 时，执行以下步骤：

### 1. 收集参数

```python
# 从用户输入或使用默认值
results_dir = args.get("results_dir") or "ref/tiqnqi/xmetai_model_verification/results"
baseline = args.get("baseline") or "FuXi"
output_dir = args.get("output") or "reports/comparison"
dpi = args.get("dpi") or 150
```

### 2. 加载模型数据

```python
from pathlib import Path
from xmetai_evaluation.visualization.comparison import load_model_results

# 扫描结果目录
result_dirs = {
    "FGVP": Path(results_dir) / "ts_multi_fgvp",
    "FuXi": Path(results_dir) / "ts_multi_fuxi",
    "AIFS": Path(results_dir) / "ts_multi_aifs",
}

# 加载数据
model_data = load_model_results(result_dirs, csv_pattern="ts_*.csv")

# 验证基准模型存在
if baseline not in model_data:
    print(f"错误: 基准模型 {baseline} 不在结果中")
    print(f"可用模型: {list(model_data.keys())}")
    return
```

### 3. 创建输出目录

```python
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
print(f"输出目录: {output_path}")
```

### 4. 生成图表

```python
from xmetai_evaluation.visualization.comparison import ModelComparison
import matplotlib.pyplot as plt

comparison = ModelComparison(
    model_data=model_data,
    baseline=baseline,
    figsize=(14, 8),
    dpi=dpi,
)

# 技巧评分卡
print("\n生成技巧评分卡...")
for threshold in [0.1, 10.0, 25.0]:
    output_file = output_path / f"score_card_{threshold}mm.png"
    fig = comparison.plot_score_card(
        metrics=["TS", "POD", "FAR", "BIAS"],
        threshold=threshold,
        show_improvement=True,
        output_path=output_file,
    )
    plt.close(fig)
    print(f"  ✓ {output_file.name}")

# 雷达图
print("\n生成雷达图...")
for lead in [24, 168, 360]:
    output_file = output_path / f"radar_{lead}h_0.1mm.png"
    try:
        fig = comparison.plot_radar_chart(
            lead_time=lead,
            threshold=0.1,
            output_path=output_file,
        )
        plt.close(fig)
        print(f"  ✓ {output_file.name}")
    except Exception as e:
        print(f"  ✗ {output_file.name}: {e}")

# 技巧提升曲线
print("\n生成技巧提升曲线...")
for metric in ["TS", "POD", "FAR"]:
    output_file = output_path / f"skill_improvement_{metric}_0.1mm.png"
    try:
        fig = comparison.plot_skill_improvement(
            metric=metric,
            threshold=0.1,
            output_path=output_file,
        )
        plt.close(fig)
        print(f"  ✓ {output_file.name}")
    except Exception as e:
        print(f"  ✗ {output_file.name}: {e}")

# 10mm 阈值 TS 提升
output_file = output_path / "skill_improvement_TS_10mm.png"
fig = comparison.plot_skill_improvement(
    metric="TS",
    threshold=10.0,
    output_path=output_file,
)
plt.close(fig)
print(f"  ✓ {output_file.name}")
```

### 5. 生成文本报告

```python
from xmetai_evaluation.visualization.report_generator import (
    ExecutiveSummaryGenerator,
    generate_markdown_report,
)

# 执行摘要
print("\n生成执行摘要...")
summary_gen = ExecutiveSummaryGenerator(model_data, baseline=baseline)
summary_output = output_path / "executive_summary.txt"
summary_gen.generate_summary(
    threshold=0.1,
    window=24,
    lead_range=(24, 168),
    output_path=summary_output,
)
print(f"  ✓ {summary_output.name}")

# Markdown 报告
print("\n生成 Markdown 报告...")
markdown_output = output_path / "REPORT.md"
generate_markdown_report(
    model_data=model_data,
    baseline=baseline,
    figures_dir=output_path,
    output_path=markdown_output,
)
print(f"  ✓ {markdown_output.name}")

# 统计表
print("\n生成统计表...")
stats_output = output_path / "model_statistics.txt"
with open(stats_output, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("模型对比统计摘要\n")
    f.write("=" * 80 + "\n\n")

    for model_name, df in model_data.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"  数据行数: {len(df)}\n")
        f.write(f"  时效范围: {df['lead_h'].min()}h - {df['lead_h'].max()}h\n")
        f.write(f"  阈值范围: {df['threshold_mm'].min()} - {df['threshold_mm'].max()} mm\n")

        # 关键指标平均值
        subset = df[
            (df["threshold_mm"] == 0.1)
            & (df["lead_h"] >= 24)
            & (df["lead_h"] <= 168)
            & (df["window_h"] == 24)
        ]
        if not subset.empty:
            f.write(f"\n  关键指标平均值 (0.1mm, 24-168h):\n")
            for metric in ["TS", "POD", "FAR", "BIAS"]:
                if metric in subset.columns:
                    avg = subset[metric].mean()
                    f.write(f"    {metric}: {avg:.4f}\n")

print(f"  ✓ {stats_output.name}")
```

### 6. 生成智能分析报告

```python
from xmetai_evaluation.visualization.intelligent_report import generate_intelligent_report

print("\n生成智能分析报告...")
for model_name, df in model_data.items():
    if model_name == baseline:
        continue

    intelligent_output = output_path / f"ANALYSIS_{model_name}.md"
    generate_intelligent_report(
        model_data=df,
        baseline_data=model_data[baseline],
        model_name=model_name,
        baseline_name=baseline,
        output_path=intelligent_output,
    )
    print(f"  ✓ {intelligent_output.name}")
```

### 7. 输出总结

```python
print("\n" + "=" * 60)
print(f"✅ 报告生成完成！")
print(f"输出目录: {output_path}")
print("=" * 60)

print("\n生成的文件:")
for file in sorted(output_path.glob("*.png")):
    print(f"  📊 {file.name}")
for file in sorted(output_path.glob("*.md")):
    print(f"  📄 {file.name}")
for file in sorted(output_path.glob("*.txt")):
    print(f"  📝 {file.name}")

print("\n推荐查看顺序:")
print(f"  1. {output_path}/ANALYSIS_*.md  (智能分析)")
print(f"  2. 查看图表")
print(f"  3. {output_path}/REPORT.md  (完整报告)")
```

---

## 完整代码模板

以下是完整的执行代码（当用户调用 skill 时运行）：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
降水模型评估报告生成

此脚本由 /precipitation-report skill 自动执行
"""

from pathlib import Path
import matplotlib.pyplot as plt

from xmetai_evaluation.visualization.comparison import ModelComparison, load_model_results
from xmetai_evaluation.visualization.report_generator import (
    ExecutiveSummaryGenerator,
    generate_markdown_report,
)
from xmetai_evaluation.visualization.intelligent_report import generate_intelligent_report


def main(results_dir: str, baseline: str, output_dir: str, dpi: int = 150):
    """
    生成降水模型对比评估报告
    
    Args:
        results_dir: 结果目录路径
        baseline: 基准模型名称
        output_dir: 输出目录路径
        dpi: 图表分辨率
    """
    
    print("=" * 60)
    print("多模型对比报告生成")
    print("=" * 60)
    
    # 1. 加载模型数据
    print("\n1. 加载模型评估结果...")
    result_dirs = {
        "FGVP": Path(results_dir) / "ts_multi_fgvp",
        "FuXi": Path(results_dir) / "ts_multi_fuxi",
        "AIFS": Path(results_dir) / "ts_multi_aifs",
    }
    
    model_data = load_model_results(result_dirs, csv_pattern="ts_*.csv")
    
    if len(model_data) < 2:
        print("错误: 至少需要两个模型的数据")
        return 1
    
    if baseline not in model_data:
        print(f"错误: 基准模型 {baseline} 不在结果中")
        print(f"可用模型: {list(model_data.keys())}")
        return 1
    
    print(f"\n成功加载 {len(model_data)} 个模型的数据")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建对比可视化对象
    comparison = ModelComparison(
        model_data=model_data,
        baseline=baseline,
        figsize=(14, 8),
        dpi=dpi,
    )
    
    # 2. 生成技巧评分卡
    print("\n2. 生成技巧评分卡...")
    score_card_configs = [
        {"threshold": 0.1, "metrics": ["TS", "POD", "FAR", "BIAS"], "filename": "score_card_0.1mm.png"},
        {"threshold": 10.0, "metrics": ["TS", "POD", "FAR", "BIAS"], "filename": "score_card_10mm.png"},
        {"threshold": 25.0, "metrics": ["TS", "POD", "FAR"], "filename": "score_card_25mm.png"},
    ]
    
    for config in score_card_configs:
        output_file = output_path / config["filename"]
        print(f"   - 生成 {config['filename']} (阈值: {config['threshold']}mm)")
        fig = comparison.plot_score_card(
            metrics=config["metrics"],
            threshold=config["threshold"],
            show_improvement=True,
            output_path=output_file,
        )
        plt.close(fig)
    
    # 3. 生成雷达图
    print("\n3. 生成雷达图...")
    radar_configs = [
        {"lead_time": 24, "threshold": 0.1, "filename": "radar_24h_0.1mm.png"},
        {"lead_time": 168, "threshold": 0.1, "filename": "radar_168h_0.1mm.png"},
        {"lead_time": 360, "threshold": 0.1, "filename": "radar_360h_0.1mm.png"},
    ]
    
    for config in radar_configs:
        output_file = output_path / config["filename"]
        print(f"   - 生成 {config['filename']} ({config['lead_time']}h 时效)")
        try:
            fig = comparison.plot_radar_chart(
                lead_time=config["lead_time"],
                threshold=config["threshold"],
                output_path=output_file,
            )
            plt.close(fig)
        except Exception as e:
            print(f"     警告: 生成失败 - {e}")
    
    # 4. 生成技巧提升曲线
    print("\n4. 生成技巧提升曲线...")
    skill_configs = [
        {"metric": "TS", "threshold": 0.1, "filename": "skill_improvement_TS_0.1mm.png"},
        {"metric": "POD", "threshold": 0.1, "filename": "skill_improvement_POD_0.1mm.png"},
        {"metric": "FAR", "threshold": 0.1, "filename": "skill_improvement_FAR_0.1mm.png"},
        {"metric": "TS", "threshold": 10.0, "filename": "skill_improvement_TS_10mm.png"},
    ]
    
    for config in skill_configs:
        output_file = output_path / config["filename"]
        print(f"   - 生成 {config['filename']} ({config['metric']}, {config['threshold']}mm)")
        try:
            fig = comparison.plot_skill_improvement(
                metric=config["metric"],
                threshold=config["threshold"],
                output_path=output_file,
            )
            plt.close(fig)
        except Exception as e:
            print(f"     警告: 生成失败 - {e}")
    
    # 5. 生成综合统计表
    print("\n5. 生成综合统计表...")
    stats_output = output_path / "model_statistics.txt"
    with open(stats_output, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("模型对比统计摘要\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, df in model_data.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  数据行数: {len(df)}\n")
            f.write(f"  时效范围: {df['lead_h'].min()}h - {df['lead_h'].max()}h\n")
            f.write(f"  阈值范围: {df['threshold_mm'].min()} - {df['threshold_mm'].max()} mm\n")
            
            subset = df[
                (df["threshold_mm"] == 0.1)
                & (df["lead_h"] >= 24)
                & (df["lead_h"] <= 168)
                & (df["window_h"] == 24)
            ]
            if not subset.empty:
                f.write(f"\n  关键指标平均值 (0.1mm, 24-168h):\n")
                for metric in ["TS", "POD", "FAR", "BIAS"]:
                    if metric in subset.columns:
                        avg = subset[metric].mean()
                        f.write(f"    {metric}: {avg:.4f}\n")
    
    print(f"   - 统计表已保存至: {stats_output}")
    
    # 6. 生成执行摘要
    print("\n6. 生成执行摘要...")
    summary_gen = ExecutiveSummaryGenerator(model_data, baseline=baseline)
    summary_output = output_path / "executive_summary.txt"
    summary_gen.generate_summary(
        threshold=0.1, window=24, lead_range=(24, 168), output_path=summary_output
    )
    
    # 7. 生成 Markdown 报告
    print("\n7. 生成 Markdown 报告...")
    markdown_output = output_path / "REPORT.md"
    generate_markdown_report(
        model_data=model_data,
        baseline=baseline,
        figures_dir=output_path,
        output_path=markdown_output,
    )
    
    # 8. 生成智能分析报告（每个模型 vs 基准）
    print("\n8. 生成智能分析报告...")
    for model_name, df in model_data.items():
        if model_name == baseline:
            continue
        
        intelligent_output = output_path / f"ANALYSIS_{model_name}.md"
        print(f"   - 分析 {model_name} vs {baseline}")
        generate_intelligent_report(
            model_data=df,
            baseline_data=model_data[baseline],
            model_name=model_name,
            baseline_name=baseline,
            output_path=intelligent_output,
        )
    
    print("\n" + "=" * 60)
    print(f"✅ 报告生成完成！输出目录: {output_path}")
    print("=" * 60)
    
    print("\n生成的图表:")
    for file in sorted(output_path.glob("*.png")):
        print(f"  - {file.name}")
    
    print("\n推荐查看顺序:")
    for file in sorted(output_path.glob("ANALYSIS_*.md")):
        print(f"  1. {file.name} (智能分析)")
    print(f"  2. 查看图表")
    print(f"  3. REPORT.md (完整报告)")
    
    return 0


if __name__ == "__main__":
    import sys
    
    # 默认参数
    results_dir = "ref/tiqnqi/xmetai_model_verification/results"
    baseline = "FuXi"
    output_dir = "reports/comparison"
    dpi = 150
    
    # 从命令行参数解析（如果有）
    import argparse
    parser = argparse.ArgumentParser(description="生成降水模型对比评估报告")
    parser.add_argument("--results-dir", type=str, default=results_dir)
    parser.add_argument("--baseline", type=str, default=baseline)
    parser.add_argument("--output", type=str, default=output_dir)
    parser.add_argument("--dpi", type=int, default=dpi)
    
    args = parser.parse_args()
    
    sys.exit(main(
        results_dir=args.results_dir,
        baseline=args.baseline,
        output_dir=args.output,
        dpi=args.dpi,
    ))
```

---

## 使用示例

### 示例 1: 使用默认配置

```
用户: /precipitation-report

Claude 执行:
- 使用默认结果目录
- 使用 FuXi 作为基准
- 输出到 reports/comparison
- 生成所有报告
```

### 示例 2: 自定义配置

```
用户: /precipitation-report --baseline AIFS --output reports/fgvp_v2

Claude 执行:
- 使用 AIFS 作为基准
- 输出到 reports/fgvp_v2
- 生成所有报告
```

### 示例 3: 交互式

```
用户: /precipitation-report

Claude: 请提供以下信息（直接回车使用默认值）:
        1. 结果目录 [ref/tiqnqi/xmetai_model_verification/results]: 
        2. 基准模型 [FuXi]: AIFS
        3. 输出目录 [reports/comparison]: reports/my_report

Claude 执行: 根据用户输入生成报告
```

---

## 诊断规则说明

### 优势识别

- TS 提升 > 2% → 整体技巧提升
- FAR 降低 > 3% → 空报控制改善
- POD 提升 > 2% → 命中率提升
- 自动识别最佳时效（TS 提升最大的 3 个）

### 问题诊断

**P0 - 严重问题**：
- BIAS > 2.0 → 严重系统性高报
- BIAS < 0.5 → 严重系统性低报
- FAR > 0.7 → 空报率过高

**P1 - 重要问题**：
- BIAS > 1.5 → 系统性高报
- FAR > 0.6 → 空报率偏高
- POD < 0.6 → 漏报率偏高
- TS@25mm < 0.15 → 强降水能力弱

### 改进建议

根据诊断自动生成 P0/P1/P2 优先级建议，包括：
- 具体方案
- 预期收益
- 实现难度
- 代码示例（如果适用）

---

## 注意事项

1. **确保数据存在**：结果目录下需要有对应的 CSV 文件
2. **图表生成失败**：某些时效可能数据不足，会跳过但不影响其他图表
3. **基准模型**：必须在结果目录中存在
4. **输出目录**：会自动创建，已存在则覆盖同名文件

---

**维护者**: 沈哲文 (szw)  
**最后更新**: 2025-09-10
