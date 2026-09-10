# 可视化使用指南

## 快速开始

### 生成完整报告

```bash
# 生成所有图表
python scripts/generate_plots.py \
    --csv output/ts_fuxi.csv \
    --output figures/ \
    --model FuXi

# 输出 6 张图表：
#   - FuXi_TS_vs_lead.png       (TS 时效曲线)
#   - FuXi_POD_vs_lead.png      (POD 时效曲线)
#   - FuXi_FAR_vs_lead.png      (FAR 时效曲线)
#   - FuXi_threshold_comparison_24h.png  (24h 阈值对比)
#   - FuXi_TS_heatmap.png       (TS 热力图)
#   - FuXi_performance_diagram_24h.png   (性能图)
```

### 单独生成某一张图

```bash
# 只生成 TS 时效曲线
python scripts/generate_plots.py \
    --csv output/ts_fuxi.csv \
    --output figures/ \
    --model FuXi \
    --type ts

# 只生成性能图（指定时效 48h）
python scripts/generate_plots.py \
    --csv output/ts_fuxi.csv \
    --output figures/ \
    --model FuXi \
    --type performance \
    --lead 48
```

### 多模型对比

```bash
python scripts/generate_plots.py \
    --csv output/ts_fuxi.csv \
    --output figures/ \
    --model FuXi \
    --compare FuXi:output/ts_fuxi.csv FGVP:output/ts_fgvp.csv AIFS:output/ts_aifs.csv
```

## 图表类型详解

### 1. TS/POD/FAR 时效曲线

**用途**：展示预报技巧随时效的衰减

**特点**：
- 每条线代表一个降水阈值
- 横轴：预报时效（小时）
- 纵轴：指标值 [0, 1]
- 可清楚看出哪个时效、哪个阈值预报最好

**解读**：
- 曲线下降 → 预报技巧随时效衰减
- 曲线平缓 → 预报稳定
- 不同阈值的曲线分离度 → 不同降水强度的可预报性差异

**示例输出**：
```
FuXi_TS_vs_lead.png
FuXi_POD_vs_lead.png
FuXi_FAR_vs_lead.png
```

---

### 2. 阈值对比柱状图

**用途**：对比不同降水等级的预报能力

**特点**：
- 固定时效（如 24h），展示不同阈值的指标
- 并排显示 TS、POD、FAR
- 直观看出哪个阈值预报困难

**解读**：
- 柱子高度 → 预报能力
- 强降水（≥100mm）柱子矮 → 预报难度大
- TS、POD、FAR 三者平衡 → 综合能力

**示例输出**：
```
FuXi_threshold_comparison_24h.png
```

---

### 3. TS 热力图

**用途**：全局概览，快速识别最佳时效-阈值组合

**特点**：
- 行：降水阈值
- 列：预报时效
- 颜色：TS 值（绿色=好，红色=差）
- 每个格子标注数值

**解读**：
- 一眼看出"热点"（高 TS 区域）
- 识别"冷点"（低 TS 区域）
- 横向看：某阈值随时效的变化
- 纵向看：某时效下不同阈值的表现

**示例输出**：
```
FuXi_TS_heatmap.png
```

---

### 4. 性能图（Performance Diagram）⭐

**用途**：气象业务评估标准图，同时展示多个指标

**特点**：
- X 轴：Success Ratio (1 - FAR)
- Y 轴：POD
- 灰色虚线：TS 等值线
- 蓝色虚线：BIAS 等值线
- 数据点：不同降水阈值

**解读**：
- 右上角 = 理想区（高 POD + 低 FAR）
- BIAS=1 线（实线）= 频率无偏
- 点越靠右上 → 预报越好
- 点在 BIAS=1 线上 → 频率偏差小

**使用场景**：
- 评估预报整体质量
- 识别系统性偏差
- 多模型对比（不同颜色点）

**示例输出**：
```
FuXi_performance_diagram_24h.png
```

---

### 5. 多模型对比图

**用途**：对比不同模型的预报能力

**特点**：
- 固定时效，不同模型并排
- 每个阈值一组柱子
- 颜色区分模型

**解读**：
- 柱子高低 → 模型优劣
- 某模型所有柱子都高 → 全面领先
- 某阈值差距大 → 该降水强度差异明显

**示例输出**：
```
model_comparison_TS_24h.png
```

---

## Python API 使用

### 基础用法

```python
import pandas as pd
from xmetai_evaluation.visualization import PrecipitationPlotter

# 读取数据
df = pd.read_csv("output/ts_fuxi.csv")

# 创建绘图对象
plotter = PrecipitationPlotter()

# 生成完整报告
plotter.create_report(
    df,
    output_dir="figures/",
    model_name="FuXi",
)
```

### 单独绘制图表

```python
# 1. TS 时效曲线
plotter.plot_metric_vs_lead(
    df,
    metric='TS',
    figsize=(12, 6),
    title='FuXi TS 评分',
    save_path='ts_curve.png',
)

# 2. 阈值对比（24h）
plotter.plot_metric_by_threshold(
    df,
    lead_h=24,
    metrics=['TS', 'POD', 'FAR'],
    figsize=(10, 6),
    save_path='threshold_24h.png',
)

# 3. TS 热力图
plotter.plot_heatmap(
    df,
    metric='TS',
    figsize=(12, 8),
    cmap='RdYlGn',  # 红-黄-绿
    save_path='heatmap.png',
)

# 4. 性能图
plotter.plot_performance_diagram(
    df,
    lead_h=24,
    figsize=(10, 10),
    save_path='performance.png',
)
```

### 多模型对比

```python
# 读取多个模型的结果
fuxi_df = pd.read_csv("output/ts_fuxi.csv")
fgvp_df = pd.read_csv("output/ts_fgvp.csv")
aifs_df = pd.read_csv("output/ts_aifs.csv")

# 绘制对比图
plotter.plot_multi_model_comparison(
    model_dfs={
        'FuXi': fuxi_df,
        'FGVP': fgvp_df,
        'AIFS': aifs_df,
    },
    metric='TS',
    lead_h=24,
    figsize=(12, 6),
    save_path='model_comparison.png',
)
```

### 自定义样式

```python
# 使用不同的 matplotlib 样式
plotter = PrecipitationPlotter(style='bmh')  # 或 'ggplot', 'seaborn-v0_8-darkgrid'

# 修改全局字体大小
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
```

## 配色方案

### 默认配色（色盲友好）

```python
THRESHOLD_COLORS = {
    '≥0.1':  蓝色   # 微量降水
    '≥10':   绿色   # 小到中雨
    '≥25':   橙色   # 中到大雨
    '≥50':   红色   # 大到暴雨
    '≥100':  紫色   # 暴雨到大暴雨
    '≥250':  棕色   # 特大暴雨
}
```

### 自定义配色

```python
from xmetai_evaluation.visualization.precipitation_plots import THRESHOLD_COLORS

# 修改配色
THRESHOLD_COLORS['≥0.1'] = '#1f77b4'  # 自定义颜色
```

## 高级用法

### 批量生成多个时效的性能图

```python
import matplotlib.pyplot as plt

for lead in [24, 48, 72, 96, 120]:
    plotter.plot_performance_diagram(
        df,
        lead_h=lead,
        save_path=f'figures/performance_{lead}h.png',
    )
    plt.close()  # 释放内存
```

### 生成 PDF 报告

```python
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('report.pdf') as pdf:
    # 页面 1: TS 曲线
    fig = plotter.plot_metric_vs_lead(df, metric='TS')
    pdf.savefig(fig)
    plt.close()

    # 页面 2: 性能图
    fig = plotter.plot_performance_diagram(df, lead_h=24)
    pdf.savefig(fig)
    plt.close()

    # ... 更多页面
```

### 交互式图表（Plotly）

```python
import plotly.express as px

# 将数据转换为 Plotly 格式
fig = px.line(
    df,
    x='lead_h',
    y='TS',
    color='grade',
    title='TS 随时效变化',
    labels={'lead_h': '预报时效 (h)', 'TS': 'TS 评分'},
)
fig.write_html('interactive_ts.html')
```

## 常见问题

### Q1: 图片分辨率不够清晰？

```python
import matplotlib.pyplot as plt

# 提高 DPI
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
```

### Q2: 中文显示乱码？

```python
# Windows
plt.rcParams['font.sans-serif'] = ['SimHei']

# macOS
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# Linux
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
```

### Q3: 图例显示不全？

```python
# 方法 1: 调整图例位置
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# 方法 2: 增加图片宽度
figsize=(14, 6)  # 加宽

# 方法 3: 使用紧凑布局
plt.tight_layout()
```

### Q4: 如何导出矢量图（用于论文）？

```python
# 保存为 PDF（矢量格式）
fig.savefig('figure.pdf', format='pdf')

# 保存为 SVG（矢量格式）
fig.savefig('figure.svg', format='svg')

# 保存为 EPS（LaTeX 常用）
fig.savefig('figure.eps', format='eps')
```

### Q5: 批量处理时内存不足？

```python
import matplotlib.pyplot as plt

# 每张图绘制完立即关闭
for lead in range(24, 241, 24):
    fig = plotter.plot_performance_diagram(df, lead_h=lead, save_path=f'perf_{lead}h.png')
    plt.close(fig)  # 释放内存
    plt.close('all')  # 关闭所有图
```

## 报告模板

### 完整评测报告结构

```
figures/
├── 1_overview/
│   ├── TS_heatmap.png              # 全局概览
│   └── summary_table.png           # 摘要表格
├── 2_skill_curves/
│   ├── TS_vs_lead.png              # TS 时效曲线
│   ├── POD_vs_lead.png             # POD 时效曲线
│   └── FAR_vs_lead.png             # FAR 时效曲线
├── 3_threshold_analysis/
│   ├── threshold_24h.png           # 24h 阈值对比
│   ├── threshold_48h.png           # 48h 阈值对比
│   └── threshold_72h.png           # 72h 阈值对比
├── 4_performance_diagrams/
│   ├── performance_24h.png         # 24h 性能图
│   ├── performance_48h.png         # 48h 性能图
│   └── performance_72h.png         # 72h 性能图
└── 5_model_comparison/
    ├── comparison_TS_24h.png       # 多模型 TS 对比
    └── comparison_POD_24h.png      # 多模型 POD 对比
```

### 自动生成目录结构

```python
from pathlib import Path

def create_report_structure(base_dir: Path):
    """创建报告目录结构"""
    sections = [
        '1_overview',
        '2_skill_curves',
        '3_threshold_analysis',
        '4_performance_diagrams',
        '5_model_comparison',
    ]
    for section in sections:
        (base_dir / section).mkdir(parents=True, exist_ok=True)
```

## 参考资源

- **Matplotlib 官方文档**: https://matplotlib.org/stable/
- **Seaborn 画廊**: https://seaborn.pydata.org/examples/index.html
- **气象可视化最佳实践**: WMO Guidelines on Performance Assessment
- **色盲友好配色**: ColorBrewer 2.0

---

**文档版本**: v1.0  
**最后更新**: 2026-09-10
