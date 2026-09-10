# 项目技能（Skills）

本目录包含项目中可复用的自动化工作流和评估工具。

---

## 可用 Skills

### 📊 XMetAI 模型评估

**Skill**: `xmetai-evaluation`

**描述**: 气象模型多维度评估工具集（降水、频谱、RMSE 等）

**功能**:
- ✅ 降水模型评估（自动诊断 + 智能建议）
- ⏳ 频谱分析（待实现）
- ⏳ RMSE 空间分布（待实现）

**使用方法**:
```bash
# 推荐：统一报告（一个完整的 MD）
/xmetai-evaluation compare \
  --new-model fgvp_tp \
  --new-results results/fgvp_tp \
  --baseline fgvp_v2 \
  --baseline-results results/fgvp_v2 \
  --reference FuXi,AIFS \
  --change "优化了 TP 参数化" \
  --output reports/fgvp_tp.md

# 传统方式：降水评估（多文件）
/xmetai-evaluation precipitation-report

# 其他功能（待实现）
/xmetai-evaluation spectrum-analysis
/xmetai-evaluation rmse-analysis
```

**文档**: [xmetai-evaluation/README.md](./xmetai-evaluation/README.md)

---

## 快速开始

### 1. 注册 Skill

**方式 1：符号链接（推荐）**
```bash
cd .claude/skills
ln -s ../../skills/xmetai-evaluation/SKILL.md xmetai-evaluation.md
```

**方式 2：直接复制**
```bash
cp skills/xmetai-evaluation/SKILL.md .claude/skills/xmetai-evaluation.md
```

### 2. 使用 Skill

```bash
# 在 Claude Code 中调用
/xmetai-evaluation precipitation-report
```

---

## 目录结构

```
skills/
├── README.md                    # 本文件（Skills 索引）
└── xmetai-evaluation/           # 模型评估 skill
    ├── SKILL.md                 # Skill 主定义
    ├── README.md                # 详细说明
    ├── scripts/                 # 评估脚本
    │   └── precipitation_report.py
    ├── references/              # 参考文档
    │   ├── precipitation-evaluation.md
    │   ├── diagnostic-rules.md
    │   └── evaluation-metrics.md
    ├── assets/
    │   └── templates/
    └── agents/                  # 子 agent（如需要）
```

---

## 添加新 Skill

### 步骤

1. **创建 Skill 目录**
   ```bash
   mkdir -p skills/my-new-skill/{scripts,references,assets/templates,agents}
   ```

2. **创建 SKILL.md**（参考 xmetai-evaluation/SKILL.md）
   ```markdown
   ---
   name: my-new-skill
   description: 一句话描述
   ---
   
   # My New Skill
   
   ## Start Here
   ...
   ```

3. **实现脚本**（放在 scripts/）

4. **添加文档**（放在 references/）

5. **更新本 README**（添加索引）

6. **注册到 .claude/skills/**

---

## Skill 设计原则

参考 `xmetai-weather-modeling` 的标准结构：

### 1. SKILL.md 保持简洁
- 元数据和描述
- Start Here 快速入口
- 执行步骤（调用 scripts）
- Reference Routing（指向详细文档）

### 2. 代码放在 scripts/
- 完整的可执行脚本
- 可以独立运行（不依赖 Claude）
- 良好的参数和帮助文档

### 3. 文档放在 references/
- 详细的使用说明
- 诊断规则
- 技术细节

### 4. 一个 Skill = 一个领域
- xmetai-evaluation: 所有评估相关功能
- xmetai-weather-modeling: 所有建模相关功能
- 不要为每个小功能创建单独的 skill

---

## 维护记录

- **2026-09-09**: 初始版本
  - 创建 xmetai-evaluation skill
  - 实现降水评估功能
  - 完善文档体系

---

**维护者**: 沈哲文 (szw)  
**最后更新**: 2026-09-09
