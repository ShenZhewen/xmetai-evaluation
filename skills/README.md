# 项目技能（Skills）

本目录包含项目中可复用的自动化工作流和评估工具。

---

## 可用 Skills

### 📊 XMetAI 模型评估

**Skill**: `xmetai-evaluation`

**描述**: 气象模型评测结果的分析与出图报告

**覆盖的评测能力**:

| 能力（流程名） | 覆盖 | 状态 |
|---|---|---|
| `weather_field_scores` | 确定性连续场：RMSE / ACC / 预报活跃度 / 纬向谱 | ✅ |
| TS 系列（`weather_ts_det` / `weather_ts_ens` / `fdp_precip_ts` …） | 站点降水分类检验：TS / POD / FAR / 频率偏差 | ✅ |
| 其余 8 条流程（CRPS、FSS、概率评分、Bias…） | — | ⛔ 报告模板还没做，调用时会明确报错 |

**能力身份自动识别**：给一个评测产物目录，skill 从 `manifest.json` 的
`resolved_config.pipeline` 自己认这是哪条链路再派发，**用户不用记能力名，
也不要靠目录名判断**（配置名与流程名本来就不绑定）。

**使用方法**——说一句话就行，**命令由智能体执行，你不用自己跑 python**：

> "帮我对 `evaluation_results/weather_ts_det_fgvp` 这个评测结果生成报告"

智能体跑的就是这一条：

```bash
python skills/xmetai-evaluation/scripts/generate_report.py <产物目录> --out reports/<名字>
```

它从 `manifest.json` 自己认出能力（TS 系列 / 连续场），派发到对应模板，出图并写 `REPORT.md`。
认不出、或认出来但模板还没做时**明确报错**，不会静默套用另一个模板。
更多选项见 `xmetai-evaluation/SKILL.md`。

**文档**: [`xmetai-evaluation/SKILL.md`](./xmetai-evaluation/SKILL.md)

---

## 快速开始

### 1. 注册 Skill

Claude Code 认的是**目录形式**：`.claude/skills/<name>/SKILL.md`。
本仓库用**复制**（不是符号链接——Windows 建符号链接要管理员或开发者模式）：

```bash
mkdir -p .claude/skills/xmetai-evaluation
cp -r skills/xmetai-evaluation/. .claude/skills/xmetai-evaluation/
```

> **改完源文件要重新复制**。`skills/` 下的是源，`.claude/skills/` 下的是副本，
> 两边不会自动同步。

### 2. 使用 Skill

在 Claude Code 里直接说需求即可（例如"这是连续场评测的结果目录，出个报告"），
skill 会按 `description` 里的触发词被自动选中；也可以用 `/xmetai-evaluation` 显式调用。

---

## 目录结构

```
skills/
├── README.md                    # 本文件（Skills 索引）
└── xmetai-evaluation/           # 模型评估 skill
    ├── SKILL.md                 # Skill 主定义：什么时候用、怎么用、硬性规则
    ├── scripts/
    │   └── generate_report.py   # skill 的唯一入口：产物目录 → 图表 + REPORT.md
    ├── assets/
    │   └── templates/           # 报告格式契约（按能力家族一份）
    │       ├── README.md        # 模板目录说明 + 索引
    │       └── ts.md            # TS 系列：三份填空骨架（单模型 / 单基准 / 多模型）
    ├── references/              # 参考文档（口径与判读规则）
    │   ├── evaluation-metrics.md
    │   ├── precipitation-evaluation.md
    │   ├── diagnostic-rules.md  # TS 系列判读规则
    │   └── field-evaluation.md  # 连续场判读规则
    └── agents/                  # 空。暂无子 agent
```

---

## 添加新 Skill

1. **创建 Skill 目录**：`mkdir -p skills/my-new-skill/references`
2. **创建 `SKILL.md`**（带 `name` / `description` frontmatter；`description` 写清**触发场景**，
   它是被自动选中的唯一依据）
3. **放参考文档**（`references/*.md`）
4. **更新本 README**（添加索引）
5. **注册到 `.claude/skills/`**（复制，见上）

---

## Skill 设计原则

参考 `xmetai-weather-modeling` 的标准结构：

### 1. SKILL.md 保持简洁
- 元数据和描述（`description` 决定什么时候被触发，要写触发词而不是功能清单）
- 快速入口
- 执行步骤
- Reference Routing（指向 `references/` 里的详细文档）

### 2. `scripts/` 只放入口，渲染逻辑放主仓库
`scripts/generate_report.py` 是 skill 的入口——解析参数、插 `sys.path`、
调 `xmetai_evaluation.visualization.report.build_report`、打印产物清单，**仅此而已**。

- **认能力、选模板、出图、写报告的逻辑全在主仓库**（`xmetai_evaluation/visualization/`）：
  那里能被 `pytest` 覆盖、能 import、能复用。入口脚本一旦开始有判断逻辑，
  就又变成一份会跟主仓库漂移的副本；
- 入口脚本存在的理由是**免安装**：`python -m xmetai_evaluation.visualization.report`
  要求包已经 `pip install -e .`，而它把仓库根插进 `sys.path`，直接就能跑；
- `references/*.md` 是代码的**散文镜像**，每篇都声明"以代码为准"。

### 3. `assets/templates/` 是报告格式契约
放的是**一份完整的报告骨架**——章节、表头、分隔行、口径说明全是真的，
只有数值位置用 `{…}` 空着。**不是说明文档，也不参与渲染**：
报告由主仓库的渲染器逐行拼出，骨架是给人看、给测试对的真值来源。
按**能力家族**分文件、不按流程分——三条 TS 流程共用一份 `ts.md`。
`tests/unit/visualization/test_ts_report.py` 会实跑一遍逐项比对：
改了渲染器的章节结构就必须同步改骨架，否则测试红。
逐字扫描只能钉住**不含占位符**的行，所以另有一组「编号与排版」用例
补上含 `{…}` 的那部分（图注/表题编号连续、每节小结带数字、每张图都被安排章节）。

### 4. 文档放在 references/
详细的使用说明、口径、诊断规则、与代码函数的对应关系。

### 5. 一个 Skill = 一个领域，能力靠路由区分
- `xmetai-evaluation`：**所有**评测相关能力，不要为每条流程建一个 skill；
- 新增能力时扩 `description` 的触发词 + 加一篇 `references/*.md`，
  再加一条路由判据（见 `xmetai_evaluation/visualization/report.py` 的
  `IMPLEMENTED` / `_FALLBACK_RULES`）。

---

## 维护记录

- **2026-09-09**: 初始版本
  - 创建 xmetai-evaluation skill
  - 实现降水评估功能
  - 完善文档体系
- **2026-09-11**: 加入连续场能力与自动路由
  - 新增 `references/field-evaluation.md`（`weather_field_scores` 判读规则）
  - SKILL.md 增加"能力路由"一节与两条硬性规则（`n_valid` 语义、`unit` 随数据源走）
  - 本 README 修正：删掉失效的 `xmetai-evaluation/README.md` 链接与旧的斜杠命令写法，
    目录结构改成实况，设计原则 2 改为"代码在主仓库"的实况描述
- **2026-09-11（下午）**: 接通 TS 路由 + 标准化 skill 库
  - `visualization/report.py` 按**报告家族**分派（`field` / `ts`）；
    `weather_ts_det` / `weather_ts_ens` / `fdp_precip_ts` 读
    `diagnostics/categorical_wide.csv`；报告标题默认取 `run_id`
    ——原先取 `forecast_source`，那是 reader 类型（`"fuxi"`），会张冠李戴
  - 新增 `scripts/generate_report.py`（skill 唯一入口，免 `pip install`）
    与 `assets/templates/`（`README.md` + `ts.md` 格式契约）
  - 新增 `tests/unit/visualization/test_ts_report.py`，其中一组用例实跑比对章节序列
  - 本 README：设计原则 2 从"`scripts/` 留空"改成"只放入口"，目录结构同步
- **2026-09-11（晚）**: TS 报告加「跨时效分布（箱线图）」+ 接通多模型对比
  - 渲染器新增第四节（箱线图 + 分位数表，同一个 `box_stats()` 出数）；
    `plot_metric_boxplot` 按 Tukey 几何画，口径由 `matplotlib.cbook.boxplot_stats` 当神谕钉住
  - 入口脚本新增 `--compare "名字=CSV路径"`（可重复）：对比模型是**裸结果表**，
    不是产物目录——`ts_multi_{fgvp,fuxi,aifs}` 那几个目录没有 `manifest.json`
  - `assets/templates/ts.md` 新增真正的骨架 C（多模型）；骨架 A/B 补第四节并重排章节号，
    至此三个分支各有端到端用例
  - 本 README 与 SKILL.md / references 同步：删掉"骨架 C 没有测试覆盖""多模型要手工调
    `create_report`""只有长表就合成宽表"三处已失效的说法
- **2026-09-11（深夜）**: 「对比其他模型」变成一句话
  - `evaluation_results/` 规整成**一个模型一个目录**（目录名 = config `name` = `run_id`，
    形如 `weather_ts_det_fgvp`），宽表统一落在 `diagnostics/categorical_wide.csv`；
    老式的 `ts_multi_*` 目录已迁移，被取代的挪进 `_archive/`（归档不删）
  - SKILL.md 工作流新增"第 4 步 · 用户要对比其他模型时"：**同级目录里有宽表的都是候选**，
    自己找齐、核对口径、**把清单报给用户确认**再跑——不加 `--compare-all` 之类的新开关，
    `--compare` 已经能表达一切，缺的只是"去哪儿找"

- **2026-09-11（更深）**: TS 报告改写成"论文实验章节"
  - 报告从"表格 + 末尾一堆裸 PNG"改成**每节自成一段实验叙述**：
    引导句 → 图表 → `图 N：图注`（图下方）/ `表 N：表题`（表上方）→ `**小结**：…`；
    **末尾的「图表」索引节整个取消**（图都就地带着图注出现，再列一遍是重复）
  - 新增第四节「误差形态与偏差结构」，POD / FAR / BIAS / 性能图四张图终于有了自己的章节
    （原先只在摘要里提一句）；章节链变成 二/三/四/五 +（可选）六，
    二～五在单模型/单基准/多模型三个分支里**逐字相同**
  - `ts_report.py` 新增五个 `analysis_*` 函数（分级/时效/误差形态/稳定性/对比），
    **只吃已算好的统计量，不重新读表、不重新实现指标**；无对比对象时不出对比段。
    顺带修掉两个自相矛盾：并列（差距 < 0.005）不再报领先者、
    名次改用**竞赛排名**（原先 `sorted().index()` 让并列的两家按字典顺序排，
    表里写 `3/3` 而正文算出第 2）
  - 契约补强：骨架的逐字扫描**抓不到含 `{…}` 的行**，而图注/表题/小结全在那种行里，
    所以新增一组「编号与排版」用例——`图 N：`/`表 N：` 连续不跳号、
    每节小结得有数字（防套话）、`create_report` 的每张图都必须被安排章节
  - 新增 `references/precipitation-evaluation.md` **§8 待办：空间分布图**——
    用户提的"卫星图"本轮不做，并查清了**不是缺绘图代码、是逐点场在
    `metrics/categorical.py:103-118` 就被展平求和丢掉了**，长表也没有 lat/lon 列；
    要画地图得先加"保留逐点场的 metric + 写场的 writer"再重跑，写进文档留作待办

---

**维护者**: 沈哲文 (szw)
**最后更新**: 2026-09-11
