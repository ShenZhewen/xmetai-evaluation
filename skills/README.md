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
| `weather_field_scores` | 确定性连续场：RMSE / ACC / 预报活跃度 / 纬向谱 | ⚠️ 能力与渲染器在，格式契约骨架已删（单模型报告走 `generate_report.py`） |
| TS 系列（`weather_ts_det` / `weather_ts_ens` / `fdp_precip_ts` …） | 站点降水分类检验：TS / POD / FAR / 频率偏差 | ✅ 骨架分确定性 / 集合两份 |
| RMSE 批次家族（`weather_rmse_single_<模型>` / `weather_rmse_ens_<模型>`，外加谱检验补充报告） | 格点 RMSE / ACC / FA / 纬向谱，**多模型横向对比** | ✅ 三份骨架 + 三个渲染器都做了（`generate_det_report.py` / `_ens_` / `_wave_`） |
| 其余流程（CRPS、FSS、概率评分、台风…） | — | ⛔ 报告模板还没做，调用时会明确报错 |

**能力身份自动识别**：给一个评测产物目录，skill 从 `manifest.json` 的
`resolved_config.pipeline` 自己认这是哪条链路再派发，**用户不用记能力名，
也不要靠目录名判断**（配置名与流程名本来就不绑定）。

**使用方法**——说一句话就行，**命令由智能体执行，你不用自己跑 python**：

> "帮我对 `evaluation_results/weather_ts_single_fgvp` 这个评测结果生成报告"

智能体跑的就是这一条：

```bash
python skills/xmetai-evaluation/scripts/generate_report.py <产物目录> --result reports/<名字>
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
rm -rf .claude/skills/xmetai-evaluation
mkdir -p .claude/skills/xmetai-evaluation
cp -r skills/xmetai-evaluation/. .claude/skills/xmetai-evaluation/
diff -rq skills/xmetai-evaluation/ .claude/skills/xmetai-evaluation/   # 应当无输出
```

> **改完源文件要重新复制**。`skills/` 下的是源，`.claude/skills/` 下的是副本，
> 两边不会自动同步。
>
> **先 `rm -rf` 再复制，不要只 `cp -r`**：`cp -r` 只增不减，**源里删掉的文件
> 在副本里会一直留着**。这不是假设——2026-09-21 同步时实测副本的
> `assets/templates/` 里还躺着两份早已作废的 `README.md` 与 `ts.md`，
> 而源目录里那 7 份 `weather_*.md` 一份都没有。没装 `rsync` 就照上面这三条来，
> 最后那条 `diff -rq` 是验收动作。

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
    │   ├── generate_report.py   # 产物目录 → 图表 + REPORT.md（认能力 + 派发）
    │   ├── generate_det_report.py    # 批次归档（单成员）→ 多模型对比图 + REPORT.md
    │   ├── generate_ens_report.py    # 批次归档（集合）→ 同上，另加 CRPS/Spread
    │   └── generate_wave_report.py   # 批次归档（谱检验补充）→ 同上，另加球谐带
    ├── assets/
    │   └── templates/           # 报告格式契约（按能力家族一份，名字按 config 命名）
    │       ├── weather_ts_single.md    # 确定性降水检验：三份填空骨架（单模型 / 单基准 / 多模型）
    │       ├── weather_ts_ens.md       # 集合降水检验：同上三份，另加一节概率评分（AROC/BSS）
    │       ├── weather_rmse_single.md  # RMSE 批次家族（单成员）：一份骨架
    │       ├── weather_rmse_ens.md     # RMSE 批次家族（集合）：CRPS / Spread / Spread÷RMSE
    │       ├── weather_rmse_wave.md    # 纬向 FFT + 球谐带功率谱补充分析：一份骨架
    │       ├── weather_typhoon_single.md  # 台风路径/强度（确定性）
    │       └── weather_typhoon_ens.md     # 台风路径/强度（集合口径 + 离散度）
    ├── references/              # 参考文档（口径与判读规则）
    │   ├── evaluation-metrics.md        # 指标词典：是什么（不判好坏）
    │   ├── precipitation-evaluation.md  # TS 系列：口径与产出契约
    │   ├── diagnostic-rules.md          # TS 系列判读规则
    │   ├── field-evaluation.md          # 连续场判读规则
    │   ├── rmse-batch-evaluation.md     # RMSE 批次家族（single/ens/wave）
    │   └── model-diff-analysis.md       # 模型差异归因：改动起没起作用
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
按**能力家族**分文件、不按流程分——确定性两条 TS 流程共用一份 `weather_ts_single.md`。
**文件名跟 config 名走**（`weather_rmse_single_<模型>` 对 `weather_rmse_single.md`、
`weather_rmse_ens_<模型>` 对 `weather_rmse_ens.md`、
`weather_ts_det_*` / `fdp_precip_ts_*` 对 `weather_ts_single.md`、
`weather_ts_ens_*` 对 `weather_ts_ens.md`），
两字母缩写（`det` / `ts` / `field`）和家族简称（`precip_ts` / `rmse_single`）都不再用。
**例外是 `weather_rmse_wave.md`**：它不由 config 名而来——**没有
`weather_rmse_wave_*` 这个 config**，那份报告吃的是 `weather_rmse_single_<模型>` /
`weather_rmse_ens_<模型>` 的同一批产物，只是换个口径重讲一遍谱。
**`weather_field_scores` 这一族现在没有骨架了**（`field_scores.md` 已删）；
注意它与 RMSE 批次家族**是同一个 pipeline**——前者出单模型报告，后者出多模型对比。
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
    形如 `weather_ts_single_fgvp`），宽表统一落在 `diagnostics/categorical_wide.csv`；
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

- **2026-09-15**: 模板文件按 config 命名 + 新增集合预报骨架
  - `assets/templates/` 改名：`det.md`→`rmse_single.md`、`ts.md`→`precip_ts.md`、
    `field.md`→`field_scores.md`，新增 `rmse_ens.md`（集合口径：CRPS / Spread /
    Spread÷RMSE，对应 `weather_rmse_<模型>_ens`）。全仓库引用同步
    （本 README 目录结构、SKILL.md 路由表与参考文档、`ts_report.py` 的字节契约注释）
  - 顺手修掉 `rmse_single.md` 里两处与事实不符的旧说法：章节号写的是
    `一、二、…十一`（实际骨架是 `# 1.`…`# 11.`）、以及声称有
    `test_det.py` 守着（`tests/` 是空的，README 的「怎么被保证」才是实况）
  - `rmse_ens.md` **不含球谐带功率**：`vfc/metrics/spectrum.py` 只有纬向 FFT 谱，
    球谐停在「升级备忘」，收进来就是一条永远渲染不出来的章节；谱只保留纬向谱
  - 该骨架的渲染器（`visualization/ens_report.py` + `scripts/generate_ens_report.py`）
    **还没做**，模板索引里如实标为「还没做，调用时明确报错」

- **2026-09-15（下午）**: 模板全面改成 config 全名 + 新增谱检验骨架
  - `assets/templates/` 弃用家族简称：`precip_ts.md`→`weather_ts_single.md`、
    `rmse_single.md`→`weather_rmse_single.md`、`rmse_ens.md`→`weather_rmse_ens.md`；
    新建 `weather_ts_ens.md`（当时是空文件，当天下午写完，见下一条）
  - 新增 `weather_rmse_wave.md`——`weather_rmse_wave_<模型>` 批量归档的
    **纬向 FFT + 球谐带功率谱补充分析**报告骨架。章节 `# 1.`…`# 7.` + 附录、
    表不编号、图 `图 1：`…`图 7：` 连续；原件是语雀富文本导出，
    落地时去掉 `<font>` 彩色标签、语雀 CDN 图链换成 `![名字](名字.png)`，
    并按编号契约给每张图补了图注（原件一张图注都没有）。
    **渲染器还没做**
  - `field_scores.md` 与 `assets/templates/README.md` 同日删除；模板索引的职责
    收归 `xmetai-evaluation/SKILL.md` 的「参考文档」一节
  - 本文件的目录结构与「模板文件按 config 命名」一节、`SKILL.md` 的路由表与
    参考文档、三份 RMSE 骨架头部的互相引用，全部按新名字同步

- **2026-09-15（傍晚）**: `weather_ts_ens.md` 写完，TS 骨架正式分成两份
  - 集合那份＝确定性那份的二～五节**逐字相同**，加两处：头部多一行「集合成员数」
    与**集合口径声明**（TS/POD/FAR/漏报率/BIAS 都算在成员平均场上，不是逐成员
    评分再平均）；并**新增第六节「概率评分（AROC / BSS）」**，后面章节号整体后移
    一位（A 到八、B/C 到九）
  - AROC/BSS 一节的口径按 `core/categorical_ref.py` 的实际实现写，不是照抄指标名：
    `BS_ref` **只有给了外部气候概率参考的 6h 窗才有值**、BSS=1−BS/BS_ref、
    AROC 是**逐成员插值后**按超越式阈值算、每个时效末尾还有一行 `grade="AVG"`
    的**可算阈值等权平均**（不是按样本数加权）
  - **这一节目前没有代码支撑，骨架头部如实标了两处**：`run_ts_ens` 把 `window_aroc`
    置空（该能力不产 AROC/BSS 表）、`weather_ts_ens_prob` 不在 `report.py` 的
    `IMPLEMENTED` 里且 `README.md` 标着 `[BLOCKED]`。数据表本身是有的
    （`aroc_bss_<name>.csv`，11 列）
  - `weather_ts_single.md` 头部从"覆盖三条流程、三条共用这一份"改成
    **只管确定性两条**，并指向集合那份；`SKILL.md` 路由表拆成两行、
    "三条 TS 流程共用一套模板"那句改写成"同一个渲染器、两份骨架"

- **2026-09-15（晚）**: 清掉删除文件留下的失效引用
  - `field_scores.md` 被删之后 `SKILL.md` 还有三处指着它：路由表那格改成
    「能力与渲染器都在，但格式契约已删」；连续场那节「'结构固定'是被测试守着的」
    改成**明说真值来源没了**（别再假装测试还钉得住）；参考文档节删掉那一行
  - 顺带查出 `assets/templates/README.md` **也是同日被删的**、参考文档节还列着它，
    一并删掉，换成一句「`assets/templates/` 现在只剩五份，连续场的骨架与
    模板目录的 README 都已删」
  - 两份 RMSE 骨架头部「与 `weather_ts_single.md` / `field_scores.md` 的差别」
    去掉后半段；本文件设计原则 3 删掉 `weather_field_scores` 对
    `weather_field_scores.md` 这个已失效的举例，`weather_ts_*` 那条例举按
    single / ens 拆开
  - `weather_rmse_ens.md` 里「本骨架不含球谐带功率」那段改成**指向
    `weather_rmse_wave.md`**：球谐带那两节归谱检验骨架
    （对应 5.1 分频 / 5.2 分变量 / 5.3 配对检验），不在这份重复
  - 历史条目里出现的 `ts.md` / `det.md` / `field_scores.md` **不动**——
    那是当时的实况记录，不是失效链接

- **2026-09-15（深夜）**: 规整 `references/`——补上 RMSE 批次家族那篇
  - 最大的洞是**三份 RMSE 骨架在 `references/` 里一个字都没有**：新增
    `rmse-batch-evaluation.md`，把 `visualization/det_report.py` 的归档形状、
    派生指标（相对 RMSE 的 `100` 是**几何均值基线**、FA 偏差取了绝对值方向就丢了、
    频谱对数 RMS 的底数）、时效分段 `(0,120]/ (120,240]/ (240,360]`、
    Wilcoxon + Holm，以及**跨批次比较必须一致的六个口径参数**写进去；
    并如实标明 `_ens` / `_wave` 的渲染器都还没做、球谐带没有数据来源
  - `evaluation-metrics.md` 从 788 行的杂糅改成**纯指标词典**：删掉「空间聚合与 FSS」
    「业务指标组合」「Case 1-4」「时效典型模式」这些教科书内容和与
    `diagnostic-rules.md` 重复的 BIAS 分级（**冲突一律以 `diagnostic-rules.md` 为准**），
    补上 CRPS / Spread / Spread÷RMSE / AROC / BS / BSS / base_rate / `AVG` 汇总行的定义
  - `diagnostic-rules.md` 新增 §6 概率评分（AROC / BSS），并**明说这节没有代码支撑**
    （`run_ts_ens` 把 `window_aroc` 置空、`weather_ts_ens_prob` 未实现）；
    后面两节顺延为 §7 / §8
  - `precipitation-evaluation.md` 补概率评分表契约（`aroc_bss_<name>.csv`）、
    「概率评分与 TS 口径完全不同」对照表、A/B/C 三支线的**图与表编号表**，
    并把 `weather_ts_ens` 多一节导致整体后移一位这件事写清
  - `field-evaluation.md` 加一句`field_scores.md` 已删、格式契约没有真值来源
    （口径与判读规则不受影响——它们对的是代码，不是骨架）
  - 顺带记下一处**代码与骨架对不上**：纬向谱 log-RMS 实现用 `log10`
    （`det_report.py:551`），`weather_rmse_wave.md` 的骨架文字写的是 `ln`，
    差 ×2.3026，**尚未与实现核对**，两处都标了「以代码为准」

- **2026-09-15（凌晨）**: SKILL.md 按标准 skill 模板重排 + 新增归因框架
  - **起因是核对真实结果目录时发现文档与实况对不上**：`outputs/results/` 下是
    **扁平布局**（`ts_*.csv` + `*_meta.json`），而 SKILL.md 一直描述的是
    `manifest.json` + `diagnostics/categorical_wide.csv`。`categorical_wide.csv`
    **在本仓库里没有任何流程会写**（grep 只有读者没有写者）——今天那份
    `report/weather_ts_single_fgvp_fuxi_aifs/` 是手工 staging 才跑通的
  - SKILL.md 重排成标准形态并**只增不减**：`干什么/不干什么` → `触发场景` →
    **`输入：结果目录`**（扁平布局 + `_meta.json` 逐键说明 + TS 主表契约）→
    **`归类`**（四级判据 + 归类对照表 + 报清单）→ **`工作流`六步** →
    `能力路由` → `输出契约` → `硬性规则`（新增 13/14/15 三条）→
    `常见情形` → `参考文档` → `维护说明`
  - 工作流从原来的"第 1 步跑命令"改成**归类 → 定主模型 → 核对口径 → 跑 →
    转述 → 失败直说**六步：主模型可由用户点名（决定"本模型"的全部说法
    与差值图纵轴方向），未点名则按「目录名标记 → `<capability>.json` → 反问」推断
  - 新增 `references/model-diff-analysis.md`——**"改的这个模块起没起作用"**
    的判断框架：先把改动翻译成预期特征（六类改动的指纹表）→ 在结果里找特征 →
    做**波及面检查**（未改动变量/量级/时效有没有一起动）→ 四条判定规则
    （方向对 / 不是噪声 / **有代价评估** / 跨时效一致）→ 症状词典
    （八种数字形态反推成因，含"单位不匹配长得像技巧超好"那个真坑）
  - 如实标注四处**已知缺口**（SKILL.md 维护说明节）：主模型入口认不得扁平布局、
    `_ens`/`_wave` 渲染器没做、AROC/BSS 取数绘图没做、球谐带不存在
  - 本 README 的能力状态表同步：`weather_field_scores` 从 ✅ 改 ⚠️（骨架已删）、
    补 RMSE 批次家族一行

- **2026-09-16**: 补上 `weather_rmse_ens` 的渲染器，RMSE 三条支线都能跑了
  - 新增 `visualization/ens_report.py` + `scripts/generate_ens_report.py`，
    照 `weather_rmse_wave` 的五段式结构写；`analyze_ens()` 第一句就是
    `det_report.analyze()`，所以归档形状、相对 RMSE、FA 偏差、配对检验全部复用，
    集合特有的只有 CRPS / Spread / Spread÷RMSE / ACC 四块
  - `det_report._pick_metric_file` 的候选名**加一个** `f"{stem}_{date}_ensmean.csv"`
    （排在最末，单成员行为不变），这一行让 `analyze()` 原样吃下集合归档，
    且拿到的正是骨架要求的**集合平均场**口径
  - **两条口径红线写进骨架引用注释**：集合对数用 **ln**、单成员用 **log10**（差 2.3026×）；
    逐日表只读归档自带的 `<口径>_<日期>.csv`，**不用 glob 兜底**——
    否则 `rmse_<日期>_member_000.csv`（单成员）会被当集合平均静默读进来
  - 同步纠正三处**早已过时**的说法：`_wave` 的渲染器其实早就做好了
    （`wave_report.py` + `generate_wave_report.py`），球谐带功率也早就有了
    （`vfc/metrics/spectrum.py:210` 一直在写 `spherical_bands_*.csv`，实测 1308 份）。
    `references/rmse-batch-evaluation.md` §0 的三行表格、SKILL.md 的已知缺口、
    本文件的目录树一并改过来
  - 实测数据现状：三份集合归档（FuXi 349 天 / AIFS 27 天 / FGVP 2 天）
    **三方共同日期为 0**，所以本轮只用 FuXi↔AIFS 的 27 天验证渲染器，
    正式三方报告等 FGVP 数据补齐

- **2026-09-21**: 三份 RMSE 骨架改版 + 文档按 `xmetai_evaluation/` 实况同步
  - **骨架改版**（`weather_rmse_single.md` / `_ens.md` / `_wave.md`）：
    编号契约第一次写明确——正文与必出附录 `图 N：`（`_single` 必出 1–4、
    `_ens` 必出 1–6、`_wave` 必出 1–7，其余按数据可得性出、**出不了就保留节位
    标「本批未出」，允许出现断号**），可选块另开 `图 L*` / `图 S*` 两个命名空间，
    **整块删掉不用重排任何编号**
  - 骨架新增三节：**变量 × 时效 RMSE 热力图**（各自除以本模型首时效的 RMSE
    倍数，单位无关才能同色标横比）、**谱比随时效**、**单起报谱曲线**；
    以及「附 L 分纬度带结果」「附 S 分季节结果」两个可选块
  - 渲染器接纬度带：`det_report.region_table()` 成为 det/ens/wave 共用的汇总，
    正文只留全球行、附 L 只留带行（**两类行混着 `pivot_table(mean)` 会静默
    抹平带间差异**）；并列第一按显示精度判、逐个列出，不再拿 `idxmin` 挑一个
    冒充当赢家
  - 新增图品类 `multi_model_rmse_heatmap` / `ensemble_rmse_heatmap` /
    `spectrum_curve_<变量>_<起报>`；**「谱比随时效」渲染器和骨架都留着但必然
    是本批未出**——产物里没有任何逐 lead × 逐波数的谱（长表那条是标量、
    `spectrum_by_init` 没有 lead 列）
  - **文档按仓库实况重写**：`references/rmse-batch-evaluation.md` 的 §1 归档形状
    与 §4 口径参数整节重写，SKILL.md 的归类判据 / RMSE 家族说明 / 已知缺口同步，
    三个入口脚本的 docstring 同步。作废的东西统一登记在文首那条 ⚠️ 里
  - **三处作废说法**（旧文档写于另一个仓库布局，本仓库里没有对应物）：
    `vfc/regr_ens.py` / `vfc/regr_summary.py` / `vfc/regr_pair.py` 全不存在
    （只有 `xmetai_evaluation/`）；`--summarize-det` / `--summarize-ens`
    两个开关不存在；「批次归档 = `summary.csv` + `batch_meta.json` +
    `<YYYYMMDD>/`、没有 `manifest.json`」不成立——产物就是标准评估产物目录
    （`scores.csv` + `manifest.json` + `diagnostics/`），`pipeline` 也正是
    `weather_field_scores`。RMSE 家族单开入口的原因是**多模型对比**，不是形状不同
  - **实拍发现、写进文档让下次不用重新发现**：
    (a) 长表契约 **24 列**（含 `group`），而现存归档的 `scores.csv` 只有 **23 列**、
    没有 `group` —— 球谐带 5 个频带的行因此认不出来，`wave_report` §5.1–5.3
    会标「本批未出」，**重跑评测即可**；
    (b) `manifest.json` 的 `artifacts` 存的是**写盘时的绝对路径**，
    产物目录拷走后即失效，别拿它定位文件；
    (c) `--model NAME=目录` 的 `NAME` 与目录名、与 `manifest.run_id`
    **三者可以全都不一样**（实测把 FuXi 的产物拷进 `..._fgvp` 目录，
    `run_id` 照样是 `..._fuxi`）

---

**维护者**: 沈哲文 (szw)
**最后更新**: 2026-09-21
