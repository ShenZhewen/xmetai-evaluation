# xmetai-evaluation

气象模型评估框架 - 模块化、可复用、配置驱动

> 文档版本：v2 实现版 · 更新日期：2026-09-09
>
> **当前状态**：框架核心已实现，支持配置驱动的评估任务。详细架构设计见文档末尾。

## 1. 项目目标与边界

**目标：将 ref 中的天气、气候、确定性、集合和专项评测能力整合成一套框架，统一数据接入、处理、指标计算、执行和报告，并为后续增加指标与数据集提供稳定扩展点。**

项目负责人：沈哲文。团队成员：刘子航、安达维、周辰光、王晨鹏。

### 1.1 做什么，不做什么

- 输入：已经完成推理的模型产品，以及观测、再分析、气候态和专项参考资料。
- 接入范围：预报主要为 NetCDF；参考数据涉及 NetCDF、GRIB/GRIB2 和 Diamond3 站点文件。
- 输出：结构化指标、诊断场/曲线、数据覆盖统计、图表、单模型和多模型报告。
- 支持确定性场、集合成员场，以及后续直接输入的概率产品。
- **不负责模型推理、模型权重加载、训练或推理调度。**
- “通用”指气象评测领域内不绑定模型名称、文件布局、数据集或业务脚本，不追求跨所有 AI 领域的万能评测平台。

与推理框架的关系：`xmetai-inference → 预报产品文件 → xmetai-evaluation → 评测结果与报告`。

### 1.2 必须整合的业务能力

| 业务族 | 现有能力范围 | 主要验证资料 |
|---|---|---|
| FDP | 短期降水、多个大气变量、确定性/集合评分、活动度、功率谱 | 站点、CMPAS、CRA及气候态 |
| Tianqi | 天气尺度分类检验、站点插值、概率评分 | 站点观测、参考事件概率 |
| Qihou | 气候态、距平、长时效序列、周/组合周聚合、相关和误差场 | CRA与模型回报资料 |
| MJO | 多变量预处理、RMM指数、相关、误差和技巧时效 | OLR/U200/U850、历史资料、投影基底 |

原业务范围约为 FDP 0–10 天、Tianqi 0–15 天、Qihou 15–60 天。这些是**任务模板的范围，不是执行内核的硬编码边界**；时间窗由任务配置定义。

### 1.3 整合原则

1. **整合能力，不整合脚本外壳**：迁移算法和适配规则，不保留硬编码路径、全局配置和重复主循环。
2. **契约先于目录**：先定义可比较的数据、合法指标输入和结果语义，再实现插件。
3. **一个执行内核**：简单任务和长流水线共用规划、缓存、失败处理和结果存储。
4. **配置组合已有能力，代码扩展新能力**：已有格式的新数据源可只写 YAML；全新格式或算法仍需插件。
5. **数值语义显式化**：集合处理、单位、累积窗口、阈值、权重、基准和聚合口径不能靠猜测。
6. **先单机、后扩展**：首先实现内存受控的分块执行，后续按实际瓶颈扩展并发/执行后端。

## 2. 总体结构与职责分层

### 2.1 两种使用方式，一个运行内核

- **常规任务**：用户配置数据源、处理规则、指标和输出，由任务规划器生成标准流程。
- **专项流水线**：用户显式声明气候态、距平、历史窗、投影等步骤及依赖。
- 两者最终都生成同一种 `ExecutionPlan`，交给同一个 `PipelineExecutor` 执行。
- 不为 FDP、Tianqi、Qihou 分别建立读取器、指标库、缓存系统和执行循环。

### 2.2 分层职责

| 层 | 模块 | 负责 | 禁止承担 |
|---|---|---|---|
| 入口与配置 | `cli.py`、`core/config.py` | 配置解析、校验、组件解析、启动任务 | 指标公式、模型专用读取 |
| 数据契约 | `core/contracts.py` | 命名维度、元数据、样本和结果约束 | 读文件、任务调度 |
| 数据接入 | `io/` | 文件发现、解码、目录/变量/维度映射 | 业务累积、重网格、计算评分 |
| 数据处理 | `preprocess/` | 单位规范化、时间窗、空间变换、派生产品 | 隐式扫描数据目录、写报告 |
| 样本匹配 | `evaluators/matching.py` | 验证时间、站点/网格配对、共同样本与掩码 | 默认丢弃不匹配数据而不记录 |
| 指标算法 | `metrics/` | 统计量、合并与最终评分、诊断产品 | 读文件、猜轴、偷偷取集合平均 |
| 任务编排 | `evaluators/`、`pipeline/` | 生成依赖计划、执行、缓存、恢复 | 重复实现业务算法 |
| 结果与展示 | `results/`、`visualization/` | 结果落盘、对比校验、图表和报告 | 从原始预报重新计算指标 |

**依赖方向**：算法依赖数据契约；编排层依赖组件接口并组合算法；绘图依赖结果契约。禁止指标反向调用 Evaluator，禁止核心运行时导入 `ref/`。

### 2.3 标准数据流

1. 配置解析，确定数据源、验证协议和请求指标。
2. Catalog 枚举可用文件与逻辑样本，生成输入清单。
3. Planner 根据指标输入要求推导读取范围和处理依赖。
4. Reader 读取数据，映射标准变量/维度，保留原始语义。
5. Preprocess 显式执行单位转换、累计/差分、空间转换及派生。
6. Matcher 生成预报与观测配对数据、有效掩码和覆盖统计。
7. Metric 分块累积、合并统计量并生成结果。
8. ResultStore 保存结果与运行记录；展示模块消费这些结果。

数据契约在各步骤边界检查。**相同 shape 不代表数据可比较，坐标对齐也不代表物理量与时间区间一致。**

## 3. 预期项目结构

以下是目标目录，不要求第一阶段一次性创建所有空文件。随着各阶段验收逐项落地；所有 Python 包按需补充 `__init__.py`。

```text
xmetai-evaluation/
├── README.md                         # 本架构报告
├── pyproject.toml                    # 打包、CLI入口、核心/可选依赖声明
├── configs/
│   ├── defaults.yaml                 # 通用执行默认值，不放隐式科学口径
│   ├── data_sources/
│   │   ├── forecasts/                # fuxi_ens、fuxi_det、pangu、fengqing、aifs等
│   │   ├── observations/             # cra、cmpas、stations等
│   │   └── references/               # 气候态、事件概率、MJO投影基底
│   ├── protocols/                    # 固定验证口径：区域/时间窗/权重/基准
│   └── tasks/
│       ├── fdp/                      # 降水、大气变量、集合、活动度/谱
│       ├── tianqi/                   # 站点分类检验
│       └── qihou/                    # 长时效确定性、MJO
├── xmetai_evaluation/
│   ├── cli.py                        # validate/plan/run/report统一入口
│   ├── core/
│   │   ├── config.py                 # 分层配置解析和强校验
│   │   ├── contracts.py              # DataRequest/DataBundle/EvaluationBatch
│   │   ├── variables.py              # 标准物理量、单位与时间语义
│   │   ├── registry.py               # reader/transform/metric/step注册
│   │   ├── errors.py                 # 配置/数据/计算错误分类
│   │   └── logger.py
│   ├── io/
│   │   ├── base.py                   # Catalog/Reader协议
│   │   ├── forecast/
│   │   │   ├── catalog.py            # init/lead/member文件索引
│   │   │   ├── netcdf_reader.py      # 分成员文件/内置成员维/多时效
│   │   │   └── layouts.py            # 文件序号、时效与路径模板映射
│   │   ├── observation/
│   │   │   ├── cra_reader.py
│   │   │   ├── cmpas_reader.py
│   │   │   └── station_reader.py     # Diamond3解码、原始质控字段
│   │   └── reference/
│   │       ├── climatology_reader.py
│   │       └── mjo_basis_reader.py
│   ├── preprocess/
│   │   ├── normalize.py              # 单位、坐标规范化及显式缩放
│   │   ├── accumulation.py           # 时段累计、起报累计差分
│   │   ├── temporal.py               # 日/周/组合窗聚合与历史窗
│   │   ├── interpolation.py          # 格点到站点
│   │   ├── regrid.py                 # 格点到格点
│   │   ├── ensemble.py               # 集合均值/概率产品派生
│   │   ├── climatology.py            # 气候态、事件概率参考构建
│   │   ├── anomaly.py
│   │   └── mjo.py                    # 多变量规范化、历史均值去除、RMM投影
│   ├── metrics/
│   │   ├── base.py                   # MetricRequirements/Metric协议
│   │   ├── states.py                 # 可合并统计状态与维度校验
│   │   ├── deterministic/
│   │   │   ├── error.py              # RMSE/MAE/有符号误差
│   │   │   ├── acc.py
│   │   │   └── tcc.py
│   │   ├── categorical/
│   │   │   ├── contingency.py        # TS/POD/FAR/漏报率/频率Bias
│   │   │   └── fss.py
│   │   ├── probabilistic/
│   │   │   ├── crps.py
│   │   │   ├── brier.py              # BS/BSS
│   │   │   └── roc.py                # ROC曲线/AROC
│   │   ├── ensemble/
│   │   │   ├── spread.py
│   │   │   └── ssr.py
│   │   └── specialized/
│   │       ├── activity.py
│   │       ├── spectrum.py
│   │       └── mjo.py                # 指数评分和技巧时效
│   ├── evaluators/
│   │   ├── evaluator.py              # 用户门面：构建并执行任务
│   │   ├── planner.py                # 指标需求→读取/处理/计算依赖
│   │   ├── matching.py               # 时间/空间匹配和共同样本
│   │   └── protocols.py              # 可比性规则与验证口径
│   ├── pipeline/
│   │   ├── base.py                   # Step/ArtifactRef/ExecutionPlan
│   │   ├── executor.py               # 唯一执行内核
│   │   ├── cache.py                  # 指纹、产物有效性检查
│   │   ├── context.py                # 日志、产物存储、运行上下文
│   │   └── steps/
│   │       ├── read.py               # Reader的薄封装
│   │       ├── transform.py          # Preprocess的薄封装
│   │       ├── evaluate.py           # 匹配、累计、合并、finalize
│   │       └── export.py             # 结果/报告导出
│   ├── results/
│   │   ├── schema.py                 # MetricResult/ResultBundle
│   │   ├── store.py                  # 表格、诊断、统计状态保存
│   │   ├── provenance.py             # 输入清单/版本/配置来源记录
│   │   └── comparison.py             # 对比口径核查、外部结果接入
│   ├── visualization/
│   │   ├── timeseries.py
│   │   ├── spatial.py
│   │   ├── categorical.py
│   │   ├── diagnostics.py            # ROC/谱/MJO等结果展示
│   │   └── report_generator.py
│   └── utils/
│       ├── paths.py                  # 路径模板、环境变量解析
│       └── atomic.py                 # 同文件系统原子落盘
├── scripts/                          # 可选便捷入口，只转发CLI/API
├── tests/
│   ├── fixtures/                    # 小型合成数据，不复制生产数据
│   ├── unit/
│   ├── contracts/
│   ├── integration/
│   └── regression/                  # ref同口径对照与差异说明
├── docs/
│   ├── data_contract.md
│   ├── metric_protocols.md
│   ├── configuration.md
│   ├── migration_inventory.md       # 逐文件/能力迁移台账
│   └── extensions.md
└── ref/                             # 保留原始参考代码，不作为运行依赖
    ├── fdp/verify/verify/
    ├── tiqnqi/xmetai_model_verification/
    └── qihou/
```

**命名说明**：仓库当前真实路径是 `ref/tiqnqi/`，保留原名便于追溯；新任务配置使用 `tianqi`。FDP/Tianqi/Qihou 主要体现在配置模板中，不复制三套 Python 业务继承树。

## 4. 统一数据契约

### 4.1 数据载体与命名维度

模块边界使用 `DataBundle`，包含 `xarray.Dataset`、数据说明和来源记录。不强制一次加载整个集合或整年数据。以下为概念布局：

| 数据对象 | 典型维度与坐标 |
|---|---|
| 预报格点场 | 维度 `(init_time, lead_time, member, level?, lat, lon)` |
| 预报站点场 | 维度 `(init_time, lead_time, member, station)` |
| 观测格点场 | 维度 `(valid_time, level?, lat, lon)` |
| 观测站点场 | 维度 `(valid_time, station)`，经纬度作为站点坐标 |
| 配对后的评测批次 | 按 `sample` 组织；起报、有效时间、时效为样本坐标 |
| 指数/诊断数据 | 按需使用 `component`、`wavenumber`、`threshold`等命名维度 |

- `valid_time` 在预报中是由起报和时效确定的派生坐标，不另建独立的笛卡尔积维度。
- `lead_time` 内部统一为时长；配置中的 `lead_hours` 是小时数，不是文件序号。
- `member` 保留来源成员标识，不把 `member_001` 偷换成真实成员号 0。
- 确定性预报在标准预报产品中可表示为单成员；另有明确的 `forecast_kind=deterministic`。**成员数为 1 不足以推断产品类型。**
- 生成确定性场或概率产品后允许不带 member 维，但必须声明产品类型和生成方法。
- `z`（位势）与 `gh`（位势高度）作为不同物理量标识；选层和物理量转换分别表达。

### 4.2 必须记录的语义

每个变量记录：标准物理量、原始名称、原始/目标单位、垂直层、时间语义、缺测值、转换记录。每个数据源记录：格式、布局、时区/日历、网格、坐标名称、版本和可用成员。

时间语义至少区分：

- `instantaneous`：瞬时量。
- `interval_accumulation`：指定区间的累计量。
- `since_init_accumulation`：从起报开始累计，需要差分才能得到目标区间。
- `interval_mean`：指定区间平均量。
- 速率量：明确时间单位和积分方式，不等同于累计量。

累计和平均必须带 `time_start`、`time_end` 或可无歧义推导的边界。配对时同时比较有效时间和区间；不足完整窗口不得静默填 0。

### 4.3 标准化与对齐规则

- Reader 解码文件、映射命名、保留原始单位/时间信息；数值单位转换由显式 normalize 步骤完成。
- 规则经纬网第一版统一纬度升序、经度 `[0, 360)`，去除重复经度端点；保留转换记录。非规则网格必须有对应适配器，不能伪装成规则网格。
- 站点 ID 使用稳定标识，配对依据 ID 和坐标校验，不依据行序。
- 第一版时间规范使用 UTC；站点本地时由配置转换。遇到未支持日历应明确拒绝，不静默按公历解释。
- 变量映射固定采用 `标准名 → 源变量说明`，避免不同配置方向相反。
- 显式配置优先于自动探测；元数据与配置冲突时失败。无法确认单位、累计方式或时效映射时拒绝评分。
- 重网格、站点插值、区域裁剪和集合处理顺序进入计划与结果记录，不能由不同 Reader 自行选择。

### 4.4 样本与缺测

`SampleKey` 至少表达模型、起报、时效、验证时间区间、变量和验证目标。`EvaluationBatch` 包含预报产品、配对观测、共同有效掩码、权重、参考资料和样本标识。

默认规则：

- 配置错误、单位冲突、损坏文件：失败并给出原因。
- 明确允许的缺文件：记录缺失并按协议跳过，保存请求/有效样本数。
- 集合默认要求声明的成员齐全；允许不完整集合必须显式设置最小成员数及概率分母策略，并记录逐样本有效成员数。
- 不用 `NaN` 同时表示缺资料、指标不适用、零分母和程序异常；结果状态分别记录。

## 5. 核心接口与协作方式

以下是**接口草案，不是已经存在的代码，也不要求一次实现所有可选能力**。

```python
class Catalog:
    def discover(self, request: DataRequest) -> DataIndex: ...

class Reader:
    def read(self, selection: DataSelection) -> DataBundle: ...

class Transform:
    def apply(self, inputs: dict[str, DataBundle], params: dict) -> DataBundle: ...

class Metric:
    def requirements(self, params: dict) -> MetricRequirements: ...
    def accumulate(self, batch: EvaluationBatch, spec: MetricSpec) -> MetricState: ...
    def merge(self, states: list[MetricState]) -> MetricState: ...
    def finalize(self, state: MetricState) -> MetricResult: ...

class Step:
    def run(self, inputs: dict[str, ArtifactRef], context: RunContext) -> StepOutputs: ...

class Evaluator:
    def plan(self, task: TaskConfig) -> ExecutionPlan: ...
    def run(self, plan: ExecutionPlan) -> ResultBundle: ...
```

### 5.1 Catalog 与 Reader

Catalog 只负责枚举文件/样本和成员可用性，不在构造函数里加载全量预报。Reader 负责按选择读取；确定性与集合优先共用 NetCDF 读取能力，差异由布局适配器处理。

原 `read_mean` 的集合均值逻辑迁移到 `preprocess/ensemble.py`；如果来源已有集合平均产品，可直接读取，但必须记录成员范围和生成方式。**没有预计算均值时，计算集合平均仍需读取相应成员，不能声称完全省去成员 IO。**

GRIB 索引缓存写入运行缓存目录，不在只读数据目录产生索引文件。文件句柄由 Reader 明确管理生命周期。

### 5.2 MetricRequirements 与指标结果

输入要求至少包括：

- 产品类型：`deterministic_field`、`ensemble_samples`、`event_probability`、`index_series`。
- 所需变量和参考资料。
- 必须完整保留的维度，以及允许分块合并的维度。
- 阈值/窗口/权重/样本最低要求。
- 输出维度与支持的聚合语义。

不再用单一 `requires_all_members` 代替全部输入契约。不允许用户在 YAML 中把指标的固有要求改成 false。

`MetricResult` 是统一容器：可以携带标量评分、空间场、ROC曲线、谱或指数诊断，以及有效样本数、权重和、状态与协议标识。不得用 `float | DataArray` 的随机返回方式把解释责任交给报告层。

### 5.3 计算与聚合约定

| 指标族 | 目标输入/参考 | 汇总设计 |
|---|---|---|
| RMSE/MAE | 确定性场；集合均值需先显式派生 | 合并误差平方和/绝对值和及权重，再 finalize |
| TS/POD/FAR/频率Bias | 阈值事件 | 合并列联表；逐时次均分是独立命名的另一口径 |
| ACC/TCC | 声明相关轴、距平方式和基准 | 保留所需轴或可合并相关统计；零方差返回明确状态 |
| FSS | 格点场、事件阈值、邻域 | 保留邻域信息及边界/缺测策略，合并分子分母 |
| 经验集合 CRPS | 成员样本和观测 | 每个评分点需完整成员；可沿独立空间/样本维分块 |
| BS/BSS/AROC | 事件概率和观测事件；BSS另需参考 | 合并误差/参考误差或显式ROC统计，不默认平均每日比值 |
| Spread/SSR | 成员场；SSR另需配对观测 | 固定方差自由度及合并方式，再计算比值 |
| 活动度/功率谱 | 指定场、基准或完整空间轴 | 声明去均值、区域和谱定义，不随意拆断空间轴 |
| MJO | 配对RMM指数及专项协议 | 指数相关、误差、技巧时效分别输出 |

`accumulate/merge/finalize` 不意味着所有算法都能任意切块。对无法跨某轴合并的指标，Planner 必须保留该轴；超出内存预算时调整其他分块或拒绝任务。相关、谱、邻域算法不能以“通用并行”为由改变定义。

指标名必须消除歧义，例如 `frequency_bias` 与 `mean_error` 分别表达频率偏差与有符号平均误差。

### 5.4 专项步骤不进入通用核心

- 气候态支持观测基准与模型回报基准分别配置；保留按起报季节/预报时效变化的结构，不强制压成一个365天向量。
- 周聚合使用显式区间，包含第1–7天、第15–28天等组合窗口；尾部不足一周的策略必须配置，不误称为统一7天滑动平均。
- MJO 投影使用经确认、版本化的参考基底、变量顺序、符号和归一化参数，**不得迁移时自行重新拟合一套 EOF 代替历史基底**。
- 去120天均值等步骤声明所需历史观测范围，缺历史资料时按协议处理，不能只读取预报窗口。
- MJO 技巧时效的阈值、首次越界/插值/截断规则单独记录。

## 6. 执行、缓存与资源控制

### 6.1 统一执行计划

`ExecutionPlan` 是带显式依赖的步骤集合，第一版按拓扑顺序在本地执行。每个步骤声明输入引用、输出产物、参数、组件版本和数据契约。

标准任务由 Planner 自动展开：读取 → 标准化 → 窗口/空间处理 → 匹配 → 产品派生 → 指标 → 导出。气候与MJO任务增加共享气候态、距平和投影分支，但仍交给同一个 Executor。

Step 只包装已有 Reader/Transform/Metric，不把算法再写一遍。Pipeline 输入引用产物 ID，不靠重复手写缓存文件路径串联。

### 6.2 缓存有效性

缓存指纹包含：解析后的步骤参数、输入身份/版本或内容摘要、上游指纹、组件实现版本、协议与参考资料版本。

- 快速模式可使用文件大小和修改时间，但只作为明确标记的弱校验；严格复现使用内容摘要或可信不可变版本。
- 不默认对所有大文件重复全量哈希；输入清单和已确认的身份信息可复用。
- 缓存命中同时要求指纹一致、产物完整、契约匹配和成功状态存在。
- 文件先写同文件系统临时位置，通过校验后原子替换；多文件步骤最后发布完成清单。
- 恢复执行先验证依赖；指定某一步时，需要有效上游产物或显式重建上游，不能只跳过前面步骤。
- 缓存清理单独提供显式操作，不默认删除用户数据或参考产物。

### 6.3 分块与失败处理

- 第一版按起报、时效组和空间块控制内存，不全量加载“全年×全成员×全球场”。
- 成员读取和集合均值/概率产品可在同一处理链共享，避免每个指标重新开文件。
- FSS 分块需要邻域边界扩展；CRPS需要每点全部有效成员；TCC需要完整时间统计；谱需要规定空间轴。
- 若采用惰性数组后端，必须同时定义何时计算和关闭文件，不能仅返回未管理的文件句柄。
- 并发由一个执行层控制，避免 Reader、Step、Metric 各开进程池。
- 配置/科学语义错误不重试；可恢复 IO 错误可限次重试；下游依赖失败标为 blocked。
- 绘图失败可独立于评分结果报告；请求的指标依赖缺失时明确报错，不静默返回 NaN。

## 7. 配置驱动与验证协议

### 7.1 配置分层

1. `data_sources`：数据在哪里、如何解码、源单位/时间语义是什么。
2. `protocols`：比较什么、时间区间、目标网格/站点、阈值、权重、气候基准、缺测和聚合规则。
3. `tasks`：模型与数据源引用、请求指标、流程与输出。
4. `defaults`：执行资源和非科学默认值。

第一版只做显式引用和有限覆盖，不引入复杂的多重继承配置系统。路径支持环境变量；配置内相对路径相对于所属配置文件解析，解析后的绝对路径进入运行快照。命令行覆盖不得静默改变协议后仍沿用旧协议标识。

### 7.2 数据源示意（非可运行配置）

```yaml
schema_version: 1
id: example_ensemble
reader: forecast.netcdf
forecast_kind: ensemble
path:
  root: ${FORECAST_ROOT}
  template: "{init:%Y%m%d}/member_{member:03d}/{step:03d}.nc"
time:
  timezone: UTC
  lead_mapping:
    step_origin: 1
    first_lead_hours: 6
    step_hours: 6
members: [1, 2, 3]
coordinates: {latitude: lat, longitude: lon}
variables:
  tp:
    source_name: TP
    source_units: mm
    temporal_kind: interval_accumulation
    interval_hours: 6
```

此例中 `001.nc` 对应6小时，不是1小时；不能把 `{step}` 与 `{lead_hours}` 混用。实际源单位和时间语义需由文件与数据提供方确认，不能照抄示例。

### 7.3 任务示意（非可运行配置）

```yaml
schema_version: 1
id: precipitation_station_comparison
forecasts: [example_ensemble, example_deterministic]
observation: china_hourly_stations
protocol: precipitation_station_v1
selection:
  init_times: ["2025-01-06T00:00:00Z", "2025-01-07T00:00:00Z"]
  lead_hours: [24, 48]
  variables: [tp]
verification:
  windows_hours: [24]
  target: stations
  interpolation: bilinear
  weights: station_equal
  sample_policy: common_valid
metrics:
  - name: rmse
    product: deterministic_field
    ensemble_reduction: mean
  - name: ts
    product: deterministic_field
    ensemble_reduction: mean
    params: {thresholds: [0.1, 10.0, 25.0], operator: ge}
  - name: crps
    product: ensemble_samples
    models: [example_ensemble]
output:
  root: ${RESULT_ROOT}
  save_scores: true
  save_diagnostics: true
```

任务显式限定 CRPS 的模型范围，避免把“不适用”当成功执行。请求不兼容指标默认在规划期失败；显式允许跳过时，报告必须显示跳过原因。

### 7.4 预期使用入口

以下命令仅说明目标交互，核心代码和配置落地前不能运行：

```bash
pip install -e .
xmetai-eval validate --config configs/tasks/fdp/tp_ensemble.yaml
xmetai-eval plan --config configs/tasks/fdp/tp_ensemble.yaml
xmetai-eval run --config configs/tasks/fdp/tp_ensemble.yaml
xmetai-eval run --config configs/tasks/qihou/mjo.yaml --resume
xmetai-eval report --run-dir results/example_run
```

`validate` 校验配置结构与注册组件；`plan` 读取必要目录/元数据并检查契约，不执行指标计算。普通与专项任务使用同一个 run 入口，不另建第二套运行系统。

## 8. 结果、报告与公平比较

### 8.1 标准产物

```text
results/<run_id>/
├── resolved_config.yaml              # 完整解析后配置
├── manifest.json                     # 输入、协议、版本、状态、产物索引
├── scores.csv                        # 长表评分
├── scores.json                       # 可选机器可读评分
├── diagnostics/                      # 空间场、ROC、谱、RMM等NetCDF
├── states/                           # 可选可合并统计状态及其版本
├── coverage.csv                      # 请求/有效/缺失样本与成员覆盖
├── logs/
├── figures/
└── report.html                       # 或Markdown报告
```

评分表至少包含 `run_id/model/metric/variable/lead/window/region/threshold/product/aggregation/protocol_id/value/status/n_valid`，并按需添加层次、起报等字段。无关字段允许为空；时效与窗口单位统一。空间场不强制展开成巨型 CSV。

结果携带指标实现版本、单位、权重和、缺测原因、参考基准、来源配置与输入清单。报告仅消费结果和已生成诊断，缺少产物时提示，不偷偷重算。

### 8.2 可比性

- 单模型报告可使用自身有效样本，必须报告覆盖率。
- 正式多模型对比默认使用共同有效样本和验证掩码；按具体指标的产品要求确定比较组。
- 共同样本不只看起报日期，还包括有效时段、站点/格点有效性、变量和层次。
- 固定观测版本、目标域、插值、累计、阈值、权重与基准；模型集合规模及成员缺失规则作为比较元数据展示。
- 仅有独立汇总 CSV 时，不能声称已经按共同样本重新统计；需保存可合并状态/样本结果或重新评测。
- 外部公开评分可导入展示，但协议缺失或不一致时标为“不可直接比较”，不自动生成统一排名。
- 气候态/参考事件概率默认来自明确的独立基准期。复现历史脚本使用评测样本估计参考的做法时，单独命名并标注，不与独立基准结果混比。

## 9. ref 整合与迁移矩阵

目标是覆盖所有业务能力，不是逐行复制所有历史文件。以下为能力级迁移清单；实施时在 `docs/migration_inventory.md` 逐文件记录“待核查/已迁移/已验证/被共享实现替代/非运行资产”。不能因主脚本已迁移就宣布整类参考代码完成。

### 9.1 天气尺度：`ref/tiqnqi/xmetai_model_verification/`

| 来源 | 目标 | 保留与调整 |
|---|---|---|
| `vfc/reader_categorical.py`：ClassDataset | `io/forecast/` | 目录发现、成员/时效读取；与文件内member维统一 |
| 同文件 read_mean/read_members | Reader + `preprocess/ensemble.py` | 原始成员读取和均值派生分离 |
| `vfc/station_categorical.py`：HourlyStations | `io/observation/station_reader.py` | Diamond3解码、站点ID、缺测字段 |
| 同文件 accumulate_window/RefProb | `preprocess/accumulation.py`、参考数据模块 | 时间完整性、时区、参考概率基准显式化 |
| `vfc/metric_categorical.py`：interp_to_stations | `preprocess/interpolation.py` | 经度周期、边界处理转为可测策略 |
| 同文件 TSContingency/ProbEventHistogram | 分类/概率指标与状态 | 共享列联表/概率统计；成员数变化不得破坏概率分箱 |
| `run_categorical.py` | `configs/tasks/tianqi/` +统一入口 | 保留任务意图，不保留第二套主循环 |

### 9.2 FDP：`ref/fdp/verify/verify/`

| 来源 | 目标 | 保留与调整 |
|---|---|---|
| `ensemble_verifier.py` | 概率/集合指标库 | CRPS、Spread、SSR、BSS、AROC；基准与聚合拆出 |
| 同文件CRA/站点/模型读取 | `io/`、normalize | 变量映射、位势转换、通道式数据布局、索引缓存 |
| `tp_deterministic_verifier.py` | 分类指标、CMPAS Reader、降水任务 | TS/Bias/FSS、站点与格点验证分支、累计窗口 |
| `multi_model_verifier_fix.py` | 大气变量任务、确定性指标、comparison | 多变量/模型适配、验证对齐、结果对比 |
| `activity_spectrum_verifier.py` | specialized/activity、spectrum | 活动度与谱定义、所需空间轴、诊断输出 |

多个文件中的重复读取、单位转换、插值、字体设置和CSV导出只保留一套实现。模型特殊缩放在数据源配置中明确记录，不写成通用变量的隐式规则。

### 9.3 Qihou：`ref/qihou/确定性预测/`

| 来源 | 目标 | 保留与调整 |
|---|---|---|
| `step1_5_流程编排.py` | 长时效任务模板、ExecutionPlan | 集中配置与步骤依赖，替代subprocess调用链 |
| `step1_观测数据处理.py` | 气候态、距平、重网格、时间窗变换 | 观测处理与60天合并；闰日和跨年策略显式化 |
| `step2_模型数据处理.py` | 集合派生、模型气候态/距平 | 回报基准期、时效相关气候态，不硬编码模型 |
| `step3_模型和观测数据处理_step3（周平均）.py` | `preprocess/temporal.py` | 周与组合周区间、尾段、lead坐标约定 |
| `step4_绘图_tcc_rmse_bar_柱状图代码.py` | 指标库 +结果展示 | 先分离评分与绘图，再统一聚合口径 |
| `step5_绘图_tcc_rmse_空间图.py` | 空间指标结果 +spatial展示 | 空间场不先压成标量；缺文件策略进入协议 |

### 9.4 MJO：`ref/qihou/MJO评测/`

| 来源 | 目标 | 保留与调整 |
|---|---|---|
| `mjo_CRA_v1.5_full_pipeline.py` | MJO观测任务与共享变换 | 观测气候态、距平及多变量准备 |
| `fengshun_v1.5_mjo_full_pipeline.py`、`fengshun_v2.0_mjo_full_pipeline.py` | 数据源配置 +同一MJO流程 | 模型差异配置化，避免两个近重复流程 |
| `A_00_cal_S2S-ecmf_mjo_20240920_optimized.py` | `preprocess/mjo.py`、MJO步骤 | 历史窗口、投影、必要性能策略；并发交给Executor |
| `mjo_caculate.py` | `metrics/specialized/mjo.py` +diagnostics展示 | 指数相关、RMSE、技巧时效及展示分离 |
| 投影参数、辅助数据和说明 | 参考数据配置与provenance | 核查单位、符号、版本及使用条件，不作为无来源常量 |

`ref` 中的结果样例可作为受控回归参考；地理底图、辅助资料和文档需记录用途与来源，不全部复制进核心包。运行时通过资源配置读取必要资产。

### 9.5 迁移验收方法

每项迁移包含：输入契约确认 → 算法提取 → 配置替代硬编码 → 合成样例测试 → 同口径历史对照 → 差异说明 → 台账更新。

历史脚本用于对照，不是绝对真值。发现缺测、单位或统计口径问题时，保留可解释的差异记录和新测试；不得为了逐位一致而复制已知错误。

## 10. 如何扩展

### 新增指标

1. 在对应 `metrics/` 分类中实现算法，声明输入产品、参考数据、必需维度和输出。
2. 定义 accumulate/merge/finalize；不能合并的轴必须声明。
3. 注册稳定名称及版本，重复名称报错。
4. 编写已知答案、缺测/零分母/维度错误、分块一致性测试。
5. 在 YAML 请求该指标，不改 Evaluator 的主循环。

### 新增数据集或模型

- 已支持布局：新增数据源 YAML，指定路径、变量、单位、时间、成员和坐标。
- 全新格式/布局：实现 Reader 或布局适配器，返回标准契约，注册名称并添加小型 fixture。
- 新验证数据：复用 Matcher 与指标，不新增一套“某数据集专用评分器”。
- 自定义插件必须由明确安装/加载的代码注册；不根据不可信 YAML 任意 import 文件或执行表达式。

### 新增预处理或专项流程

- 新算法放 Preprocess，读写/缓存包装放 Step。
- 输入依赖全部声明；新增配置模板引用已有步骤。
- 只有通用执行能力不足时才修改 Executor，不能为一个专项追加业务分支。

### 新增图表或报告

实现消费 ResultBundle 的展示组件，注册类型。需要新诊断量时由指标或诊断步骤先生成，展示组件不读取原始预报自行评分。

## 11. 分阶段实施与验收

下列为实施顺序，不承诺未经数据规模验证的周数；第一版无需先创建完整目录树。

| 阶段 | 交付 | 验收门槛 |
|---|---|---|
| P0 契约与骨架 | 配置/registry、数据/结果协议、Planner/Executor最小接口、迁移台账 | 合成任务可校验并生成执行计划；未知组件和不兼容要求明确失败 |
| P1 数据闭环 | ForecastCatalog、NetCDF/站点/CRA/CMPAS接入、单位/时间/空间匹配 | 目录成员与文件成员维结果一致；UTC/本地时、累计区间可核验 |
| P2 FDP/Tianqi | RMSE/MAE、列联表/FSS、CRPS/BS/BSS/AROC/Spread/SSR，表格和覆盖报告 | 确定性/集合、站点/网格最小端到端用例通过；同口径ref对照完成 |
| P3 Qihou | 气候态、距平、周/组合窗、ACC/TCC、空间结果和缓存恢复 | 跨年/闰日/缺测/缓存失效测试通过；恢复与全量运行一致 |
| P4 专项与报告 | MJO、活动度、功率谱、多模型比较、单模型报告 | 基底/历史窗/谱定义核查；全部业务族迁移台账有结论 |
| P5 加固与扩展 | 性能、可选后端、插件文档、运维与配置样例 | 内存预算可控；分块/并发结果在约定容差内一致 |

P0 先定协议，P1/P2形成最小可用闭环；P3/P4检验长期与专项能力，不能到后期才发现接口只支持二维标量评分。

### 测试要求

- **算法测试**：已知答案、零误差、事件全无/全有、零方差、无有效样本、零参考误差。
- **契约测试**：变量/单位/坐标/日历/时区/成员类型/时间窗不一致应拒绝或显式处理。
- **聚合测试**：整批与分块合并一致；不把平均RMSE当总体RMSE，不把平均TS当合并列联表评分。
- **数据测试**：成员缺失、站点顺序变化、经度接缝、累计重置、重复文件/样本。
- **流程测试**：部分写入、依赖失败、参数/输入/组件版本变化导致缓存失效。
- **公平性测试**：验证共同有效样本和掩码；协议不同的结果不得直接排名。
- **回归测试**：按具体算法和数据类型设容差，参考代码有问题时记录差异，禁止一概追求逐位相同。
- **端到端测试**：小型确定性降水、集合概率、长时效距平、MJO任务分别验证产物及来源记录。

核心依赖、GRIB读取、绘图、专项算法及可选并行后端在打包时分组声明。实际依赖版本在实现阶段锁定和验证，不把未验证的版本写成安装要求。

## 12. 设计决策摘要与待落地事项

### 本版推荐基线

- 一套数据契约、一套指标扩展接口、一套执行机制、一套结果契约。
- 数据源配置 + 验证协议 + 任务配置；业务尺度通过模板表达。
- 命名维度和显式物理/时间语义，替代依靠数组维数判断类型。
- 指标需求驱动数据读取和派生，概率与原始成员需求分离。
- 可合并统计状态和多维结果，替代默认“逐时次float再mean”。
- 指纹和完成清单判断缓存，替代“文件存在即跳过”。
- 新增模型/数据集不增加核心分支；新增指标不修改任务主循环。
- ref 保留用于追溯，运行时不依赖 ref；完整迁移按能力台账验收。

### 实现前逐项核查，不在文档中猜测

真实源文件的变量、单位、累计方式和布局；允许的成员缺失策略；气候基准与闰日规则；MJO投影资产；首批验收数据规模和服务器内存；哪些外部报告具备可比较的协议。

上述细节落实到具体数据源/协议和测试样例，不阻塞总体分层设计，也不允许用隐式默认值替代核查。

## 13. 实施者必须遵守的编码契约

本节是给后续实施者的直接执行规范。若实现细节与本节冲突，应先更新设计文档或提出变更记录，不要在代码中形成未记录的第三套约定。

### 13.1 标准对象的最小字段

以下对象可以用 `dataclass`、Pydantic 模型或等价的不可变结构实现；具体技术选型不影响字段语义。

#### `DataRequest`

表示“希望读取什么”，不表示“已经读到了什么”。最少包含：

```text
source_id       数据源配置ID
variables       标准变量名列表
init_times      起报时间选择
lead_times      时效选择，使用时长而非文件序号
members         成员选择；None表示按产品策略选择
levels          层次选择，可为空
region          目标区域或裁剪规则
```

`DataRequest` 不允许放 `mean()`、插值算法或指标名称。指标需求由 Planner 转换为数据请求，不能让 Reader 反向读取整个任务配置。

#### `DataIndex`

表示 Catalog 的发现结果。每一条记录至少包含：

```text
sample_key      起报、时效、变量、成员等稳定键
uri             文件路径或可解析的数据位置
source_id       数据源ID
member_id       原始成员标识，可为空
variable_map    文件中实际变量名
time_map        文件中实际时间/时效信息
size/mtime      可选的输入身份信息
availability    available / missing / ambiguous / invalid
```

Catalog 发现到多个可能文件时，状态必须是 `ambiguous` 并阻断执行，不能按文件名排序后静默选一个。

#### `DataBundle`

表示已读取并经过最低限度结构标准化的数据。最少包含：

```text
payload         xarray.Dataset 或 DataArray
kind            forecast / observation / reference / derived
source_id       数据源ID
standard_vars   标准变量映射
semantic        单位、时间语义、网格和成员语义
provenance      输入文件、读取器版本、转换记录
quality         缺测、异常值、质量标记
```

Reader 可以保留源变量和源坐标作为辅助字段，但 `payload` 必须能通过数据契约校验。Reader 不得返回只在某个调用者内部有效的裸数组。

#### `EvaluationBatch`

表示已经可以交给指标计算的一批配对数据。至少包含：

```text
forecast        预报 DataBundle 或标准 DataArray
observation     观测 DataBundle 或标准 DataArray
reference       气候态/概率/投影基底等可选参考
sample_keys     每个样本的稳定标识
valid_mask      预报、观测和参考共同有效掩码
weights         空间/站点/样本权重及其语义
alignment       时间、空间、变量和单位对齐记录
```

Metric 不负责构造 `EvaluationBatch`。如果传入的数据没有 `alignment` 或 `valid_mask`，应在边界校验阶段失败，而不是在指标内部自行 `dropna`。

#### `MetricResult`

表示一项指标的最终结果。最少包含：

```text
metric_name     稳定指标名
metric_version  公式/实现版本
value           标量或带命名维度的诊断数据引用
coordinates     threshold、lead、region等结果坐标
status          success / partial / not_applicable / no_valid_data / failed
n_requested     请求样本或评分点数量
n_valid         实际有效数量
weights_sum     有效权重和（适用时）
aggregation     统计口径与聚合方式
protocol_id     验证协议ID
provenance      数据和参考来源
warnings        非致命问题列表
```

`value` 不建议把大型空间场直接塞入 JSON；ResultStore 应将其写入 NetCDF/Zarr 等诊断产物，并在 `MetricResult` 中保存引用和摘要。

### 13.2 Metric 的实现规则

每个 Metric 按以下顺序实现：

1. `requirements(params)`：返回所需变量、产品类型、维度、参考数据、不可分块轴和参数约束。
2. `validate(batch, spec)`：检查单位、维度、成员、坐标、时间语义和阈值；校验不能依赖数组位置。
3. `accumulate(batch, spec)`：只计算本批状态，不把状态写磁盘，不修改输入对象。
4. `merge(states)`：合并同一指标、同一协议、同一结果键的状态；参数和版本不一致时失败。
5. `finalize(state)`：生成 `MetricResult`，负责零分母、无有效数据和部分完成等状态。

小型指标可以在 `accumulate` 内直接完成计算，但仍要保留这五个概念边界，避免后续无法加入分块执行。

Metric 的以下行为一律禁止：

- 根据 `forecast.ndim` 猜测成员轴、时间轴或空间轴。
- 在内部自动把集合变成均值，除非该指标的 `requirements` 明确声明需要 `deterministic_field`，并且 Planner 已生成该派生产品。
- 读取文件、调用 Catalog、修改全局缓存或写图。
- 用异常捕获包住全部计算并返回 NaN。
- 在 `finalize` 中把逐批结果简单平均，除非协议明确规定该统计口径。

### 13.3 Reader 的实现规则

每个 Reader 分为三部分：

1. **Discovery**：根据 `DataRequest` 生成 `DataIndex`。
2. **Decode**：打开文件、读取源变量和坐标、处理源格式特有的编码问题。
3. **Normalize metadata**：把源信息转换为标准字段，但不做任务级插值、累计、气候态或评分。

Reader 必须：

- 对源变量不存在、变量候选冲突、单位未知、时间无法解析给出结构化错误。
- 只读取所请求的变量、成员和时间范围，不能为了简单而默认读整年全部成员。
- 明确关闭文件资源，或返回一个拥有明确生命周期的上下文对象。
- 保存实际读取的文件列表和源变量名，进入 provenance。
- 对同一请求保持确定性，不因目录遍历顺序改变结果。

Reader 不得：

- 在 `read()` 内部调用某个具体指标。
- 因为文件名包含 `tp` 就推断它一定是毫米累计量。
- 因为只有一个成员就自动把集合产品改标为确定性产品。
- 把缺失成员复制成其他成员来“补齐”集合。

### 13.4 Transform 与 Pipeline Step 的边界

`Transform` 是可复用的数据运算，输入和输出是 `DataBundle`；`Step` 是可缓存的执行包装，输入和输出是 `ArtifactRef`。同一套算法只能实现一次：

```text
纯算法：preprocess.accumulation.accumulate_time_window()
执行包装：pipeline.steps.transform.AccumulationStep
```

Step 必须声明：

```text
name            稳定名称
version         步骤实现版本
inputs          命名输入引用
outputs         命名产物
params          已解析参数
resources       内存/并发/临时空间提示
cache_policy    可缓存、不可缓存或强制重算
```

Step 不允许通过当前工作目录寻找隐含输入，也不允许在函数内部拼接生产路径。所有输入都来自 `ArtifactRef` 或显式配置。

### 13.5 Registry 与插件规则

Registry 是显式的组件目录，至少支持：

```text
reader.<name>
transform.<name>
metric.<name>
step.<name>
visualization.<name>
```

注册项包含名称、版本、工厂函数、能力描述和依赖提示。注册时发现重复名称必须失败；未注册名称必须在 `validate` 阶段失败。

第一版建议先使用代码内显式注册，稳定后再增加 Python entry points。无论采用哪种方式：

- 配置只能引用注册名和受校验的参数。
- 不允许从 YAML 读取任意模块路径并动态执行。
- 插件不能隐式覆盖核心指标或修改全局默认值。
- 插件返回的数据必须经过标准契约校验。

### 13.6 Planner 的确定性规则

Planner 是科学语义和资源需求的集中判断点。它应完成：

1. 解析数据源、协议和任务配置。
2. 加载 Reader/Catalog 的能力描述。
3. 汇总所有指标的 `MetricRequirements`。
4. 合并读取需求，避免相同文件被每个指标重复打开。
5. 插入必要的标准化、累计、插值、集合派生和匹配步骤。
6. 检查指标之间是否有不可兼容的处理顺序。
7. 检查成员数、内存估计、时间区间和参考数据是否足够。
8. 生成稳定、可打印、可审查的 `ExecutionPlan`。

Planner 不计算指标，不加载整批数据，不因为某指标不适用而静默删除用户配置。它要么生成包含状态的计划，要么在计划阶段给出可定位错误。

`plan` 命令的输出应至少列出：

```text
计划ID
数据源和文件数量估计
读取变量/成员/时效
每个处理步骤及依赖
每个指标及输入产品
预计输出产物
预计内存/并发提示
可能的跳过项及原因
```

## 14. 执行状态机与运行记录

### 14.1 Run 状态

一次运行的顶层状态固定为：

```text
PLANNED
  -> RUNNING
  -> SUCCEEDED
  -> SUCCEEDED_WITH_WARNINGS
  -> FAILED
  -> CANCELLED
```

Step 状态至少包括：

```text
PENDING       尚未满足依赖
READY         依赖满足，可以执行
RUNNING       正在执行
CACHED        使用了指纹匹配的有效产物
SUCCEEDED     本次执行成功
SKIPPED       按协议跳过，并有明确原因
BLOCKED       上游失败或输入不完整
FAILED        本步骤失败
```

状态改变必须写入运行日志或 manifest。进程在 `RUNNING` 状态异常退出时，下一次启动不能直接认为步骤成功；应通过完成清单、产物契约和指纹重新判断。

### 14.2 失败分类

错误类别建议固定为：

| 类别 | 示例 | 默认行为 |
|---|---|---|
| `ConfigError` | 字段缺失、类型错误、未知组件 | 计划阶段失败，不重试 |
| `ContractError` | 单位/维度/时间语义不符 | 当前步骤失败，不重试 |
| `DiscoveryError` | 文件缺失、重复匹配、路径歧义 | 按协议决定失败或记录缺失 |
| `DecodeError` | GRIB/NetCDF/站点文件损坏 | 可重试一次后失败 |
| `AlignmentError` | 起报、有效时段、网格无法匹配 | 步骤失败，保留样本诊断 |
| `MetricError` | 零分母、参数非法、公式无法计算 | 生成明确结果状态或失败 |
| `ResourceError` | 内存、磁盘、并发资源不足 | 可调整分块后重试 |
| `OutputError` | 写入失败、校验失败、发布失败 | 运行失败，不发布半成品 |

不能用裸 `Exception` 作为用户最终看到的科学错误。底层异常应保留为 cause，并附带数据源、样本键、变量、路径和步骤名。

### 14.3 日志最小字段

每条结构化日志至少包含：`run_id`、`step_id`、`sample_scope`、`component`、`level`、`event`、`message`、`timestamp`。数据文件路径可进入日志和 manifest，但大批量样本不应逐条刷屏；详细清单写入机器可读文件。

## 15. 配置 Schema 与版本策略

### 15.1 Schema 版本

所有配置顶层必须有 `schema_version`。配置结构变更时递增版本，加载器只负责支持明确列出的版本，不根据字段猜测旧版本。

建议分三种版本：

```text
schema_version       配置结构版本
protocol_id/version   科学验证口径版本
component_version     Reader/Metric/Step实现版本
```

它们不能互相替代：修改阈值属于协议变化；修复 CRPS 实现属于组件变化；新增 YAML 字段属于 schema 变化。

### 15.2 参数覆盖优先级

建议优先级从低到高为：

```text
defaults.yaml
< 被引用的数据源/协议配置
< task配置
< 命令行显式覆盖
```

覆盖后必须生成完整的 `resolved_config.yaml`。运行 manifest 中保存覆盖来源；如果覆盖了科学口径字段，必须生成新的协议摘要或拒绝覆盖。

### 15.3 科学参数的必填规则

以下参数不能依赖框架默认值：

- 物理单位和时间语义。
- 起报时间到文件时效的映射。
- 集合成员列表和缺失成员策略。
- 观测时区和日历。
- 降水累积窗口与边界是否闭开区间。
- 空间插值/重网格方法和目标网格。
- 阈值、事件比较符和概率分母。
- 纬度权重/站点权重/区域权重。
- 气候态参考期和 BSS 参考概率。
- MJO 投影基底版本、变量顺序、归一化参数。

执行资源、日志等级、临时目录等工程参数可以有安全默认值，但也要写入解析后的配置。

## 16. 结果键、状态与聚合口径

### 16.1 结果唯一键

单项评分不能只用 `(metric, lead_time)` 标识。推荐的逻辑唯一键至少包括：

```text
model_id
variable
level
metric_name
product_kind
init_group 或 valid_period
lead_time/window
region/target
threshold
aggregation
protocol_id
```

缺少某个维度时显式使用 `null` 或 `not_applicable`，不要复用空字符串造成不同语义碰撞。

### 16.2 状态和值分离

`value = NaN` 不是状态协议。结果必须同时有 `status`：

```text
success             有效结果
partial             有效但部分样本/成员缺失，符合协议
not_applicable      产品类型不支持该指标
no_valid_data       请求存在但没有有效配对
undefined           数学定义遇到零分母/零方差
failed              实现或输入错误
```

对于 `undefined`，可以选择不发布数值或发布空值，但必须保留分母、样本数和原因。

### 16.3 空间与时间聚合

实现者必须区分以下统计：

- `mean_over_samples`：评分点或样本上的平均。
- `aggregate_then_score`：先合并误差/列联表，再计算评分。
- `score_then_average`：先逐样本评分，再平均，必须显式命名。
- `area_weighted`：使用经纬度/区域权重。
- `station_equal`：站点等权。

结果中同时保存 `aggregation` 和 `weights_id`。报告层不得看到一个 `value` 就自行决定如何跨时效、阈值或区域求平均。

## 17. 第一批代码的明确任务边界

实施者拿到本设计后，建议按以下顺序提交小步变更。每一步都应该可测试，不要先创建大量空目录。

### Change 1：核心契约与错误

创建：

```text
xmetai_evaluation/core/contracts.py
xmetai_evaluation/core/variables.py
xmetai_evaluation/core/errors.py
xmetai_evaluation/results/schema.py
```

只实现：标准维度/语义常量、`DataRequest`、`DataIndex`、`DataBundle`、`EvaluationBatch`、`MetricResult`、错误类别和基础校验。

验收：用合成 xarray 数据验证合法/非法维度、单位、成员和时间语义；不接任何真实数据，不实现指标。

### Change 2：显式 Registry 与配置校验

创建：

```text
xmetai_evaluation/core/registry.py
xmetai_evaluation/core/config.py
```

只实现：注册、重复检测、未知组件检测、环境变量解析、路径解析、schema_version 检查和 resolved config 输出。

验收：正确配置通过；缺字段、重复组件、未知组件、非法覆盖和环境变量缺失均有明确错误。

### Change 3：Metric 基类与最小结果流

创建：

```text
xmetai_evaluation/metrics/base.py
xmetai_evaluation/metrics/states.py
```

只实现 Metric 协议和一个极小的 RMSE 参考实现，用合成数据测试整批与分块合并一致性。

禁止在此阶段实现所有指标、真实 Reader 或报告系统。

### Change 4：Pipeline 最小执行器

创建：

```text
xmetai_evaluation/pipeline/base.py
xmetai_evaluation/pipeline/context.py
xmetai_evaluation/pipeline/cache.py
xmetai_evaluation/pipeline/executor.py
```

只支持内存中的合成 Step 和本地文件产物，完成拓扑排序、状态机、指纹、完成清单、失败传播和原子发布。

验收：成功、缓存命中、上游失败、参数变化失效、部分写入恢复、resume 五类测试通过。

### Change 5：Catalog/NetCDF 最小闭环

在真实接入 CRA、CMPAS 和 Diamond3 之前，先完成：

```text
xmetai_evaluation/io/base.py
xmetai_evaluation/io/forecast/catalog.py
xmetai_evaluation/io/forecast/netcdf_reader.py
```

只支持一个合成 NetCDF 布局和两个真实布局：成员子目录、文件内 member 维。输出必须经过 DataBundle 契约校验。

验收：确定性、集合目录、集合文件三种 fixture 的成员识别、时效映射、变量映射和元数据保持一致。

### Change 6：首个端到端任务

先实现一个最小确定性降水任务：NetCDF 预报 + 合成格点观测 + RMSE/TS + CSV/manifest。

然后再接入：

1. 站点插值和 Diamond3；
2. CMPAS；
3. 集合 CRPS/BS/AROC；
4. Qihou Pipeline；
5. MJO、活动度和功率谱。

这样可以在进入复杂格式和专项算法前验证整个框架的扩展点是否真实可用。

## 18. 交付审查清单

实施者提交任何模块时，审查者至少检查：

- 是否修改了不必要的核心边界或新增了隐式全局状态。
- 是否所有输入、输出、单位、维度、时间窗和缺测规则都有来源。
- 是否用命名维度而非位置索引判断数据含义。
- 是否把读取、预处理、指标、绘图混在同一个函数。
- 是否能在小 fixture 上测试，不依赖生产路径、服务器环境或 ref 脚本。
- 是否记录了 `protocol_id`、组件版本、输入清单和有效样本数。
- 是否区分了“无数据”“不适用”“数学未定义”“程序失败”。
- 是否验证了整批与分块、缓存命中与重新执行、确定性与集合布局的一致性。
- 是否增加了对应的迁移台账状态和回归差异说明。
- 是否把未确认的物理单位、时间语义或气候基准写成了默认值。

如果某项能力暂时无法满足上述约束，应在结果中标记为未实现或在设计变更中说明，不能通过降低校验强度来“先跑起来”。
