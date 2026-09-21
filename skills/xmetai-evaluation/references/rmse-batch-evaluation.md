# RMSE 批次家族：口径与产出

覆盖 `weather_rmse_single_<模型>` / `weather_rmse_ens_<模型>` 两条配置支线，
以及吃同一批产物的 `_wave` 谱检验补充报告（**模型名在最后**，别写成
`weather_rmse_<模型>_single`——那是旧文档的写法）。

**它们的产物形状就是标准评估产物目录**（`scores.csv` + `manifest.json` +
`diagnostics/`），与其它流程没有区别，`resolved_config.pipeline` 是
`weather_field_scores`。**形状不是它们单开入口的原因，口径才是**：

| | 吃几份产物 | 出什么 |
|---|---|---|
| `generate_report.py` → `field_report.py` | **一份** | 单模型的连续场报告（RMSE / ACC / 活跃度 / 谱） |
| `scripts/generate_det_report.py` → `det_report.py` | **N 份**（≥2） | N 个模型**横向对比**的报告（相对 RMSE、配对 Wilcoxon、名次） |

`generate_report.py` 认得出这些产物（有 `manifest.json`、有
`resolved_config.pipeline`），它只是**不做多模型对比**。反过来，把 TS 的产物目录
丢给 `generate_det_report.py` 会当场报错——它要 `scores.csv` 里
`metric="rmse"` 且 `product_kind="deterministic"` 的行。

**冲突时以代码为准。** 本文件是代码的散文镜像：

| 事实 | 权威来源 |
|---|---|
| 产物目录的形状 | `xmetai_evaluation/output/store.py` 的写盘代码（`scores.csv` / `manifest.json`）+ `components.py`（`diagnostics/`） |
| 老批次归档 ↔ 新长表的等价关系 | `visualization/det_report.py` 的**模块 docstring**（有一张三列对照表） |
| 派生指标的定义 | 同上（那几个指标**没有生成脚本，是反推出来的**，docstring 里写明了） |
| 报告长什么样 | `assets/templates/weather_rmse_*.md` 三份骨架 |

改口径先改代码，再回来改这里。

> ⚠️ **旧文档里的 `vfc/` 在本仓库不存在**（只有 `xmetai_evaluation/`）。
> 本文件早期版本引用的 `vfc/regr_ens.py` / `vfc/regr_summary.py` /
> `vfc/regr_pair.py` 与 `--summarize-det` / `--summarize-ens` 两个开关
> **一并作废**：那套「批次归档 + 逐日 CSV」的中间层已经没有了，
> 评测框架直接落 `scores.csv` 长表。下文凡提到这几个名字的地方都是历史沿革，
> 现行口径一律以 `xmetai_evaluation/` 下的代码为准。

---

## 0. 三条支线

| 支线 | config | 吃什么 | 骨架 | 渲染器 |
|---|---|---|---|---|
| 确定性单成员 | `weather_rmse_single_<模型>` | 评估产物目录（`metric="rmse"` + `product_kind="deterministic"`） | `weather_rmse_single.md` | ✅ `visualization/det_report.py` + `scripts/generate_det_report.py` |
| 集合 | `weather_rmse_ens_<模型>` | 评估产物目录（多 `crps` / `spread` / `spread_error_ratio` 指标） | `weather_rmse_ens.md` | ✅ `visualization/ens_report.py` + `scripts/generate_ens_report.py` |
| 谱检验补充 | 吃上面任一份产物 | 评估产物目录 + `spherical_bands` 指标 | `weather_rmse_wave.md` | ✅ `visualization/wave_report.py` + `scripts/generate_wave_report.py` |

三条**各有各的入口，按支线选**：`_single` 用 `generate_det_report.py`，
`_ens` 用 `generate_ens_report.py`，`_wave` 用 `generate_wave_report.py`。
骨架是**格式契约**，三份章节口径互不相同——**拿错入口比报错危险**：
集合与单成员的产物目录形状**一模一样**，`generate_det_report.py` 拿集合产物
不会报错（它挑 `product_kind="deterministic"` 的行，集合的 rmse 行恰好也有
确定性那一份），出来的却是 `_single` 的章节口径。

单模型的入口用法：

```bash
python skills/xmetai-evaluation/scripts/generate_det_report.py \
  --model FGVP=/path/to/weather_rmse_single_fgvp \
  --model FuXi=/path/to/weather_rmse_single_fuxi \
  --result report/det_2models
```

`--model NAME=目录` 可重复、**至少两个**（单模型对比没有意义），
`NAME` 是**报告里显示的模型名**，由调用方给、**不从目录名或
`manifest.run_id` 猜**——这个仓库的规矩是身份不靠目录名。
其余选项：`--result`（必给，报告输出目录）、`--title`、`--change`、`--archive`、
`--declared NAME=v1,v2,...`（对应骨架第 6.2 节的「声明变量数」，不给则该列记 `—`）。

---

## 1. 产物形状

### 1.1 评估产物目录

一个模型一个目录，由评测框架（`pipeline/runner.py` + `output/store.py`）直接落盘：

```
<model_root>/
  scores.csv                  统一长表，23 列，见 §1.2——报告唯一的取数来源
  manifest.json               运行记录：statuses / valid_times / regions /
                              n_results / metrics / resolved_config / artifacts
  diagnostics/
    spectrum_<变量>.csv       全体样本加权平均谱：wavenumber, pred_mean, obs_mean
    spectrum_by_init.csv      逐起报谱：variable, init_time, wavenumber, pred, obs
  .states/                    逐段状态缓存（pkl），resume 用，**报告不读**
```

**没有 `summary.csv` / `batch_meta.json` / `<YYYYMMDD>/` 了**，
那是中间层还在时的形状。逐日明细的等价物是 `scores.csv` 的 `init_time` 列。

> ⚠️ **`manifest.json` 里的 `artifacts` 存的是写盘时的绝对路径**
> （`/workspace/.../scores.csv`）。**产物目录被拷到别处后这些路径就失效了**，
> 别拿 `artifacts` 定位文件——一律用产物目录内的相对路径。

### 1.2 长表 `scores.csv`

一行一个 `(变量, 指标, 区域, 起报, 时效)` 组合，23 列：

```
run_id, model_id, dataset_id, variable, level, metric, product_kind, aggregation,
weights_id, region, threshold, window_h, lead_h, init_time, valid_time,
sample_unit, value, unit, status, n_valid, n_requested, metric_version, protocol_id
```

报告取数只看其中几列：

| 报告里的东西 | 长表里怎么认 |
|---|---|
| 分变量 RMSE | `metric="rmse"` **且** `product_kind="deterministic"`，按 `init_time` × `lead_h` 透视 |
| ACC | `metric="acc"` |
| FA ratio | `metric="activity_ratio"` |
| 纬向谱偏差 | `metric="spectrum_power_ratio"`（标量汇总，恒全球）；逐波数曲线在 `diagnostics/`。**指标名叫 `zonal_spectrum`，落盘的 `metric` 值却是 `spectrum_power_ratio`**，两者不是同一个字符串 |
| 起报日 | `init_time` 前 10 位（`2025-01-02T00:00:00.000000` → `20250102`） |
| 全球行 / 纬度带行 | `region` 为空 = 全球，非空 = 该带（见 §1.3） |

> ⚠️ **`metric="rmse"` 有两套**：确定性 RMSE（`product_kind="deterministic"`）
> 与集合 `spread_error` 展开出来的那个 rmse（`product_kind="ensemble"`）。
> **名字一样、含义不同**，取数必须先按 `product_kind` 过滤。
> 渲染器里这一步是 `det_report._deterministic()`，别绕过它。

**ACC 与 FA 需要气候态**（`reference_reader`），配置里没给就不会有这两类行——
报告里对应的表会整块消失，**这不是缺数据**。

### 1.3 全球行与纬度带行

配了 `options["regions"]` 时，**格点标量指标**的每一个 `(起报, 时效)` 会写一行
全球行（`region` 为空）加每个带一行。实拍的三带是：

```
tropics[-20, 20]      nh_extratropics[20, 90]      sh_extratropics[-90, -20]
```

`n_valid`（格点数）三种行各不相同：全球 1038240、两个中高纬带各 404640、
热带 228960 —— **这个比例可以用来核对区划是否真的切开了**。

> **哪些指标分带、哪些不分**：`rmse` / `acc` / `activity_*` / `crps` /
> `spread` / `spread_error_ratio` **分带**；**谱类恒全球**——
> `spectrum_power_ratio` 与 `spherical_band_power_*` 的行 `region` 全为空。
> 判据是「这个量能不能在子区域上重新定义」，谱类不能。
> 所以「附 L」那一块拿到的是这些标量指标的分带值，**谱没有分带版本**，
> 骨架里分纬度带那张谱表会标「本批未出」，这不是数据没跑到。

> ⚠️ **两类行混在一起 `pivot_table(aggfunc="mean")` 会静默把带间差异平均掉**，
> 得到一堆看着正常、其实无意义的数。报告正文一律先过 `_global()` 只留全球行；
> 「附 L 分纬度带」那一块反过来只留 `region` 非空的行。**两者永不合并。**

带名与边界**照搬配置**，报告不翻译、也不写死——`det_report.region_names()`
是从 `scores.csv` 的 `region` 列现读的，所以换个切法重跑，报告那边一个字都不用改
（`manifest["regions"]` 存的是带名 + 边界的显示形式，两者对不上时以长表为准）。

### 1.4 谱诊断表（`diagnostics/`）

逐波数的谱**不在长表里**（长表一行一个变量的总量），走 `spectrum` writer 落到
`diagnostics/`：

| 文件 | 形状 | 谁在用 |
|---|---|---|
| `spectrum_<变量>.csv` | `wavenumber, pred_mean, obs_mean`，全体样本加权平均 | `field_report` 的分波段那节 |
| `spectrum_by_init.csv` | `variable, init_time, wavenumber, pred, obs`，逐起报一条 | `det_report.load_spectrum()`，报告里按日期画的谱曲线都取自它 |

**没有 `spectrum_by_init.csv` = 配置里没挂 `spectrum` writer**，报告保留节位、
写明「本批未出」，不静默省略。谱曲线**没有逐时效的版本**（长表里
`zonal_spectrum` 是标量，没有波数轴；`spectrum_by_init` 也没有 lead 列），
所以骨架里「谱比随时效」那一节现在必然是本批未出，这不是数据没跑到。

---

## 2. 统一口径与派生指标

### 2.1 共同日期

所有统计（表与图）都用 N 个模型的**共同日期**，逐日结果按 lead 平均后比较。
因此各表的样本量一致、可直接横向比较。**跨批次比较必须用同一批起报日**——
"涨到几倍"这个说法只在当前这套输入下成立。

### 2.2 派生指标（照 `det_report.py` docstring 抄）

| 指标 | 定义 | 方向 |
|---|---|---|
| **相对 RMSE** | 逐变量 `100 × v_m / geomean_模型(v)`，再对**共享变量**等权平均 | 越低越好，**100 = 几何均值基线** |
| **ACC (%)** | 共同日期 × 全 lead 的 ACC 变量平均 | 越高越好 |
| **FA 偏差 (pp)** | `100 × mean(\|ratio − 1\|)`，对全部 FA 变量、全 lead 平均 | 越低越好 |
| **频谱对数 RMS×100** | `100 × mean_v sqrt(mean_k (log10(pred_k / obs_k))²)` | 越低越好 |
| **平均排名分** | 上述四个指标上名次的平均 | 越低越好 |

> **相对 RMSE 的 100 是"这几个模型的几何均值"**，不是某个绝对标准。
> 换一批模型参赛，同一个模型的分就变了——**它只在本次对比内部有意义**。
> 基线由 `det_report.py` 的 `BASELINE = 100.0` 钉住。

### 2.3 时效分段

```
LEAD_BANDS = ((0,120], (120,240], (240,360)]   单位 h，左开右闭
```

**空段不输出**，段名按段内实际的 lead 现写——所以模型时效范围不同时，
表的行数可能不一样。

### 2.4 显著性

逐日期配对的 **Wilcoxon 符号秩检验** + **Holm 多重校正**（手写在 `_holm`，
五行，不为它引入 statsmodels）。**统计显著 ≠ 业务重要**：样本量够大时
0.001 的差距也会显著，报告里必须同时给出差值本身。

### 2.5 FA 这个缩写的歧义

这里的 **FA = Forecast Activity**（预报活跃度，距平加权标准差），
**不是**降水二分类里的 False Alarm。两份报告都用这个缩写，所以
`weather_rmse_*.md` 的骨架里都有一段专门声明，**不许省**。
且 FA 统一用**全程**（首–末时效）统计、不做短中长分段——分段只给 RMSE。

---

## 3. 判读要点

### 3.1 相对 RMSE 看"倍数"而不是"差值"

`100` 是几何均值基线：`89.3` 表示比几何均值好约 11%，`117.6` 表示差约 18%。
**跨变量比倍数是可以的**（比值无量纲），比原始 RMSE 的差值是安全的——
原始 RMSE 的单位随数据源走（`z500` 可能是 `m²/s²` 也可能是 `m`），
**不同单位的 RMSE 不能画在同一根纵轴上比大小**。

### 3.2 ACC 饱和要单独提示

`det_report.py` 里有 `SATURATED_ACC = 99.0`：首时效 ACC 高于 99% 就算"饱和"，
写进 6.3 的风险提示。ACC 到 99% 以上时，**模型之间的差异已经淹没在噪声里**，
这时候比 ACC 的名次没有意义，要看 RMSE 和谱。

### 3.3 FA 偏差是"偏离 1 的幅度"

`ratio = pred 活跃度 / obs 活跃度`，理想值 1。报告用的是
`mean(|ratio − 1|)`——**取了绝对值，方向就丢了**：两个模型一个偏平滑（0.8）
一个偏噪（1.2），FA 偏差都是 0.2，但含义完全相反。
**要方向必须回去看长表里 `metric="activity_ratio"` 的行**
（`activity_bias` / `activity_forecast` / `activity_observation` 三行也在，
分别是分子分母的原始量）。

### 3.4 谱指标

`频谱对数 RMS×100` 是对**全波数**算的，高波数的近零观测功率会主导结果。
要定位"从第几个波数开始失真"必须看谱曲线
（`diagnostics/spectrum_by_init.csv` 取单日起报，或 `spectrum_<变量>.csv` 看平均谱），
一个汇总数说不出来。

> ⚠️ **底数在两条支线之间不一致，这是现行代码的真实状态、不是文档过时**：
> `det_report.py` 用 **log10**（`:810`），`ens_report.py`（`:282`）与
> `wave_report.py`（`:253`、`:389`）用 **ln**，两者差 ×2.3026。
> 两个渲染器都**在正文里自己声明了底数并互相点名**，所以单看一份报告不会错；
> **跨份报告比这个数之前必须先统一底数**。要不要把三条支线对齐是一个
> 待定决策（改了会让已有归档的数整体变样，所以没有擅自动）。

---

## 4. 口径参数（跨批次比较必须一致）

现在这些参数都写在**评测配置**（`xmetai_evaluation/configs/weather_rmse_*.py`）里，
换一个值就是换一个口径，**两批结果的数不可横向比**：

| 配置项 | 作用 | 坑 |
|---|---|---|
| `options["sample_by"]` | 采样口径：`"init_lead"`（每起报报满整段时效）或 `"valid_time"`（每有效时刻一个样本、最新起报获胜） | **这是当年 `pred_shift` 那个坑的现代表达**。逐日起报配 15 天时效**必须**用 `init_lead`，否则每个起报只剩头 4 个时效 |
| `forecast_reader["step_hours"]` | 时效轴步长（小时） | 等于当年的 `lead_step`。写错整条时效轴就整体缩放 |
| `reference_reader`（`type: daily_climatology` / `climatology`） | 气候态 | **ACC 与 FA 必需**；没配就没有这两类行（见 §1.2） |
| `reference_reader["smooth_days"]` | 日序气候态的环形平滑天数（实测 15 = ±7.5 天） | 等于当年的 `climo_window`。**不同平滑窗的气候态不可比** |
| `metric_options["acc"]["centered"]` | `True` = 经典皮尔逊（距平去均值）；`False` = **uncentered**（FDP / WeatherBench2） | 实测配置用的是 `False`。**两种口径的 ACC 不可比** |
| `metric_options["zonal_spectrum"]["max_wavenumber"]` | 谱取到第几个波数（720 = 0.25° Nyquist） | 写小了只截曲线前段。等于当年的 `spec_lead_range` 那一类"谱口径" |
| `metric_options`（每个指标各自的 `variables` 列表） | 算哪些变量 × 哪些指标 | 只有 `rmse` 时，ACC / 谱**整块没有**；漏点名的变量会拿到整批数据、`single_variable` 直接报错（**框架故意的，防静默出假数**） |
| `options["regions"]` | 分纬度带区划 | 改了区划，`region` 列的值就变了——**带间不可跨批次比**（见 §1.3） |

> **`pred_shift` / `spec_lead_range` 这两个老参数在新框架里没有对应物**，
> 对齐由框架按 `sample_by` 自己算。旧文档里"pred 不含 0 时次取 1"那条经验
> 是针对老数据的，**不要往新配置里搬**。

---

## 5. 坑

1. **全球行和纬度带行不能混着平均**——`pivot_table(aggfunc="mean")` 会把带间
   差异**静默**抹平，出一堆看着正常的假数（见 §1.3）。正文只留全球行，
   「附 L」只留带行。
2. **`--model` 少于两个要报错**，不要拿单模型的产物去套多模型骨架
   （骨架里到处是「{模型数}模型」「四条曲线」，一个模型填不出来）。
3. **`metric="rmse"` 有两套口径**，取数必须先按 `product_kind` 过滤
   （见 §1.2）。忘记过滤 = 把集合的 `spread_error` 展开项混进确定性 RMSE。
4. **模型名由调用方给**（`--model NAME=目录`），**不从目录名或
   `manifest.run_id` 猜**——这个仓库的规矩是身份不靠目录名。
   `manifest["run_id"]` 是**配置名**（如 `weather_rmse_single_fuxi`），
   与目录名、与报告里的模型名**三者可以全都不一样**；实测把 FuXi 的产物
   拷进一个叫 `..._fgvp` 的目录，`run_id` 照样是 `..._fuxi`。
5. **谱变量默认优先 `z500`**（`PREFERRED_VARIABLE`），缺了才退到字典序第一个。
   换变量会让频谱那条线整体变样，**别拿不同谱变量的数比大小**。
6. ⚠️ **球谐带功率行在旧产物里认不出频带**。长表契约里有 `group` 列
   （球谐带的 `1_4` / `5_20` …，见 `output/table.py:37`），**共 24 列**；
   但现存归档的 `scores.csv` 只有 **23 列、没有 `group`**——那是修 `group`
   之前跑出来的产物。此时 5 个频带的行混在一起无法区分，
   `wave_report` 会走 `SPHERICAL_ABSENT`、§5.1–5.3 标「本批未出」。
   **修法是重跑评测**（配置里带 `spherical_bands` 指标即可），不是改报告端。

---

## 6. 不要做的事

1. 不要给相对 RMSE 的 `100` 编绝对含义（它是**本次参赛模型的几何均值**）；
2. 不要把不同单位的 RMSE 画在同一根纵轴上；
3. 不要用 FA 偏差的绝对值断言"偏平滑"或"偏噪"——它丢了方向；
4. 不要把全球行与纬度带行混着平均（见 §5.1）；
5. 不要用"结果里没有"反推"数据源里没有"（ACC/FA 缺席多半是没配气候态；
   谱曲线缺席多半是没挂 `spectrum` writer；球谐带认不出多半是旧产物没有 `group` 列）；
6. 不要把集合的产物喂给 `generate_det_report.py`——**形状一样、读得进去**，
   出的却是 `_single` 的章节口径，**拿错模板比报错危险**；
7. 不要跨口径比较（不同起报集合、不同 `acc` 口径、不同气候态平滑窗、
   不同 `max_wavenumber`、不同 `regions` 区划的结果不可直接横向比）；
8. 不要跨份报告比「频谱对数 RMS」——三条支线的底数不一致（见 §3.4）。

---

**维护者**：沈哲文 (szw)
**最后更新**：2026-09-21（按 `xmetai_evaluation/` 的实际产物形状重写 §1 与 §4；
作废 `vfc/` 系列路径与 `--summarize-*` 开关）
