# RMSE 批次家族：口径与产出

覆盖 `weather_rmse_<模型>_single` / `_ens` / `_wave` 三条支线。它们**不走
`generate_report.py` 的能力路由**——吃的是多份**批次归档目录**
（`summary.csv` + `batch_meta.json` + `<YYYYMMDD>/`），**没有 `manifest.json`**，
认不出能力，所以走独立入口 `scripts/generate_det_report.py`。

**冲突时以代码为准。** 本文件是代码的散文镜像：

| 事实 | 权威来源 |
|---|---|
| 归档目录的形状 | `vfc/regr_ens.py` / `vfc/regr_summary.py` 的写盘代码 |
| 派生指标的定义 | `visualization/det_report.py` 的**模块 docstring**（那几个指标**没有生成脚本，是反推出来的**，docstring 里写明了） |
| 报告长什么样 | `assets/templates/weather_rmse_*.md` 三份骨架 |

改口径先改代码，再回来改这里。

---

## 0. 三条支线

| 支线 | config | 吃什么 | 骨架 | 渲染器 |
|---|---|---|---|---|
| 确定性单成员 | `weather_rmse_<模型>_single` | `--summarize-det` 的批次归档 | `weather_rmse_single.md` | ✅ `visualization/det_report.py` + `scripts/generate_det_report.py` |
| 集合 | `weather_rmse_<模型>_ens` | `--summarize-ens` 的批次归档 | `weather_rmse_ens.md` | ✅ `visualization/ens_report.py` + `scripts/generate_ens_report.py` |
| 谱检验补充 | `weather_rmse_wave_<模型>` | 批次归档 + `spherical_bands_*` | `weather_rmse_wave.md` | ✅ `visualization/wave_report.py` + `scripts/generate_wave_report.py` |

三条**各有各的入口，按支线选**：`_single` 用 `generate_det_report.py`，
`_ens` 用 `generate_ens_report.py`，`_wave` 用 `generate_wave_report.py`。
骨架是**格式契约**，三份章节口径互不相同——**拿错入口比报错危险**：
`_ens` / `_wave` 的归档都能被 `generate_det_report.py` 读进来而不报错
（它认 `*_ensmean.csv` 后缀），但出来的章节口径不是集合/谱检验的。

单模型的入口用法：

```bash
python skills/xmetai-evaluation/scripts/generate_det_report.py \
  --model FGVP=/path/to/single_fgvp \
  --model AIFS=/path/to/single_aifs \
  --out reports/det_4models
```

`--model NAME=目录` 可重复、**至少两个**（单模型对比没有意义）。
其余选项：`--out`（必给）、`--title`、`--change`、`--archive`、
`--declared NAME=v1,v2,...`（对应骨架第 6.2 节的「声明变量数」，不给则该列记 `—`）。

---

## 1. 归档形状

### 1.1 批次归档目录

一个模型一个目录，由 `vfc/regr_ens.py` 批量跑出来：

```
<model_root>/
  summary.csv                 每起报日一行：init_date, n_members, n_leads, variables,
                              rmse_<变量>_{24h,last,mean}
  batch_meta.json             pred_root / target_zarr / n_dates / dates / n_ok /
                              failures / outdir_root / options / mean_files
  <YYYYMMDD>/                 ← 逐日起报的明细，报告真正读的是这里
```

`batch_meta.json` 的 `n_ok` / `n_dates` / `failures` 就是骨架第 2 节
「完整性检查」那张表的来源。

### 1.2 逐日明细文件

| 指标 | 文件名 | 形状 |
|---|---|---|
| RMSE | `rmse_<date>_det.csv`（集合是 `rmse_<name>_ensmean.csv`） | index = `lead_h`，columns = 变量 |
| ACC | `acc_<date>_det.csv` | 同上 |
| FA | `fa_<date>_det.csv` | columns = `<变量>_{pred,obs,bias,ratio}` |
| 纬向谱 | `spectrum_<date>_<变量>.csv` | index = 波数 `k`，columns = `{det_pred, det_obs}` |
| CRPS（仅集合） | `crps_<name>.csv` | index = `lead_h`，columns = 变量 |
| Spread（仅集合） | `spread_<name>.csv` | 同上 |
| Spread/RMSE（仅集合，可选） | `spread_rmse_ratio_<name>.csv` | 同上 |

**ACC 与 FA 需要气候态**，配置里没给 `climo_path` 就不会有这两个文件——
报告里对应的表会整块消失，**这不是缺数据**。

### 1.3 写盘时顺带生成的汇总表（**报告不读**）

归档目录里还会有这些，但 `det_report.py` **一概不读**：

| 文件 | 谁写的 | 口径 |
|---|---|---|
| `mean_<tag>.csv`、`mean_spectrum_<变量>.csv` | `vfc/regr_ens.py` | 跨日平均，**该模型自己的全部日期** |
| `det_summary_*.csv`（`--summarize-det`） | `vfc/regr_summary.py` | **全日期**口径的汇总 |
| `ens_summary_*.csv` / `.png`（`--summarize-ens`） | 同上 | CRPS 与 Spread/RMSE 的跨日汇总 |

**为什么不读**：报告正文与附录图统一用 N 个模型**共同的日期**
（被模板化的那份原报告有「正文 169 天 / 附录 349 天」的双口径，这里不保留）。
拿全日期口径的汇总表混进来，会让表与图各说各话——**表里是 349 天、图里是 169 天**。

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
**要方向必须回去看 `ratio` 本身**（`fa_<date>_det.csv` 里 `_ratio` 列在）。

### 3.4 谱指标

`频谱对数 RMS×100` 是对**全波数**算的，高波数的近零观测功率会主导结果。
要定位"从第几个波数开始失真"必须看谱曲线（`spectrum_<date>_<变量>.csv`），
一个汇总数说不出来。

> ⚠️ **底数记录**：实现（`det_report.py`）用 **log10**，
> 而 `weather_rmse_wave.md` 骨架的行文写的是 **ln**。两者差 ×2.3026。
> **以代码为准（log10）**；骨架那处是照抄源报告的文字，尚未与实现核对。

---

## 4. 口径参数（跨批次比较必须一致）

`vfc/regr_pair.py` 的 `regr_pair()` 收一批参数，**它们变了数就不可比**：

| 参数 | 作用 | 坑 |
|---|---|---|
| `pred_shift` | pred 相对 obs 的时步错位 | 本批数据 pred 不含 0 时次，取 1；**含 0 时次的新数据取 0**。搞错就是整体错位一个时次 |
| `lead_step` | 时效轴重标（小时） | 文件头步长可能标错（如 `time` 轴是 `linspace`），要按真实步长传 |
| `climo_path` / `climo_source` / `climo_window` | 气候态 | **ACC 与 FA 必需**；没有就没有这两个指标 |
| `acc_centered` | ACC 是否用减加权平均的经典皮尔逊 | 默认 `False` = **uncentered**（FDP / WeatherBench2 口径）。**两种口径的 ACC 不可比** |
| `spec_lead_range` | 功率谱平均的时效窗 | `None` = 全部。不同窗的谱不可比 |
| `metrics` | 算哪些指标 | 只有 `("rmse",)` 时，ACC / FA / 谱**整块没有** |

---

## 5. 坑

1. **不要把 `det_summary_*.csv` 喂给报告**——它是全日期口径，和共同日期口径的表
   放一起会自相矛盾（见 §1.3）。
2. **`--model` 少于两个要报错**，不要拿单模型的归档去套多模型骨架
   （骨架里到处是「{模型数}模型」「四条曲线」，一个模型填不出来）。
3. **归档目录不是产物目录**。它没有 `manifest.json`，`generate_report.py`
   认不出它——反过来也一样，别把 `weather_field_scores` 的产物目录丢给
   `generate_det_report.py`。
4. **模型名由调用方给**（`--model NAME=目录`），**不从目录名猜**——
   这个仓库的规矩是身份不靠目录名。
5. **谱变量默认优先 `z500`**（`PREFERRED_VARIABLE`），缺了才退到字典序第一个。
   换变量会让频谱那条线整体变样，**别拿不同谱变量的数比大小**。
6. **球谐带功率是有的**，别再照旧文档说「没有数据来源」：
   `vfc/metrics/spectrum.py:210 spherical_band_power` + `:270 band_power_frame`
   已经在写 `spherical_bands_<date>_<var>.csv`（实测 1308 份）。
   但**列名前缀同族归档不一致**——`weather_rmse_ens_*` 两个归档里既有
   `spherical_pred_1_4`（AIFS）也有 `ensmean_spherical_pred_1_4`（FGVP），
   连列序都不一样（AIFS 逐带交错，FGVP 先全 pred 再全 obs 再全 ratio）。
   `wave_report._band_column:176` 按名字两种前缀都认，**不受影响**；
   自己写新读取代码时别只看前缀或只看列序。

---

## 6. 不要做的事

1. 不要给相对 RMSE 的 `100` 编绝对含义（它是**本次参赛模型的几何均值**）；
2. 不要把不同单位的 RMSE 画在同一根纵轴上；
3. 不要用 FA 偏差的绝对值断言"偏平滑"或"偏噪"——它丢了方向；
4. 不要拿全日期口径的汇总表和共同日期口径的图混着讲；
5. 不要用"结果里没有"反推"数据源里没有"（ACC/FA 缺席多半是没配气候态）；
6. 不要把 `_ens` / `_wave` 的归档喂给 `generate_det_report.py`——
   它读得进去（认 `*_ensmean.csv`），但出的是 `_single` 的章节口径，
   **拿错模板比报错危险**；
7. 不要跨口径比较（不同起报集合、不同 `acc_centered`、不同 `spec_lead_range`、
   不同 `pred_shift` 的结果不可直接横向比）。

---

**维护者**：沈哲文 (szw)
**最后更新**：2026-09-15
