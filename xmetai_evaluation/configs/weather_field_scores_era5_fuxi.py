# -*- coding: utf-8 -*-
"""确定性连续量检验（FuXi × ERA5）：对标 xu ``scripts/single_fuxi.sh``。

流程：``weather_field_scores``（格点有效时刻配对；集合先降维）。
要素表与逐变量指标表**逐条转写** xu 的命令行，只是把名字换成框架里的对应物：

    xu --metrics rmse spectrum acc fa   ->  rmse / zonal_spectrum / acc / activity
    xu --vars  ...                      ->  VARS
    xu --var-metrics z500:rmse,acc,fa   ->  VAR_METRICS（再由
                                            metric_options_from_var_metrics 反转）

单位口径也对齐 xu 的报告层，所以结果 CSV 能逐格对拍：
    z500 报 m²/s²（**不除 g**，xu 的 z500 单位就是 ``m2 s-2``）
    q    报 g/kg（源文件已是 g/kg；xu 内部转 kg/kg、报告层再 ×1000，两步相消）
    tp   报 mm（ERA5 zarr 源是 m，已在 layout 里 ×1000）

``ws850`` 任何文件里都没有，由 ``sqrt(u²+v²)`` 现合成（预报、实况、气候态三侧
都合成）。``ws10m`` 例外：ERA5 store 自带 10m 风速通道，实况侧直接取用
（见 ``layouts.ERA5_ZARR_LAYOUT`` 的 ws10m 条目），预报与气候态两侧仍是现合成。

**与 xu 的偏差**：xu 的 ``--vars`` 是 16 个，这里只留 12 个——砍掉的 4 个在两侧数据里
凑不齐，留着只会每块刷一条"已跳过"的 WARNING、指标也出不全：``q2m`` ERA5 store 没有；
``u200`` / ``v200`` 预报侧没有（运行时 WARNING 点名，而 ERA5 侧是有的）；``ws200``
靠这两个分量现合成，分量缺了自然出不来。逐变量指标表已同步删干净。跟 xu 的 CSV
逐格对拍时，这 4 个变量对不上属预期。

xu 默认把 tp 排除在格点指标外（走站点 TS）。要复刻它的 ``--tp-grid``，
把 ``"tp"`` 加进下面的 VARS 和 VAR_METRICS 即可。

内存：``grid_valid_time`` 会把一段评测时段内**全部**有效时刻读进来，
12 个要素 × 全球 0.25° 很吃内存。真正的旋钮在 ``execution`` 里：单块内存由
``lead_chunk_days``（时效跨度）与 ``chunk_days``（起报跨度）共同决定，逐键
说明见下面 ``execution`` 那段注释。

本配置当前跑的是**全年段**（``limit=None``，348 个起报）。想先冒烟再正式，
把 ``limit`` 改成 2 之类的小值跑一遍，改回 ``None`` 重跑即可——``resume`` 打开时
冒烟算过的块会被复用，不会白算。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# xu: --vars z500 q2m q700 t700 t2m t850 msl u200 v200 u850 v850 u10m v10m ws10m ws850 ws200
#     本配置只取其中 12 个（砍掉 q2m / u200 / v200 / ws200，理由见模块 docstring）
# VARS = [
#     "z500", "q700", "t700", "t2m", "t850", "msl", "u850", "v850", "u10m", "v10m",
# ]

VARS = [
    "z500",
]

# xu: --var-metrics（逐条转写；fa -> activity、spectrum -> zonal_spectrum）
VAR_METRICS = {
    "z500": ["rmse", "zonal_spectrum", "acc", "activity"],
}


# VAR_METRICS = {
#     "z500": ["rmse", "zonal_spectrum", "acc", "activity"],
#     "q700": ["rmse"],
#     "t700": ["rmse", "zonal_spectrum"],
#     "t2m": ["rmse", "zonal_spectrum"],
#     "t850": ["rmse"],
#     "msl": ["rmse", "zonal_spectrum"],
#     "u850": ["rmse", "activity", "zonal_spectrum"],
#     "v850": ["rmse", "activity", "zonal_spectrum"],
#     "u10m": ["rmse", "activity"],
#     "v10m": ["rmse", "activity"],
# }

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# xu 的 ACC 是 uncentered（FDP / WeatherBench2 口径），不是经典皮尔逊
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# xu 的谱取到 720 波（全球 0.25° 的 Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

cfg = EvalConfig(
    name="weather_field_scores_era5_fuxi",
    description="确定性连续量检验（FuXi × ERA5）：12 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",

    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
        ),
        "variables": VARS,
    },
    observation_reader={
        "type": "era5_zarr",
        "stores": {
            "pl": os.environ.get(
                "ERA5_PL_STORE",
                f"{_ERA5_STORE_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
            ),
            "sfc": os.environ.get(
                "ERA5_SFC_STORE",
                f"{_ERA5_STORE_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
            ),
        },
        "variables": VARS,
    },
    reference_reader={
        "type": "daily_climatology",
        "root_dir": os.environ.get("ERA5_CLIMO", "/workspace/data/worm/era5_clim_phys_14.nc"),
        # 与 xu 的 --climo-window 一致：15 天环形平滑
        "window": 15,
    },

    start_date=os.environ.get("START_DATE", "20250102"),
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_field_scores_era5_fuxi",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # 逐波数谱曲线（720 个点）不走 value——value 里每个分组键都会展开成明细行，
    # 2165 行/样本 × 20880 样本 ≈ 四千五百万行。改用 spectrum writer：出
    # diagnostics/spectrum_{var}.csv（全体均值）与 spectrum_by_init.csv（逐起报），
    # 对标参考实现的 det_summary_spectrum_{var}.csv / spectrum_{init}_{var}.csv。
    writers=["csv_long", "spectrum"],

    # 采样口径：每个起报各报满 15 天（60 个 step），逐个 step 出分。
    # 缺省是 valid_time——「一个有效时刻一个样本、最新起报获胜」的连续场口径；
    # 逐日起报配 15 天时效时，后一个起报的短时效会把前一个起报的长时效覆盖掉，
    # 每个起报只落得下头 4 个时效（347×4 + 末个起报 60 = 1448 行/指标），
    # 后面 56 个 step 全在归并时被扔掉。init_lead 给每个 (起报, 时效) 一个样本。
    options={"sample_by": "init_lead"},

    # ── 并发与数据加载：execution ──────────────────────────────────────
    # 可用键**就下面这 6 个**，全部是字面量，只覆盖写到的键（不写 = 用推导值）。
    # 写错键名会直接报配置错，不会静默忽略。
    #
    #   mode             并发形态，四选一：
    #                      serial     单进程顺序跑块（冒烟、单块用）
    #                      threads    1 个进程内 N 线程并发块；resident/window
    #                                 全局只有 1 份（loader 有锁），适合轻指标
    #                      processes  进程池，块按**时间连续分段**、一段固定一个
    #                                 子进程顺序处理（段内相邻块能命中窗块缓存）
    #                      auto       推导：单块 → serial；重指标 → processes；
    #                                 轻指标 → threads
    #                    本流程含谱（zonal_spectrum / activity）→ 推导即 processes
    #   n_workers        进程/线程数。推导缺省 processes = CPU 核数、threads = 4。
    #                    进程模式下它的上限是 `内存 ÷ 单worker工作集`，不一定是核数
    #                    （这份配置实测下来内存不是卡口，核数才是，见下面的内存账）
    #   chunk_days       一个工作块装几个**起报日**（缺省 1）——管起报跨度
    #   lead_chunk_days  一个工作块装几天**时效**（缺省 1）——管时效跨度，
    #                    是单块内存的**主旋钮**。延伸期/集合预报（15 天 × 51 成员）
    #                    不切时效时单块要把整段时效 × 成员全物化（约 25GB）。
    #                    取值 0 = 不切（整段时效一块，与改造前逐位一致，逃生口）
    #   loads            角色驻留覆盖 {角色: 策略}，角色只有 observation / reference：
    #                      slice      每个块现读现弃
    #                      resident   整个 run 读一次驻留内存
    #                      window:W   按 W 天块滚动缓存（LRU），省相邻块的重复读
    #                      window     裸写不带数字 = 自动定窗，计划层按**单块**
    #                                 观测跨度算（时效窗 + 预热 + 起报跨度，threads
    #                                 下再加并发跨度）；落成的值会回写成 "window:N"
    #                                 记进 manifest 的 execution.loads 供核对
    #                    推导缺省：站点观测 / 气候态 = resident，其余 = slice。
    #                    参考源是 ref_probability 时逐样本直读，不可配这项
    #   resume           True 时已完成块的状态落 output_dir/.states/，重跑跳过
    #
    # 本配置把推导值也显式写出来了（下面 loads 那两行就是推导结果），改起来一目了然。
    #
    # 规模：lead 到 360h、6h 步长 → 按 lead_chunk_days=1 切出 16 个时效窗；
    # 348 个起报 × 16 窗 = 5568 块，24 段各 232 块（段数 = min(n_workers, 待跑块数)）。
    #
    # 内存账（实测，全年段）：日序气候态 resident 预热 67.25G / 991.7s；单 worker
    # 工作集 ~2.0G（16 要素时的量；现在 12 个只会更小）。总量 ≈ 67 + 2.0 × n_workers
    # —— n_workers=24 时约 115G，240G 上限下占不到一半。**卡口是核数不是内存**。
    # 驻留那 67G 是 fork 前预热、子进程 COW 共享的**固定份额**，与 worker 数无关，
    # 所以加 worker 只在工作集上线性花钱。别把它改成 window:——窗块是每个子进程
    # 各自物化的（不共享），日序气候态一个 3 天窗块就 ~6G，24 个进程反而比 67G 贵。
    # 单块工作集 = 预报窗口 + 观测切片 + 配对批次，随 chunk_days × lead_chunk_days 线性缩。
    # 反过来，块数变多会让**每块的固定开销**（重建组件、catalog discover、预报
    # 索引）按块数线性涨；真嫌这块慢，把 lead_chunk_days 调到 2~3 减块数。
    execution={
        "mode": "processes",
        "n_workers": 24,       # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        # loads 这两行就是推导值，写出来是为了好改。代价：写死之后不再自动
        # 跟随数据源——换了观测/参考读取器要记得回来核对一遍。
        "loads": {
            "observation": "slice",      # era5_zarr 按请求跨度切，现读现弃
            "reference": "resident",     # 日序气候态整 run 驻留（实测 67G，fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
