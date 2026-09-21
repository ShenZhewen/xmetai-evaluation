# -*- coding: utf-8 -*-
"""确定性连续量检验（风清单卡 × ERA5）：对标 xu ``scripts/single_fengqing.sh``。

与 ``weather_rmse_single_fuxi.py`` 同流程、同单位口径，差别只有两处数据事实：
风清单卡输出里没有 q2m，且地面风变量名是 ``U10``/``V10``（由布局翻译成
标准的 ``u10m``/``v10m``，与 ERA5 zarr 对齐）。

    xu --vars z500 q700 t700 t2m t850 msl u200 v200 u850 v850 u10m v10m ws10m ws850 ws200
    xu --var-metrics ...（transcribed below；fa -> activity、spectrum -> zonal_spectrum）

单位口径（与 xu 报告层一致，便于逐格对拍）：
    z500 报 m²/s²（不除 g）、q 报 g/kg、tp 报 mm。

``ws10m`` / ``ws850`` / ``ws200`` 由 u/v 分量按 ``sqrt(u²+v²)`` 现合成。

内存：``grid_valid_time`` 会一次读入评测时段内全部有效时刻，默认给的是
冒烟级小区间；正式跑直接改下面的 ``start_date`` / ``end_date`` / ``limit``。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# xu: --vars z500 q700 t700 t2m t850 msl u200 v200 u850 v850 u10m v10m ws10m ws850 ws200
VARS = [
    "z500", "q700", "t700", "t2m", "t850", "msl",
    "u200", "v200", "u850", "v850", "u10m", "v10m",
    "ws10m", "ws850", "ws200",
]

# xu: --var-metrics（逐条转写；fa -> activity、spectrum -> zonal_spectrum）
VAR_METRICS = {
    "z500": ["rmse", "zonal_spectrum", "acc", "activity"],
    "q700": ["rmse"],
    "t700": ["rmse", "zonal_spectrum"],
    "t2m": ["rmse", "zonal_spectrum"],
    "t850": ["rmse"],
    "msl": ["rmse", "zonal_spectrum"],
    "u200": ["rmse", "activity", "zonal_spectrum"],
    "v200": ["rmse", "activity", "zonal_spectrum"],
    "u850": ["rmse", "activity", "zonal_spectrum"],
    "v850": ["rmse", "activity", "zonal_spectrum"],
    "u10m": ["rmse", "activity"],
    "v10m": ["rmse", "activity"],
    "ws10m": ["activity", "rmse"],
    "ws850": ["activity", "rmse", "zonal_spectrum"],
    "ws200": ["rmse", "zonal_spectrum"],
}

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
    name="weather_rmse_single_fengqing",
    description="确定性连续量检验（风清单卡 × ERA5）：15 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",

    forecast_reader={
        "type": "fengqing_phys",
        "root_dir": os.environ.get(
            "FENGQING_OUTPUT", "/workspace/data/shenzw/fengqing_output"
        ),
        "variables": VARS,
        "step_hours": 6.0,      # 布局步长：lead_from="filename"，文件名里的 3 位时效 × 6h
        # 时效列表（小时）。**不写 = 从目录索引推**：取目录里的最大时效 lead_max，
        # 按 step_hours 从 0 铺起。要"不限制"写 None 或整个键不写，两者等价。
        #
        # 本流程**没有累积窗**（weather_field_scores 的变换只有 ensemble_mean），
        # 每个读到的时效都是独立样本，所以可以自由写稀疏集：
        #     "lead_times": [24, 48, 72, 96, 120],   # 只评前 5 天，省读盘省内存
        # 写出来就是**覆盖**：框架不再从目录推，块也只按列表里的时效读。
        "lead_times": None,
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
        # 15 天环形平滑（±7.5 天，跨年首尾相接）：取场前对年内日序做中心对齐的
        # 滑动平均，抹平单日气候态的天气尺度噪声。与 xu 的 --climo-window 同口径，
        # 是**气象参数**，跟 execution.loads 的 window:W（IO 窗块缓存）没有关系。
        "smooth_days": 15,
    },

    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20250102"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_rmse_single_fengqing",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # 逐波数谱曲线（720 个点）不走 value——value 里每个分组键都会展开成明细行，
    # 2165 行/样本 × 20880 样本 ≈ 四千五百万行。改用 spectrum writer：出
    # diagnostics/spectrum_{var}.csv（全体均值）与 spectrum_by_init.csv（逐起报）。
    # 与模板自带的 writers 一致，写出来是为了好改。
    writers=["csv_long", "spectrum"],

    # ⚠ 采样口径尚未与 weather_rmse_single_fuxi 对齐，正式跑前请定夺。
    # 本配置没写 options，走协议缺省 valid_time——「一个有效时刻一个样本、最新
    # 起报获胜」。逐日起报配 15 天时效时，后一个起报的短时效会把前一个起报的
    # 长时效覆盖掉，每个起报只落得下头 4 个时效，后面 56 个 step 在归并时全被
    # 扔掉。同流程的 fuxi 那份显式写了 init_lead（每个 (起报, 时效) 一个样本）。
    # 要改就取消下一行的注释；现在保持缺省不动，结果与改动前逐位一致。
    # options={"sample_by": "init_lead"},

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
    #   chunk_days       一个工作块装几个**起报日**（缺省 1）——管起报跨度
    #   lead_chunk_days  一个工作块装几天**时效**（缺省 1）——管时效跨度，
    #                    是单块内存的**主旋钮**。取值 0 = 不切（整段时效一块，逃生口）
    #   loads            角色驻留覆盖 {角色: 策略}，角色只有 observation / reference：
    #                      slice      每个块现读现弃
    #                      resident   整个 run 读一次驻留内存
    #                      window:W   按 W 天块滚动缓存（LRU），省相邻块的重复读
    #                      window     裸写不带数字 = 自动定窗，按**一个起报日组的
    #                                 观测日跨度**算（整段时效跨度 + 预热 + 起报跨度）
    #                      window:W   W 是**上限**，实际用 min(W, 自动值)——比自动值
    #                                 大不减少重读、只多占内存（进程模式下窗块是每个
    #                                 子进程各一份）；比它小则是主动拿内存换 IO。
    #                                 实际落成的值一律回写成 "window:N" 记进 manifest
    #                                 的 execution.loads 供核对（收到过的另打一行日志）
    #                    推导缺省：站点观测 / 气候态 = resident，其余 = slice。
    #                    参考源是 ref_probability 时逐样本直读，不可配这项
    #   resume           True 时已完成块的状态落 output_dir/.states/，重跑跳过
    #
    # 下面 loads 两行就是推导结果，写出来是为了好改。代价：写死之后不再自动
    # 跟随数据源——换了观测/参考读取器要记得回来核对一遍。
    #
    # 规模：lead 到 360h、6h 步长 → 按 lead_chunk_days=1 切出 16 个时效窗；
    # 起报数 × 16 窗 = 总块数（段数 = min(n_workers, 待跑块数)）。
    #
    # 内存账（**未实测，按 weather_rmse_single_fuxi 的量级外推**）：
    # 日序气候态 resident 随要素数线性涨——fuxi 那份 12 要素实测 67.25G / 991.7s，
    # 本配置 15 要素更多，**正式跑前务必先小样本量一次**（日志里 loader 会打
    # 「角色 reference 预热完成：耗时 X.Xs，驻留 X.XX GB」）。单块工作集同理，
    # 随 chunk_days × lead_chunk_days 线性缩。
    # 驻留那部分是 fork 前预热、子进程 COW 共享的**固定份额**，与 worker 数无关，
    # 所以加 worker 只在工作集上线性花钱。别把 reference 改成 window:——窗块是每个
    # 子进程各自物化的（不共享），反而更贵。
    #
    # ⚠ 观测的 window 是**另一笔账**（2026-09 从 slice 改过来）：era5_zarr 是立即读进
    # numpy 的，窗块真占内存、而且每进程一份。稳态留 3 个块（当前块 ±1，回跳要用），
    # 单通道一个 16 天块 ≈ 0.27G；本配置读 13 个通道 → **~10.4G/进程**，
    # n_workers=24 就是 ~250G，再加气候态 resident 会爆。
    # 所以正式跑这份配置前必须二选一：n_workers 压到 6~8（~83G，代价是并发降下来），
    # 或者把 observation 退回 slice（读放大 16 倍，但不占常驻内存）。
    # **这份的 n_workers 卡口是内存，不是核数**——别照抄 fuxi 那份的 24。
    execution={
        "mode": "processes",
        "n_workers": 24,        # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        # observation 从 slice 改成 window：块序是「起报日外层、时效窗内层」，
        # 相邻起报组的观测日大面积重叠（扫完 16 天再回跳 14 天），滚动窗块让一个
        # 观测日整 run 只读一次，而不是每个块都重读。
        # 16 = 整段时效跨度 15 天（lead 360h）+ 1 天起报跨度，即自动定窗的值
        # （见 execution/plan.py 的 _auto_window_days）；写更大不会更省，只会多占
        # 内存，所以这里写 16 与裸写 "window" 等价——显式写出来是为了好改。
        "loads": {
            "observation": "window:16",
            "reference": "resident",    # 日序气候态整 run 驻留（fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
