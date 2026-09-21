# -*- coding: utf-8 -*-
"""要素检验（Fengqing × CRA40）：RMSE / Bias / ACC。

流程：``fdp_field_scores``（格点有效时刻配对；集合先降维）。

数据：
    预报  Fengqing 单卡（z500，逐 6h，文件名里带 3 位时效）
    观测  CRA40 再分析（框架现有 ``cra`` reader，gh@500hPa）
    参考  气候态 —— **ACC 需要，RMSE / Bias 不需要**；见下面 CRA_CLI_ROOT 那段

规模：起报与时效都由配置显式钉死（``init_times`` 一个起报 × ``lead_times`` 5 个
时效 = 5 个样本），不是按 start/end 滚一整段。所以这份是**单场次冒烟/对比**用的
配置，跑起来只有 1 个块。

内存：单块工作集很小（1 个起报 × 5 个时效 × 1 要素）。CRA40 是 GRIB2、按请求
跨度现读现弃；气候态若给了就是整份驻留。逐键说明见下面 ``execution`` 那段。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="fdp_rmse_single_fengqing",
    description="FDP 要素检验：z500 的 RMSE / Bias / ACC",
    pipeline="fdp_field_scores",

    forecast_reader={
        "type": "fengqing",
        "root_dir": os.environ.get(
            "FDP_FENGQING_ROOT", "/mnt/d/气象研究/春燕_weather_test_bash/fengqing"
        ),
        "variables": ["z500"],
        "step_hours": 6.0,      # 布局步长：lead_from="filename"，文件名里的 3 位时效 × 6h
        # 时效列表（小时）。本流程**没有累积窗**（fdp_field_scores 的变换只有
        # ensemble_mean），每个读到的时效都是独立样本，所以这里可以写稀疏集——
        # 与降水流程（weather_ts_det / weather_ts_ens）不同，那些必须按 6h 密排。
        "lead_times": [24, 48, 72, 96, 120],
        # 起报时刻显式钉死；给了它就不再按 start_date/end_date 去目录里找起报。
        # 一个起报 × 5 个时效 = 5 个样本，所以下面 start/end 对本段其实不起作用。
        "init_times": ["2026-08-19T00:00:00"],
    },
    observation_reader={
        "type": "cra",
        "root_dir": os.environ.get(
            "FDP_CRA_ROOT", "/mnt/d/气象研究/春燕_weather_test_bash/cra_root"
        ),
        "variables": ["z500"],
    },
    # 气候态参考是**有条件的**，这是语义不是风格：没设 CRA_CLI_ROOT 就当作"没有
    # 参考源"，而不是退回一个默认路径。后果：ACC 用零场兜底、状态标 partial 且
    # 日志给警告，**那个 ACC 数字没有意义**；RMSE / Bias 不受影响。要让 ACC 有效
    # 就 export CRA_CLI_ROOT=<气候态文件或目录>。
    reference_reader=(
        {"type": "climatology", "root_dir": os.environ["CRA_CLI_ROOT"]}
        if os.environ.get("CRA_CLI_ROOT")
        else None
    ),

    start_date=os.environ.get("START_DATE", "20260819"),
    end_date=os.environ.get("END_DATE", "20260819"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/fdp_rmse_single_fengqing",
    ),
    # json 是**在本模板默认的 ["csv_long"] 之上追加的**（配置级 writers 是替换，
    # 所以这里必须把模板那份也写全，漏了 csv_long 主表就没了）。
    writers=["csv_long", "json"],
    log_level="INFO",

    # 协议口径。本配置不写 options —— 模板自带的 {"ensemble_reduction": "mean"}
    # 就是全部，而配置级的 options 是**合并**不是替换（spec.options 先取模板再
    # update 配置），所以真要加也只需写增量。
    # 另注意 ACC 的口径：本模板给的是**经典皮尔逊**（距平场再减域加权均值）；
    # FDP / WeatherBench2 的 uncentered 口径是另一个指标 acc_uncentered。

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
    #                    本流程的 rmse / bias / acc 都是轻指标。但**具体落地成什么
    #                    取决于块数**：本配置钉死了 1 个起报 × 5 个时效、chunk_days=1
    #                    → 只切出 1 个块 → 落地 serial（单块没必要并发）；哪天把
    #                    init_times 铺成一整段，块数上去就会变成 threads。
    #                    所以这一项写 auto 而不是某个具体形态——写死了反而会在
    #                    起报数变化时卡在错误的选择上。结果三者都一样，只影响速度。
    #   n_workers        进程/线程数。推导缺省 processes = CPU 核数、threads = 4
    #   chunk_days       一个工作块装几个**起报日**（缺省 1）——管起报跨度
    #   lead_chunk_days  一个工作块装几天**时效**（缺省 1）——管时效跨度。本配置
    #                    只有 lead 24~120 共 5 个采样点，落成 5 个时效窗；取值
    #                    0 = 不切（整段时效一块，本配置只有 5 个时效，完全可行）
    #   loads            角色驻留覆盖 {角色: 策略}，角色只有 observation / reference：
    #                      slice      每个块现读现弃
    #                      resident   整个 run 读一次驻留内存
    #                      window:W   按 W 天块滚动缓存（LRU），省相邻块的重复读
    #                      window     裸写不带数字 = 自动定窗；落成的值回写成
    #                                 "window:N" 记进 manifest 的 execution.loads
    #                    推导缺省：站点观测 / 气候态 = resident，其余 = slice。
    #                    本配置 observation 是 cra → slice；reference 是
    #                    climatology → resident，**但只在设了 CRA_CLI_ROOT 时
    #                    才存在这个角色**（没参考源时这一项写了也不生效）。
    #   resume           True 时已完成块的状态落 output_dir/.states/，重跑跳过
    #
    # 下面 loads 两行就是推导结果，写出来是为了好改。代价：写死之后不再自动
    # 跟随数据源——换了观测/参考读取器要记得回来核对一遍。
    execution={
        "mode": "auto",         # 单块 → serial；块数上去 → threads（本流程是轻指标）
        "n_workers": 4,         # 线程缺省 4；落地 serial 时这项不起作用
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效
        "loads": {
            "observation": "slice",     # CRA40 GRIB2 按请求跨度切，现读现弃
            "reference": "resident",    # 气候态整 run 驻留（仅当设了 CRA_CLI_ROOT）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
