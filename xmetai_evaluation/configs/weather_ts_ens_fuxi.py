# -*- coding: utf-8 -*-
"""集合降水分类检验（FuXi 集合 × Diamond 站点）：一条命令跑两段。

口径由负责人给定，对标原版 ``run_categorical.py``（``--window-ts 24 --window-aroc 6``）：

    窗口 24  集合平均场   TS / POD / FAR / 频率偏差        -> weather_ts_ens
    窗口 6   逐成员        AROC / BS / BSS（要气候概率）    -> weather_ts_ens_prob

两段的窗口不同，而一条模板只有一个 ``window_hours``（它同时决定观测累积长度和
有效时效的筛选），所以配置里用列表声明两条模板，框架按顺序各跑一段、结果合并后
落盘一次：

    scores.csv                        两段的统一长表，靠 window_h 列区分（24 / 6）
    diagnostics/categorical_wide.csv  仅 24h 的 TS 宽表
    diagnostics/probability_wide.csv  仅 6h 的 AROC/BS/BSS 宽表
    manifest.json                     运行记录（segments 里有每段的窗口）

阈值不在这里配：两条模板自带的阈值常量（24h 的 [0.1,10,25,50,100]、
6h 的 [0.1,4,13,25]）已与原版输出逐格一致，且 6h 那组必须与参考文件的 4 列同集合。

数据：
    预报  {fuxi_ens_output}/YYYYMMDD/member_*/001.nc…（TP，逐 6h）
    观测  Diamond 站点降水（北京时，协议默认 +8 对齐）
    参考  气候概率目录 ref/MMDDHH.000（**必填**，见 BSS_REF_DEFAULT）
"""

import os

from xmetai_evaluation.configs.base import EvalConfig

#: BSS 的外部气候概率目录（``ref/MMDDHH.000``：站号 + 4 个超越式概率）。
#: 负责人给定的固定输入，对应原版的 ``--ref``；换目录直接改这一行。
BSS_REF_DEFAULT = os.environ.get("BSS_REF", "/workspace/data/worm/r0/ref")

cfg = EvalConfig(
    name="weather_ts_ens_fuxi",
    description="FuXi 集合降水评估：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS",
    # 两段：先 24h 的 TS，再 6h 的概率评分（顺序即执行顺序，结果合并落盘）
    pipeline=["weather_ts_ens", "weather_ts_ens_prob"],

    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fuxi_ens_output"
        ),
        "variable": "tp",
        "step_hours": 6.0,      # 布局步长：lead_from="index"，001.nc → +6h
        # 时效列表（小时）。None / 整个键不写 = 从目录索引推：取最大时效 lead_max，
        # 按 step_hours 从 0 铺起（0, 6, 12, …, lead_max）。
        #
        # ⚠ 本配置的**两段都带累积窗**（24h 段要 4 个连续 6h 步、6h 段要 1 步），
        # 而预热时效只从这张列表里取。要写就必须**按 6h 密排**，只写采样点会让
        # 窗凑不齐步数、一个样本都出不来：
        #     ✗ [24, 48, 72, …, 360]         24h 段每个窗只有 1 步
        #     ✓ [6, 12, 18, 24, 30, …, 360]  与推导结果同集合
        # 只读部分时效可显著降内存（如 [6, 12, 18, 24]）；注意 24h 段要求时效里
        # 有 24 的倍数，裁太短会让那一段一个样本都跑不出来。
        "lead_times": None,
    },
    observation_reader={
        "type": "station",
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ),
    },
    # BSS 的气候概率参考。必须给：缺了会降级成样本气候频率 r(1-r)，
    # 那算出来的 BSS 与原版不可比。
    reference_reader={"type": "ref_probability", "root_dir": BSS_REF_DEFAULT},

    start_date=os.environ.get("START_DATE", "20250101"),
    # 止于 15 号、不是 31 号：**站点观测档案到 2025-12-31 23:00 就没了**，预报比它长。
    # 起报越靠后，长时效的验证时刻越容易越过这条线，那种块整块评不出样本（日志里是
    # 「观测窗口不完整」+「没有任何样本可评」），只会白占一个失败块。15 号的最长时效
    # 落在 12-30，留一天余量。观测补齐后可改回来，`.states/` 里的旧块会原样复用。
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT", "/workspace/szwCode/xmetai-evaluate/evaluation_results/ts_multi_fuxi_ens"
    ),
    log_level="INFO",

    # 协议口径（station_valid_time）：8 = 北京时，Diamond 观测自带的口径，也是协议
    # 缺省。另可配 region（区域筛选）、weights（"cos_lat"）。两段共用这一份。
    options={"local_utc_offset_hours": 8},

    # ⚠ **不要在这里写 writers**。本配置是两段（weather_ts_ens + weather_ts_ens_prob），
    # 而配置级的 writers 是**一个列表同时作用到两段**——写了就会让 6h 概率那段也去
    # 出 categorical_wide（TS 宽表），而它真正该出的是 probability_wide。
    # 不写，各段才用各自模板自带的那份：24h 段 ["csv_long", "categorical_wide"]、
    # 6h 段 ["csv_long", "probability_wide"]。

    # ── 并发与数据加载：execution ──────────────────────────────────────
    # 可用键**就下面这 6 个**，全部是字面量，只覆盖写到的键（不写 = 用推导值）。
    # 写错键名会直接报配置错，不会静默忽略。
    #
    #   mode             ——**写死 processes**，两段都用进程。推导缺省给 24h 段的
    #                     形态是 threads（ts_score 属轻量指标），但 threads 形态下
    #                     多个线程会并发读 netCDF：HDF5 默认不是线程安全编译的，而
    #                     netCDF4-python 读数据时会放掉 GIL，于是几个线程真能同时
    #                     进 HDF5 的全局状态。实测症状是随机的
    #                     "NetCDF: HDF error"（打开成功、读数据失败），而它被协议层
    #                     的 except + continue 吞掉，只是**静默跳过那个起报**。
    #                     进程形态下每个子进程有独立的 HDF5 状态，从根上避开这个
    #                     race；resident 观测仍靠 fork 前的预热 + 写时复制共享，
    #                     不会按进程数翻倍。代价：每块的指标结果要 pickle 回父进程。
    #                       24h 段（weather_ts_ens）      进程 × CPU 核数
    #                       6h 段（weather_ts_ens_prob）  进程 × CPU 核数（本来就是）
    #   n_workers        worker 数。**本配置写死**，理由是核数在这台机器上太容易
    #                    读错：框架的缺省已改走 affinity mask（``sched_getaffinity``，
    #                    认 cgroup/cpuset 配额），但 Slurm 不开 task/affinity 插件时
    #                    连它也会报回宿主机的 192。上一轮就是这么炸的——192 个进程
    #                    各物化一块 51 成员的预报场，几分钟内全被 OOM 打死，症状是
    #                    满屏 BrokenProcessPool，离真正的原因很远。
    #                    写多少：**以 ``nproc`` 为准**。内存要跟着核数一起算——单块
    #                    内存的主项是预报场（成员数 × 单块时效 × 格点字节数），按进程
    #                    数线性叠加；站点观测只有 0.67 GB 且写时复制共享，不是大头。
    #   chunk_days       一个工作块装几个**起报日**（缺省 1）——管起报跨度
    #   lead_chunk_days  一个工作块装几天**时效**（缺省 1）——管时效跨度，是单块
    #                    内存的**主旋钮**。两段的窗口不同（24h / 6h），但切窗是按
    #                    **采样时效**切的，1 天（= 4 个 6h 时效）对两段都成立。
    #                    取值 0 = 不切（整段时效一块，集合预报会把整段 × 51 成员
    #                    全物化，约 25GB，非必要别用）
    #   loads            角色驻留覆盖 {角色: 策略}，角色只有 observation / reference：
    #                      slice      每个块现读现弃
    #                      resident   整个 run 读一次驻留内存
    #                      window:W   按 W 天块滚动缓存（LRU），省相邻块的重复读
    #                      window     裸写不带数字 = 自动定窗；落成的值回写成
    #                                 "window:N" 记进 manifest 的 execution.loads
    #                    本配置只写 observation：站点观测的推导缺省就是 resident
    #                    （Diamond 是逐时文本，按块现读等于把全年观测读几千遍）。
    #                    reference 是 ref_probability，逐样本直读、没有整包，
    #                    **写它会直接报配置错**，所以这一项必须留空。
    #   resume           True 时已完成块的状态落 output_dir/.states/，重跑跳过
    #
    # 下面 loads 一行就是推导结果，写出来是为了好改。代价：写死之后不再自动
    # 跟随数据源——换了观测读取器要记得回来核对一遍。
    execution={
        "mode": "processes",    # 两段都走进程：避开 threads 并发读 netCDF 的 HDF5 race
        # ⚠ 先跑 nproc 确认配额，再把这个数改成它（缺省虽已认 affinity mask，
        #   Slurm 不开 task/affinity 插件时仍会读成宿主机的 192）。
        "n_workers": 24,
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {"observation": "resident"},
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
