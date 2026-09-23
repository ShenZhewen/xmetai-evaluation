# -*- coding: utf-8 -*-
"""台风路径与强度检验（FuXi 集合 × BABJ 报文）。

流程：``weather_typhoon_ens``（从实况位置**逐成员**链式诊断预报中心，再按两种
集合口径聚合成一条误差曲线）。本文件只声明"数据从哪来、评哪段时间、写到哪"，
算法口径全在流程模板里（``pipeline/pipelines.py`` 的 ``weather_typhoon_ens``）。

与 ``weather_typhoon_single_fuxi.py``（确定性）的唯一区别是数据源与口径：
预报换成带成员轴的集合目录，指标换成 ``track_error_ens``。

数据：
    预报  {root}/YYYYMMDD/member_NNN/001.nc…（msl，Pa；u10m/v10m，m/s）
    观测  BABJ 路径报文 babj<编号>.dat（diamond7，GBK，北京时）
    参考  无（路径误差不吃气候态）

**预报目录的层级不能少一层**：根目录下按起报分日期目录，日期目录下再放
``member_*/``。指到确定性目录（没有 ``member_`` 子目录）会在协议里直接报错，
不会静默当成单成员跑。

两种集合口径**都会算**（旧链路 ``core/tc_ref.py`` 的 ``ens_agg`` 是个二选一的
开关，这里不做开关）：

    B（曲线主口径）各成员位置先平均成集合平均位置，再与实况求误差
    A（*_a 三列）  各成员先各算误差，再对成员平均

两者只在路径三项（track_err / at / ct）上不同——三角不等式使然；强度类两项
（气压、风速）代数恒等：``mean_m(pmin_m − obs) ≡ mean_m(pmin_m) − obs``，
实况对成员是常数。

规模：一个 (台风, 起报) 一个场次，场次内**逐个成员串行**跑完整的确定性链。
时效 6…360h 共 60 个，整段必须在同一个工作块里（链式诊断的硬要求，见下面
execution）。每个工作块的耗时 ≈ 确定性链 × 成员数，内存则与确定性链相同
（一次只驻留一个成员的场）。

先冒烟再正式：``limit`` 改成 1、``storm_ids`` 填一号台风跑一遍，再改回去。
**冒烟用单独的 ``output_dir``**——``chunk_id`` 只由起报日与时效窗拼成，既不含
``limit`` 也不含 ``storm_ids``，而 ``resume`` 只看 ``chunk_id`` 和"这块成功过"
就复用，不比对配置。同一个 ``output_dir`` 下冒烟再正式跑，冒烟那块会被原样
复用：只跑了 2501 的那一天，其余台风静默消失。换个目录就没事。
"""
from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="weather_typhoon_ens_fuxi",
    description="台风路径与强度检验（FuXi 集合 × BABJ 报文）",
    pipeline="weather_typhoon_ens",

    # ── 数据源 ──────────────────────────────────────────────────────
    # 路径可以改，换机器/换目录直接改这两个默认值。
    forecast_reader={
        "type": "fuxi_ens_phys",        # 带 _phys = xu 复刻口径；member_* 子目录即成员
        "root_dir": "/workspace/data/shenzw/fuxi_ens_output",
        # 定中心要 msl，定强度要 u10m/v10m。给不出风速通道不报错，只是强度项
        # （fcst_vmax_ms / wind_err_ms）整列留空。
        "variables": ["msl", "u10m", "v10m"],
        "step_hours": 6.0,              # 布局步长：lead_from="index"，001.nc → +6h
        # 时效列表（小时）。**不写这一行 = 从目录索引推**：0, 6, …, 目录里的
        # 最大时效。台风这条链**不能写稀疏集**——链式诊断逐时效推进，搜索框
        # 中心是上一时效诊断出的位置，中间少一个时效链就断了。要缩短范围就写
        # 稠密前缀（如只到 240h：[6, 12, …, 240]），别写 [24, 48, 72, …]。
        "lead_times": None,
    },
    observation_reader={
        # BABJ 报文目录，一条报文 = 一号台风的整条路径（跨越多天）。整目录
        # 一次读完（KB 级），不做时间筛选。也可以直接指到单个 babj<编号>.dat。
        "type": "babj",
        "root_dir": "/workspace/data/LiuJs/babj",
    },
    # 参考集：无。这条流程是"预报路径 vs 实况路径"的纯配对误差，没有距平项。

    # ── 协议参数 ────────────────────────────────────────────────────
    options={
        # 搜索框：early_hours 之内用大框（防起报初期中心漂移带偏整条链），
        # 之后收小。单位是**经纬度**不是公里，数字直接改这里。
        # 与确定性链共用同一组参数——两条链的链式诊断是同一段代码。
        "center_half_deg": 3.0,
        "early_half_deg": 4.0,
        "early_hours": 12.0,
        # 强度框：以**当前诊断中心**为中心的最大 10m 风速（sqrt(u²+v²) 现合成）
        "intensity_half_deg": 5.0,
        "wind_variables": ["u10m", "v10m"],
        # 链的起始中心（种子）取**起报后第一条不早于「起报 + 这个值」的实况**，
        # 一条都没有就跳过该场次。单位小时，直接改这里。**每个成员共用同一个
        # 种子**——种子只由报文和起报时刻决定，与成员无关。
        #
        # 6.0 = 第一个时效的步长 = 标准归档（tc/fuxi）的口径，别乱改：那边的
        # init_pos 实测恒等于"起报后第一条不早于 init+6h 的实况"。改成 0 会让
        # 绝大多数场次的种子前移 6 小时，台风刚生成/快消散那几天 msl 场平缓，
        # 会静默多出全空场次、少掉有效场次。
        "seed_min_offset_hours": 6.0,
        # 只跑某几号台风就填编号列表；不写 = 报文目录下全部。
        # 冒烟时打开： "storm_ids": ["2501"],
    },

    # 时段只管**起报日**，这里给的是 2025 全年。两头都取交集，写宽了不会凭空
    # 多出场次也不会报错；空档期的起报日会在切块前就被剔掉。
    start_date="20250101",
    end_date="20251231",
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir="/workspace/szwCode/evaluation_results/weather_typhoon_ens_fuxi",
    log_level="INFO",

    # 输出视图：
    #   csv_long        统一长表 scores.csv（每个场次 8 行：路径/沿/横/风速/气压
    #                   五项取方案 B，路径三项另出方案 A 的 *_a 行）
    #   typhoon_cases   typhoon/tc<编号>_<起报>.csv 逐时效 15+5 列 + _meta.json
    #                   + typhoon/typhoon.csv 全场合拼
    writers=["csv_long", "typhoon_cases"],

    # ── 并发与数据加载：execution ────────────────────────────────────
    # 可用键只有 mode / n_workers / chunk_days / lead_chunk_days / loads / resume。
    #
    # lead_chunk_days 必须是 0：一块 = 一个起报日组 × 一个时效窗，时效被切开后
    # 后一块的搜索起点接不上前一块诊断出的中心（只能退回实况位置重起），结果
    # 会静默偏掉。协议在 prepare 里显式校验。
    #
    # 并行粒度是工作块 = 起报日；**成员之间是串行的**，在协议内部一个接一个读
    # 完就跑。所以这里每一块的内存与确定性链相同（单成员一个起报 ≈0.75G），
    # 但耗时是确定性链的成员数倍。要点：想跑快点加 n_workers（内存够的话），
    # 别去动成员——把成员并进工作块要改 execution 的分块与合并，是另一件事。
    #
    # 观测的 BABJ 只能是 slice——resident/window 会按请求跨度切报文的时间轴，
    # 而报文是北京时、跨度是台风的整条生命史，按起报时刻切会把实况切没。
    # 预报这一项不用写、也不能写：预报恒为 slice，且 loads 只认 observation /
    # reference，写 "forecast" 会被判成未知角色报错。
    execution={
        "mode": "processes",
        "n_workers": 8,         # 8 × 0.75G ≈ 6G 工作集；内存够可以往上加
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 0,   # 不切时效（链式诊断的硬要求，改成非 0 会报错）
        "loads": {"observation": "slice"},
        "resume": True,         # 中断后接着算
    },
)
