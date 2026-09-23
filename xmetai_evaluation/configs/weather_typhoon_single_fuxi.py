# -*- coding: utf-8 -*-
"""台风路径与强度检验（FuXi 确定性 × BABJ 报文）。

流程：``weather_typhoon_det``（从实况位置链式诊断预报中心 + 对 BABJ 分析场
配对求路径/强度误差）。本文件只声明"数据从哪来、评哪段时间、写到哪"，算法
口径全在流程模板里（``pipeline/pipelines.py`` 的 ``weather_typhoon_det``）。

数据：
    预报  {root}/YYYYMMDD/001.nc…（msl，Pa；u10m/v10m，m/s；布局见 layouts.py）
    观测  BABJ 路径报文 babj<编号>.dat（diamond7，GBK，北京时）
    参考  无（路径误差不吃气候态）

预报侧不需要专门的台风 reader——``fuxi_phys`` 已经覆盖 msl/u10m/v10m
（大小写不敏感、单位一致）。新的只有观测侧的 ``babj``。

规模：一个 (台风, 起报) 一个场次。时效 6…360h 共 60 个，**整段必须在同一个
工作块里**（链式诊断的硬要求，见下面 execution）。场次数取决于时段里有几天
有台风——2025 全年约 150+ 个场次，每个起报日一个块。

先冒烟再正式：``limit`` 改成 1、``options`` 里打开 ``"storm_ids": ["2501"]``
跑一遍，再改回去。``chunk_id`` 不含 ``limit``，``resume`` 打开时冒烟算过的
块会被原样复用，不会白算。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="weather_typhoon_single_fuxi",
    description="台风路径与强度检验（FuXi 确定性 × BABJ 报文）",
    pipeline="weather_typhoon_det",

    # ── 数据源 ──────────────────────────────────────────────────────
    # 路径可以改，换机器/换目录直接改这两个默认值。
    forecast_reader={
        "type": "fuxi_phys",            # 带 _phys = xu 复刻口径，要素表全、msl 报 Pa
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
        ),
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
        # 一次读完（KB 级），不做时间筛选——"一个文件一个时刻"那套选法在这里
        # 不成立。也可以直接指到单个 babj<编号>.dat。
        "type": "babj",
        "root_dir": os.environ.get("BABJ_DIR", "/workspace/data/LiuJs/babj"),
    },
    # 参考集：无。这条流程是"预报路径 vs 实况路径"的纯配对误差，没有距平项。

    # ── 协议参数 ────────────────────────────────────────────────────
    options={
        # 搜索框：early_hours 之内用大框（防起报初期中心漂移带偏整条链），
        # 之后收小。单位是**经纬度**不是公里，数字直接改这里。
        "center_half_deg": 3.0,
        "early_half_deg": 4.0,
        "early_hours": 12.0,
        # 强度框：以**当前诊断中心**为中心的最大 10m 风速（sqrt(u²+v²) 现合成）
        "intensity_half_deg": 5.0,
        "wind_variables": ["u10m", "v10m"],
        # 链的起始中心（种子）取**起报后第一条不早于「起报 + 这个值」的实况**，
        # 一条都没有就跳过该场次。单位小时，直接改这里。
        #
        # 6.0 = 第一个时效的步长 = **标准归档（tc/fuxi）的口径**，别乱改：那边
        # 的 init_pos 实测恒等于"起报后第一条不早于 init+6h 的实况"。改成 0
        # （种子取起报时刻那条）会让绝大多数场次的种子前移 6 小时（中位 ~100km），
        # 多半被 ±4° 的早期搜索框吸收，但台风刚生成/快消散那几天 msl 场平缓，
        # 会静默多出全空场次、少掉有效场次——上一版就是这么差出 4 个场次的。
        #
        # 台风当天的第一条定位常在 20:00（12 UTC），所以生成首日的种子会是
        # init+12h，场次 meta 里的 seed 字段能看出来。
        "seed_min_offset_hours": 6.0,
        # 只跑某几号台风就填编号列表；不写 = 报文目录下全部。
        # 冒烟时打开： "storm_ids": ["2501"],
    },

    # 时段只管**起报日**，这里给的是 2025 全年。
    #
    # 两头都取交集，写宽了不会凭空多出场次也不会报错：起报时刻由预报目录的
    # 索引来（目录里没有的日子就是没有），再由 BABJ 报文反推筛一道（台风生成
    # 前/消散后对不上分析场的日子）。空档期的起报日会在切块前就被剔掉，不会
    # 变成 manifest 里的失败块。
    #
    # 要回到和新链路逐位对拍的那一小段（旧归档 weather_typhoon_single_fuxi 的
    # 38 个场次、2501–2508），把这两行改回 20250610 / 20250801。
    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251231"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/evaluation_results/weather_typhoon_single_fuxi",
    ),
    log_level="INFO",

    # 输出视图：
    #   csv_long        统一长表 scores.csv（每个场次 5 行：路径/沿/横/风速/气压）
    #   typhoon_cases   typhoon/tc<编号>_<起报>.csv 逐时效 15 列 + _meta.json
    #                   + typhoon/typhoon.csv 全场合拼（渲染器与人工核对用）
    writers=["csv_long", "typhoon_cases"],

    # ── 并发与数据加载：execution ────────────────────────────────────
    # 可用键只有 mode / n_workers / chunk_days / lead_chunk_days / loads / resume，
    # 写错键名会直接报配置错。
    #
    # lead_chunk_days 必须是 0：一块 = 一个起报日组 × 一个时效窗，时效被切开后
    # 后一块的搜索起点接不上前一块诊断出的中心（只能退回实况位置重起），结果
    # 会静默偏掉。协议在 prepare 里显式校验，切了直接报错，不会出假数。
    #
    # 并行粒度就是工作块 = 起报日（时效维只有一个窗）。每个 (台风, 起报) 场次
    # 完整落在一个块里，块之间零共享，所以 threads / processes 都安全。
    # 这里选 processes 而不是 threads：**这套环境（netCDF4/HDF5）不是线程安全
    # 的**，仓里 io/base.py 的 hdf5_guard 已经把并发读串行化，threads 下读盘
    # 照样排队，只剩 numpy 计算能并行——而这条链的瓶颈恰恰是读盘（每块要把
    # 60 时效 × 3 通道的全球场读进来），不是那 60 次 nanargmin。
    #
    # n_workers 的约束是内存不是核数：每块要驻留
    #     起报数/天 × 60 时效 × 3 通道 × 721×1440 × 4B ≈ 0.75G（float32）
    # 1 个起报/天就是 0.75G，24 进程要 ~18G。按机器内存调这个数。
    execution={
        "mode": "processes",
        "n_workers": 8,         # 8 × 0.75G ≈ 6G 工作集；内存够可以往上加
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 0,   # 不切时效（链式诊断的硬要求，改成非 0 会报错）
        # 观测的 BABJ 只能是 slice——resident/window 会按请求跨度切报文的时间轴，
        # 而报文是北京时、跨度是台风的整条生命史，按起报时刻切会把实况切没。
        # 显式写出来（值跟推导出的默认一样，纯属把口径摆在配置里，别再靠默认）；
        # 计划层也会校验，配成 resident / window 直接报错。
        # 预报这一项不用写、也不能写：预报恒为 slice，且 loads 只认
        # observation / reference，写 "forecast" 会被判成未知角色报错。
        "loads": {"observation": "slice"},
        "resume": True,         # 中断后接着算
    },
)
