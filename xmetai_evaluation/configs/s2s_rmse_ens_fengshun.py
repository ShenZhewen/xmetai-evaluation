# -*- coding: utf-8 -*-
"""集合场检验（fengshun2.0 × CRA c87 0.25° zarr）：RMSE / CRPS / spread / 谱。

评的是 ``tools/fengshun20_infer_eval_job.yaml`` 先跑的推理产物：3 个起报
（20240901 / 20241001 / 20241101）、51 成员、60 步**日步长**，落盘
``{root}/YYYYMMDD/member_XXX/0NN.nc``，正好命中 ``fuxi_ens_phys`` 布局
（lead = 文件序号 × ``step_hours``，001.nc → +24h）。

**实况与推理输入是同一个盘** ``cra.1979_2024.c87.p25.h24.v1``（0.25°、87 通道、
每日 00Z、1979-01-01 … 2024-12-31）。起报日选这三天就是为此：每个 +60 步都落在盘内，
最后一个 20241101 + 1440h 正好是盘尾 2024-12-31——不越界、不换源、不插值，tp 也不换算。

与 ``weather_rmse_ens_tongyun_cra`` 的差别：

  * ``step_hours=24``，lead 24…1440h 共 60 个（日步长，不是 6h）。
  * 没有日气候态 ⇒ ``weather_ens_field_scores_noref``，不出 ACC / 活跃度。
  * **3 个起报**（每月 1 号、连续三个月）：样本量仍然只够看这条季节段的离散度，
    别按气候平均读。

单位：z500 报 m²/s²（不除 g）、msl 报 Pa、t2m 报 K、tp 报 **mm（~1h 量级）**。
注意盘里 tp 是 ``log1p(mm)``，读取器 ``CraZarrReader`` 会做 ``expm1``，所以这里拿到
的就是线性 mm；``CRA_ZARR_VARIABLES["tp"]`` 标的 ``inch`` 是 c71 盘的口径，对本盘
只是标注错误，**数值不需要任何 scale**（两盘同在 log1p、mean=0/std=1）。
"""

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 路径（字面量，直接改这里）──
FORECAST_ROOT = "/workspace/data/shenzw/fengshun_output"
CRA_STORE = "/workspace/data/zarr_refresh_v3/cra.1979_2024.c87.p25.h24.v1"
OUTPUT_DIR = "/workspace/data/shenzw/evaluation_results/weather_rmse_ens_fengshun"

# ── 要素表 ──
# 与推理落盘的 vars 取交集里要评的那几个（落盘还有 q700/t700/t850/u850/v850/u10m/v10m）。
VARS = ["z500", "msl", "t2m", "tp"]

# ── 逐变量指标表 ──
# 没有气候态，所以只有这五个指标可用；谱/球谐带只给 z500（与其他集合档同口径）。
# tp 不出谱：降水场不平稳，zonal_spectrum 对它没有约定俗成的读法。
VAR_METRICS = {
    "z500": ["rmse", "crps", "spread_error", "zonal_spectrum", "spherical_bands"],
    "msl": ["rmse", "crps", "spread_error"],
    "t2m": ["rmse", "crps", "spread_error"],
    "tp": ["rmse", "crps", "spread_error"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# max_wavenumber=720（全球 0.25° Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}
# ddof=0（除 N，对齐 ref_result 的 spread_*.csv）
METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

# ── 时效 ──
# 60 步 × 24h ⇒ 24…1440h，最远有效时刻 2024-12-31 00Z = 实况盘的最后一天。
LEADS = list(range(24, 1441, 24))

cfg = EvalConfig(
    name="s2s_rmse_ens_fengshun",
    description=(
        "集合场检验（fengshun2.0 × CRA c87 0.25° zarr）："
        "3 个起报（20240901/20241001/20241101）、51 成员、24…1440h 日步长，"
        "z500/msl/t2m/tp 的 RMSE / CRPS / spread / 谱（不做 ACC / 活跃度）"
    ),
    pipeline="weather_ens_field_scores_noref",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": FORECAST_ROOT,
        "variables": VARS,
        "step_hours": 24.0,
        "lead_times": LEADS,
    },
    observation_reader={
        "type": "cra_zarr",
        "path": CRA_STORE,
        "variables": VARS,
    },
    # 不给 reference_reader：本流程的指标都不需要气候态

    # ── 时段 ──
    # 三个起报日，和推理 config 的 times 一一对应（每个 +60 步都在实况盘内）。
    periods=[
        ("20240901", "20240901"),
        ("20241001", "20241001"),
        ("20241101", "20241101"),
    ],
    limit=None,  # 限起报数：冒烟改 1；None = 不限（本档 3 个）

    output_dir=OUTPUT_DIR,
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # ── 输出视图 ──
    writers=["csv_long", "spectrum"],

    # ── 协议参数 ──
    options={
        "sample_by": "init_lead",
        "regions": {},
    },

    # ── 执行 ──
    # 1 个起报 × 60 个时效窗（lead_chunk_days=1 ⇒ 每块 1 个 lead），够 18 路并发；
    # 内存吃的是 51 成员 × 4 变量 × 721×1440。
    execution={
        "mode": "processes",
        "n_workers": 18,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "window:16",
        },
        "resume": True,
    },
)
