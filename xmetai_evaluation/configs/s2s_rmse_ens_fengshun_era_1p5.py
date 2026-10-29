# -*- coding: utf-8 -*-
"""集合场检验：风顺 ERA 1.5°（原版）× s2s c76 1.5° zarr：RMSE / CRPS / spread / 谱。

评的是 ``tools/s2s_exp_job.yaml``（推理 → 评测一条龙）先跑的推理产物：

    /workspace/data/shenzw/fengshun_era_1p5/20240901/member_XXX/0NN.nc

``001.nc`` → +24h（日步长，``step_hours=24``），60 步到 +1440h = 2024-10-31，
实况与推理输入是**同一个盘** ``s2s.19500101_20260417.c76.1p5.h24.v1``
（1.5°、每日 00Z，与预报同网格；读取器按名取 z500 并反归一化）。

**本档是最小可比集**（与 SPA 实验同口径，定锚点用，不出版面）：

  * 1 个起报（20240901）——单起报只能筛「有没有崩」，臂间判读要等多起报；
  * 只评 z500（推理侧 ``vars="z500"``，落盘也只有它）；
  * 指标保留 rmse / crps / spread_error / 谱，谱与球谐带是集合离散度的必看项。

口径：z500 报 m²/s²（不除 g）、不配气候态（``weather_ens_field_scores_noref``，
无 ACC / 活跃度）、spread 用 ddof=0。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 路径（字面量，直接改这里）──
FORECAST_ROOT = "/workspace/data/shenzw/fengshun_era_1p5"
# 实况用推理的同一个盘（自洽评测）；只评 z500，tp 的 expm1 口径不涉及。
TRUTH_STORE = "/workspace/data/zarr_refresh_v3/s2s.19500101_20260417.c76.1p5.h24.v1"
OUTPUT_DIR = "/workspace/data/shenzw/evaluation_results/weather_rmse_ens_fengshun_era_1p5"

# ── 要素表：本批只评 z500 ──
VARS = ["z500"]

VAR_METRICS = {
    "z500": ["rmse", "crps", "spread_error", "zonal_spectrum", "spherical_bands"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# max_wavenumber=120（全球 1.5° Nyquist = 240/2）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 120,
}
# ddof=0（除 N，对齐 ref_result 的 spread_*.csv）
METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

# ── 时效 ──
LEADS = list(range(24, 1441, 24))

cfg = EvalConfig(
    name="s2s_rmse_ens_fengshun_era_1p5",
    description=(
        "集合场检验（风顺 ERA 1.5° 原版 × s2s c76 1.5° zarr）："
        "1 个起报 20240901、51 成员、24…1440h 日步长，"
        "只评 z500 的 RMSE / CRPS / spread / 谱"
    ),
    pipeline="weather_ens_field_scores_noref",

    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": FORECAST_ROOT,
        "variables": VARS,
        "step_hours": 24.0,
        "lead_times": LEADS,
    },
    observation_reader={
        "type": "cra_zarr",
        "path": TRUTH_STORE,
        "variables": VARS,
    },

    periods=[("20240901", "20240901")],
    limit=None,

    output_dir=OUTPUT_DIR,
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    writers=["csv_long", "spectrum"],

    options={
        "sample_by": "init_lead",
        "regions": {},
    },

    execution={
        "mode": "processes",
        "n_workers": 12,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "window:16",
        },
        "resume": True,
    },
)
