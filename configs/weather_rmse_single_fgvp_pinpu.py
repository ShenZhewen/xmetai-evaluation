"""Single FGVP (pinpu 输出版本) deterministic batch config.

Copy of weather_rmse_single_fgvp.py，仅换 pred_root/outdir/输出名，
变量、指标、q 单位（g/kg → pred_q_scale 0.001）与原 config 完全一致。
"""
from pathlib import Path

CONFIG = {
    "type": "batch",
    "label": "fgvp_pinpu",
    "output_name": "weather_rmse_single_fgvp_pinpu",
    "pred_root": "/workspace/data/shenzw/fgvp_output_pinpu",
    "target_zarr": [
        "/workspace/data/liujunjie/era5_foundation_store2/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "/workspace/data/liujunjie/era5_foundation_store2/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ],
    # 只被打印，不决定落盘（真实归档 outputs/results/<output_name>/<配置指纹>/，
    # 见 core/batch_adapter.py:290 的 outdir_root=str(work_root)）。
    "outdir_root": "/workspace/szwCode/xmetai-eval_pro/outputs",
    "periods": [
        # 20251217 起预报数据缺失/无效，止于 20251216。本模型 pred_root 只到 20251215，
        # 实际落在 348 天（其余模型 349 天）——`--dates` 会与 pred_root 取交集，无害。
        # **不要**再拆回 (0101-0630)+(0701-1216) 两段：regr_ens 的收尾步骤只按
        # **当次 CLI 的 --dates** 重写 summary.csv / batch_meta.json（见 vfc/regr_ens.py
        # 的 done_dates 取 _all_candidates，不是扫描 outdir_root），后一段会把前一段
        # 整个盖掉——归档里躺着 349 个日期目录，summary 只剩 169 行。
        # 合并不改任何逐日结果，只改这份日期清单。
        ("20250101", "20251216"),
    ],
    "metrics": ["rmse", "spectrum", "acc", "fa"],
    "variables": [
        "z500", "q700", "t700", "t2m", "t850", "msl", "d2m",
        "u200", "v200", "u850", "v850", "u10m", "v10m",
        "ws10m", "ws850", "ws200",
    ],
    "var_metrics": {
        "z500": ["rmse", "spectrum", "acc", "fa"],
        "q700": ["rmse"],
        "t700": ["rmse", "spectrum"],
        "t2m": ["rmse", "spectrum"],
        "t850": ["rmse"],
        "msl": ["rmse", "spectrum"],
        "d2m": ["rmse"],
        "u200": ["rmse", "fa", "spectrum"],
        "v200": ["rmse", "fa", "spectrum"],
        "u850": ["rmse", "fa", "spectrum"],
        "v850": ["rmse", "fa", "spectrum"],
        "u10m": ["rmse", "fa"],
        "v10m": ["rmse", "fa"],
        "ws10m": ["rmse"],
        "ws850": ["fa", "rmse", "spectrum"],
        "ws200": ["rmse", "spectrum"],
    },
    "climo": "/workspace/data/worm/era5_clim_phys_14.nc",
    "pred_q_scale": 0.001,  # FGVP outputs q in g/kg, target is kg/kg
    # 2026-09-15 日志实锤：普通日期单 worker <10G，48 并发吞吐最高；
    # 中间档撞坏日期一样崩，fallback 留 4 作纯保底。
    "n_workers": 48,
    "worker_fallback": [48, 4, 2],
    "resume": True,
    "summarize_mode": "--summarize-det",
    "env_overrides": {
        "VFC_DATES_PER_CHILD": "1",
        "VFC_CLIMO_ROLLING": "1",
        "VFC_SINGLE_STREAM": "1",
        "VFC_OBS_BLOCK": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    },
}
