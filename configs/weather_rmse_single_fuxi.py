"""Single FuXi deterministic evaluation config.

Migrated from scripts/single_fuxi.sh.
"""
from pathlib import Path

CONFIG = {
    "type": "batch",
    "label": "FuXi",
    "output_name": "weather_rmse_single_fuxi",
    "pred_root": "/workspace/data/shenzw/fuxi_single_output",
    "target_zarr": [
        "/workspace/data/liujunjie/era5_foundation_store2/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "/workspace/data/liujunjie/era5_foundation_store2/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ],
    # 只被打印，不决定落盘（真实归档 outputs/results/<output_name>/<配置指纹>/，
    # 见 core/batch_adapter.py:290 的 outdir_root=str(work_root)）。
    "outdir_root": "/workspace/szwCode/xmetai-eval_pro/outputs",
    "periods": [
        # 20251217 起预报数据缺失/无效，止于 20251216，单段跑满 349 天。
        # **不要**再拆回 (0101-0630)+(0701-1216) 两段：regr_ens 的收尾步骤只按
        # **当次 CLI 的 --dates** 重写 summary.csv / batch_meta.json（见 vfc/regr_ens.py
        # 的 done_dates 取 _all_candidates，不是扫描 outdir_root），后一段会把前一段
        # 整个盖掉——归档里躺着 349 个日期目录，summary 只剩 169 行。
        # 合并不改任何逐日结果，只改这份日期清单。
        ("20250101", "20251216"),
    ],
    "metrics": ["rmse", "spectrum", "acc", "fa"],
    "variables": [
        "z500", "q2m", "q700", "t700", "t2m", "t850", "msl",
        "u200", "v200", "u850", "v850", "u10m", "v10m",
        "ws10m", "ws850", "ws200",
    ],
    "var_metrics": {
        "z500": ["rmse", "spectrum", "acc", "fa"],
        "q2m": ["rmse"],
        "q700": ["rmse"],
        "t700": ["rmse", "spectrum"],
        "t2m": ["rmse", "spectrum"],
        "t850": ["rmse"],
        "msl": ["rmse", "spectrum"],
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
    "pred_q_scale": 0.001,  # FuXi outputs q in g/kg, target is kg/kg
    "n_workers": 48,
    "worker_fallback": [48, 36, 24, 12, 4, 2],
    "resume": True,
    "summarize_mode": "--summarize-det",
    "env_overrides": {
        "VFC_CLIMO_ROLLING": "1",
        "VFC_SINGLE_STREAM": "1",
        "VFC_OBS_BLOCK": "1",
        "VFC_DATES_PER_CHILD": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    },
}
