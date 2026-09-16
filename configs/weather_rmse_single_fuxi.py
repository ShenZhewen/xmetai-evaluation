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
    "outdir_root": "/workspace/_XMETAI_test_results/single_fuxi",
    "periods": [
        ("20250101", "20250630"),
        # 20251217 起预报数据缺失/无效，不参与评测。
        ("20250701", "20251216"),
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
    "resume_cache": True,
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
