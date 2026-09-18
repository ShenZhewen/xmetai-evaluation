"""Ensemble FuXi evaluation config.

Migrated from scripts/ensemble_fuxi.sh.
"""
from pathlib import Path

CONFIG = {
    "type": "batch",
    "label": "FuXi-ens",
    "output_name": "weather_rmse_ens_fuxi",
    "pred_root": "/workspace/data/shenzw/fuxi_ens_output",
    "target_zarr": [
        "/workspace/data/liujunjie/era5_foundation_store2/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "/workspace/data/liujunjie/era5_foundation_store2/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ],
    # 只被打印，不决定落盘位置（真实归档见 outputs/results/<output_name>/<指纹>/）。
    "outdir_root": "/workspace/szwCode/xmetai-eval_pro/outputs",
    "periods": [
        # 单段整段跑完：summarize 按当前 periods 覆写 summary，拆两段会把前一段
        # 挤掉（正是上次 349 目录只有 169 行的原因）。20251217 起预报数据缺失/
        # 无效，右端收到 20251216。
        ("20250101", "20251216"),
    ],
    "metrics": ["rmse", "crps", "acc", "fa", "spectrum"],
    "variables": ["z500", "msl", "u200", "v200", "ws200"],
    "var_metrics": {
        "z500": ["rmse", "crps", "acc", "fa", "spectrum"],
        "msl": ["rmse", "acc"],
        "u200": ["rmse", "fa", "spectrum"],
        "v200": ["rmse", "fa", "spectrum"],
        "ws200": ["rmse", "fa", "spectrum"],
    },
    "climo": "/workspace/data/worm/era5_clim_phys_14.nc",
    "n_workers": 48,
    "worker_fallback": [48, 36, 24, 12, 4, 2],
    "resume": True,
    "summarize_mode": "--summarize-ens",
    "env_overrides": {
        "VFC_ENS_ENSMEAN_ONLY": "1",
        "VFC_ENS_BLOCK": "4",
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
