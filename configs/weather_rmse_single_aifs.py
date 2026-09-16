"""Single AIFS deterministic batch config.

**AIFS 和其它模型走的不是同一条流水线**——这一点和其它五个 config 不同，别照抄：

- 其它模型的观测是共用的 era5 zarr，走 `run_batch_rmse.py --pred-root + --target-zarr`；
- AIFS 的观测是**目录式**的 `aifs-single-target/<日期>/`，由 `vfc/regr_ens.py` 的
  `run_aifs_batch*` 读 `AifsForecast`（见 `:1807` / `:1930`），只认
  `--aifs-pred-root` / `--aifs-target-root` 两个专属开关，**没有 target_zarr**。

所以本配置：**不写 `target_zarr`**，改写 `aifs_target_root`；`runner.py` 见它非空就
切命令行（`core/batch_adapter.py:run_batch_job`）。`pred_root` 与归档里的
`aifs_pred_root` 同名同值，不需要再写一份。

参数（变量清单、var_metrics、climo、n_workers=36）全部照
`outputs/results/weather_rmse_single_aifs/single_aifs/batch_meta.json` 的 `options`
回填，即 2026-09 当年实际跑的那一套，好让重跑能就地续上已有日期目录。

背景：`README.md:228` 记着「原 run_all.sh 的 6 个模型只迁了 4 个，single_aifs 未迁」，
本文件补的就是这个缺口。产物契约与通用路径一致（`summary.csv` + `batch_meta.json`
+ `<YYYYMMDD>/` + `<日期>_meta.json`），所以报告侧和 `--summarize-det` 都能直接吃。
"""
from pathlib import Path

CONFIG = {
    "type": "batch",
    "label": "aifs",
    "output_name": "weather_rmse_single_aifs",
    "pred_root": "/workspace/data/fanpy/aifs-single-output",
    # AIFS 专属：目录式观测根，写在这里就走 AIFS 那条命令行。
    "aifs_target_root": "/workspace/data/fanpy/aifs-single-target",
    "outdir_root": "/workspace/_XMETAI_test_results/single_aifs",
    "periods": [
        # 20251217 起预报数据缺失/无效，止于 20251216，单段跑满 349 天。
        # **不要**再拆回 (0101-0630)+(0701-1216) 两段：regr_ens 的收尾步骤只按
        # **当次 CLI 的 --dates** 重写 summary.csv / batch_meta.json（AIFS 并行路径
        # 也一样，见 vfc/regr_ens.py:2009 的 done_dates 取 _all_candidates），
        # 后一段会把前一段整个盖掉——归档里躺着 349 个日期目录，summary 只剩 169 行。
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
        "ws850": ["rmse", "fa", "spectrum"],
        "ws200": ["rmse", "spectrum"],
    },
    "climo": "/workspace/data/worm/era5_clim_phys_14.nc",
    # pred_q_scale 故意不写：默认 1.0，与归档 options 里的 pred_q_scale=1.0 一致。
    # 判据（Pangu/FengQing 都栽过，见 weather_rmse_single_pangu.py 的注释）：
    # raw q 值域 ~1-10 = g/kg（要写 0.001）；~0.001-0.01 = kg/kg（不写）。
    # n_workers 沿用当年实测的 36（AIFS 逐日读目录式观测，比通用路径重），
    # 换别的值不影响结果，但会让指纹变、续不上已有日期目录。
    "n_workers": 36,
    "worker_fallback": [36, 4, 2],
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
