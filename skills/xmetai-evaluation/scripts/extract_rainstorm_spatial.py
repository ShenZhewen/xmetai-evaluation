#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""取暴雨个例的空间 tp 场：把某几个起报日的 Tongyun 预报/ERA5 实况 24h 累计 tp
聚合到单个目录，供下载后画空间分布图。

地址全部走命令行参数（含默认值），后期换过程/模型/ERA store 只改参数不用改代码——
裁剪**不放这里**，本脚本只负责把「某起报日的 24h 累计 tp」全全球算好写出，
中国框裁剪由画图代码（``visualization/rainstorm_spatial.py``）在做。

用法（在服务器上跑）：
    python3 skills/xmetai-evaluation/scripts/extract_rainstorm_spatial.py
    python3 skills/xmetai-evaluation/scripts/extract_rainstorm_spatial.py \
        --init-days 20250526,20250527,20250528,20250529 \
        --forecast-root /workspace/data/shenzw/tongyun_single \
        --era-store .../era5_sfc_2025.01-2026.07.c15.p25.h6.zarr \
        --steps 4 \
        --out /workspace/data/shenzw/rainstorm_202510_spatial

产物（单位统一到 mm，按``--out``下的相对名）：
    pred_<YYYYMMDD>.nc   起报日 <YYYYMMDD> 的 24h 累计 tp（取 00{1..steps}.nc 求和）
    era_<YYYYMMDD>.nc    同 24h 实况（ERA5 6h 步累积求和，×1000 转 mm）
    manifest.json        起报日 -> (pred, era) 相对路径 + 各日 China 总量，供画图读

口径：起报日 d 的 24h 窗 = [d 00Z, d+1 00Z]（= 08:00–08:00 北京时）。
  预报 = ``{forecast_root}/{d}/00{n}.nc``(n=1..steps) 的 TP 求和；
  ERA  = ``{era_store}`` 里 tp 通道，取时刻 [6,12,...,6*steps]Z 诸值求和。
        注意 store 的 time step 虽是 6h，tp 存的却是**1h 累积**，每个值要先
        ×6 才是一个 6h 步（与推理侧 era5 loader 的 ``_ACCUMULATED_1H`` 同口径）。

数据源与单位（2026-10-27 读取实测确认）：
  Tongyun 预报 nc：北→南 lat 90→-90、lon 0→360，TP 单位 mm，每文件一个 6h 步。
  ERA sfc zarr：北→南 lat 90→-90、lon -180→180，tp 通道单位 m，time step 6h，
                **值是 1h 累积**（全球均值 8.7e-5 m，按 1h 折 2.1 mm/day 才对）。
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr


def _mask_missing(arr, missing=3.4e38):
    """把 sfc zarr 的 GRIB missing 值填 NaN。"""
    return np.where(np.abs(arr) > missing * 0.5, np.nan, arr).astype("float32")


def load_forecast_24h(forecast_root, init_day, steps):
    """读 steps 个 6h 步 TP 求和 → 24h(mm)，返回 (z_mm, lat, lon)。"""
    stacked = None
    lat = lon = None
    for n in range(1, steps + 1):
        ds = xr.open_dataset(f"{forecast_root}/{init_day}/{n:03d}.nc")
        if lat is None:
            lat = ds["lat"].values
            lon = ds["lon"].values
        v = ds["TP"].values.astype("float64")
        stacked = v if stacked is None else stacked + v
        ds.close()
    return stacked, lat, lon


def load_era_24h(era_store, chans, init_day, steps):
    """读 24h 窗 steps×6h 累积求和 → mm，返回 (z_mm, lat, lon)。"""
    idx = list(chans).index("tp")
    day0 = np.datetime64(
        f"{init_day[:4]}-{init_day[4:6]}-{init_day[6:8]}T00:00:00"
    )
    # 时刻 = init 日 [6,12,..,6*steps] Z（值 = 前 6h 累积）；ISO 形式，避免 datetime 跨单位错
    tarr = era_store["time"].values
    total = None
    lat = era_store["lat"].values
    lon = era_store["lon"].values
    for k in range(1, steps + 1):
        t = day0 + np.timedelta64(6 * k, "h")
        i = int(np.argmin(np.abs(tarr - t)))
        arr = era_store["data"][i, idx, :, :].values  # 先按索引切再取，只物化一片
        arr = _mask_missing(arr)
        total = arr if total is None else total + arr
    # store 的 tp time step 是 6h，但值是 **1h 累积**，一个 6h 步要 ×6——
    # 与推理侧 `loaders/era5_store.py` 的 `_ACCUMULATED_1H` 同口径。
    # 漏掉这一步实况整体偏干 6 倍（2026-10-27 实测：全球均值 0.437 vs 预报
    # 2.55 mm/day；×6 后 2.62，与预报对得上）。
    return total * 1000.0 * 6.0, lat, lon  # 1h 累积 -> 6h，m -> mm


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="聚合暴雨个例空间 tp 场（预报+ERA 实况）")
    p.add_argument("--init-days", default="20250527,20250528,20250529",
                   help="起报日，逗号分隔（默认 202510 的 S..E 共 3 天；口径 B 的示意场只给 S 一天）")
    p.add_argument("--process", default="202510", help="个例编号，仅作 manifest/目录名标签")
    p.add_argument("--model", default="Tongyun",
                   help="预报侧模型名，写进 manifest，图注与行标签都用它")
    p.add_argument("--forecast-root", default="/workspace/data/shenzw/tongyun_single",
                   help="Tongyun 6h 步 nc 根目录")
    p.add_argument("--era-store",
                   default="/workspace/data/liujunjie/era5_foundation_store2/"
                           "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
                   help="ERA5 sfc zarr（含 tp 通道，6h 档）")
    p.add_argument("--steps", type=int, default=4,
                   help="6h 步个数（24h 窗 = 4 步）")
    p.add_argument("--out", default="/workspace/data/shenzw/rainstorm_202510_spatial",
                   help="聚合输出目录")
    args = p.parse_args(argv)

    init_days = args.init_days.split(",")
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    store = xr.open_zarr(args.era_store, consolidated=False)
    chans = store["channel"].values

    manifest = {"process": args.process, "model": args.model, "mm": True, "days": {}}
    for d in init_days:
        zf, latf, lonf = load_forecast_24h(args.forecast_root, d, args.steps)
        ze, late, lone = load_era_24h(store, chans, d, args.steps)

        ds_pred = xr.Dataset({"tp": (["lat", "lon"], zf)},
                             coords={"lat": latf, "lon": lonf})
        ds_era = xr.Dataset({"tp": (["lat", "lon"], ze)},
                            coords={"lat": late, "lon": lone})
        pf, pe = OUT / f"pred_{d}.nc", OUT / f"era_{d}.nc"
        ds_pred.to_netcdf(pf)
        ds_era.to_netcdf(pe)
        manifest["days"][d] = {
            "pred": pf.name,
            "era": pe.name,
            "pred_sum_mm": float(np.nansum(zf)),
            "era_sum_mm": float(np.nansum(ze)),
        }
        print(f"{d}: pred={pf.name} era={pe.name}")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())