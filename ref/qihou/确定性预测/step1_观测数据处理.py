"""
CRA v1.5 数据处理流水线：气候态 → 距平(DIFF) → 插值 → 合并 lead_time。

处理变量：tp（可配置为 t2m、ssr、ssrd 等）

流水线：
  1. CLIM  — 逐日气候态（2004-2023，0.25°原始分辨率）
  2. DIFF  — 逐日距平 = 原始值 - 气候态（0.25°原始分辨率）
  3. INTERP — DIFF 插值到全球 1.5° 网格
  4. COMBINE — 按 lead_time=60 合并距平数据

用法：
    python CRA_v1.5_process.py

作者：FDP Project
日期：2026-06-16
"""

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import sys
import argparse

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
#  全局配置 —— 切换变量/路径只改这里（单独运行该代码时）
# ══════════════════════════════════════════════════════════════════════

# ── 变量名 ──
# VAR_NAME = "tp"
VAR_NAME = "t2m"

# ── 路径配置 ──
RAW_DATA_DIR       = Path(f"/gpu/zhouchg/zhouchg/CRA/output/{VAR_NAME}")                #输入元数据目录格式为  /gpu/zhouchg/zhouchg/CRA/output/{VAR_NAME}/2021/20210101.nc
CLIM_OUTPUT_DIR    = Path(f"/gpu/zhouchg/liuzh/CRA_1.5/clim_1p5/{VAR_NAME}")
DIFF_OUTPUT_DIR    = Path(f"/gpu/zhouchg/liuzh/CRA_1.5/diff_1p5/{VAR_NAME}")
INTERP_OUTPUT_DIR  = Path(f"/gpu/zhouchg/liuzh/CRA_1.5/anom_1p5/{VAR_NAME}")
COMBINE_OUTPUT_DIR = Path(f"/gpu/zhouchg/liuzh/CRA_1.5/anom_combine_1p5/{VAR_NAME}")
# RAW_DATA_DIR       = Path("step1_data")                              # 根目录位置
# CLIM_OUTPUT_DIR    = Path(f"step1_data_11/clim_1p5/{VAR_NAME}")
# DIFF_OUTPUT_DIR    = Path(f"step1_data_11/diff_1p5/{VAR_NAME}")
# INTERP_OUTPUT_DIR  = Path(f"step1_data_11/anom_1p5/{VAR_NAME}")
# COMBINE_OUTPUT_DIR = Path(f"step1_data_11/anom_combine_1p5/{VAR_NAME}")

# ── 年份范围 ──
CLIM_YEARS   = list(range(2004, 2024))   # 气候态基准年 2004-2023
DIFF_YEARS   = [2024, 2025]              # 需要计算 DIFF 的年份
LEAD_TIME_N  = 60                        # 合并时未来天数

# ── 目标 1.5° 网格 ──
TARGET_LAT = np.arange(90.0, -91.0, -1.5)   # 121 点：90, 88.5, …, -90
TARGET_LON = np.arange(0.0, 360.0, 1.5)     # 240 点：0, 1.5, …, 358.5

TARGET_LAT_SIZE = len(TARGET_LAT)            # 121
TARGET_LON_SIZE = len(TARGET_LON)            # 240

# ── 缺失文件处理策略 ──
FILL_MISSING_WITH_NAN = True  # 合并阶段：True=NaN填充, False=跳过该起报日

# ── 压缩参数 ──
COMPRESSION = {"zlib": True, "complevel": 4}

# ── 变量属性表（新增变量在此添加一行即可） ──
VAR_ATTRS = {
    "tp":   {"long_name": "Total Precipitation",              "units": "m"},
    "t2m":  {"long_name": "2-Metre Temperature",              "units": "K"},
    "ssr":  {"long_name": "Surface Net Solar Radiation",      "units": "J m**-2"},
    "ssrd": {"long_name": "Surface Solar Radiation Downwards","units": "J m**-2"},
}


# ══════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    """统一日志输出，带时间戳。"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def is_feb29(mmdd: str) -> bool:
    """判断 mmdd 是否为 02-29。"""
    return mmdd == "0229"


def generate_non_leap_mmdd_list(start_year: int = 2001) -> list:
    """生成平年 365 天 mmdd 列表，排除 02-29。"""
    result = []
    dt = datetime(start_year, 1, 1)
    for _ in range(365):
        if not (dt.month == 2 and dt.day == 29):
            result.append(dt.strftime("%m%d"))
        dt += timedelta(days=1)
    return result


# 全局 365 天 mmdd 列表（平年，不含 02-29）
ALL_MMDD = generate_non_leap_mmdd_list(2001)


def find_nc_file(year: int, mmdd: str) -> Path:
    """根据年份和 mmdd 构造原始 NC 文件路径。"""
    return RAW_DATA_DIR / str(year) / f"{year}{mmdd}.nc"    


def get_var_attr(key: str, default: str = "") -> str:
    """获取变量属性，不存在则返回默认值。"""
    return VAR_ATTRS.get(VAR_NAME, {}).get(key, default)


# ══════════════════════════════════════════════════════════════════════
#  NC 文件自动检测
# ══════════════════════════════════════════════════════════════════════

def inspect_nc_file(file_path: Path) -> dict:
    """检查 NetCDF 文件结构并打印关键信息。"""
    info = {}
    try:
        ds = xr.open_dataset(file_path)
        info["dims"] = dict(ds.sizes)
        info["data_vars"] = list(ds.data_vars)
        info["coords"] = list(ds.coords)
        info["attrs"] = dict(ds.attrs)

        lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
        lon_candidates = ["lon", "longitude", "LON", "Longitude"]

        for c in lat_candidates:
            if c in ds.coords:
                info["lat_name"] = c; break
        for c in lon_candidates:
            if c in ds.coords:
                info["lon_name"] = c; break

        lat_name = info.get("lat_name")
        lon_name = info.get("lon_name")
        if lat_name is not None:
            lat_vals = ds[lat_name].values
            info["lat_size"] = len(lat_vals)
            info["lat_is_north_to_south"] = (len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1])
        if lon_name is not None:
            lon_vals = ds[lon_name].values
            info["lon_size"] = len(lon_vals)

        ds.close()
        log(f"  → 文件: {file_path.name}, 维度: {info['dims']}, 变量: {info['data_vars']}")
    except Exception as e:
        log(f"  无法打开 {file_path.name}: {e}", "WARN")
    return info


def auto_detect_structure(sample_file: Path) -> dict:
    """
    自动检测数据结构：变量名、维度名、坐标名、经纬度范围和方向。

    这是整个流水线的基础——所有后续函数从此获取 lat_name/lon_name/var_name，
    完全避免硬编码。

    返回
    -------
    dict  包含: var_name, lat_name, lon_name, lat_size, lon_size,
          lat_is_north_to_south, lon_min, lon_max, dims, data_vars, coords, ...
    """
    ds = xr.open_dataset(sample_file)
    sizes = dict(ds.sizes)

    # ── 变量名：优先配置名，回退到第一个 data_var ──
    var_name = VAR_NAME if VAR_NAME in ds.data_vars else list(ds.data_vars)[0]
    if var_name != VAR_NAME:
        log(f"  [WARN] 未找到 '{VAR_NAME}'，回退到 '{var_name}'", "WARN")

    # ── 坐标名探测 ──
    lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude"]

    lat_name = next((c for c in lat_candidates if c in ds.coords), None)
    lon_name = next((c for c in lon_candidates if c in ds.coords), None)

    if lat_name is None or lon_name is None:
        for dim in sizes:
            if lat_name is None and dim in lat_candidates:
                lat_name = dim
            if lon_name is None and dim in lon_candidates:
                lon_name = dim

    if lat_name is None:
        sorted_dims = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
        lat_name = sorted_dims[1][0] if len(sorted_dims) > 1 else "lat"
        lon_name = sorted_dims[0][0]
        log(f"  [WARN] 猜解坐标名: lat='{lat_name}' lon='{lon_name}'", "WARN")

    # ── 经纬度信息 ──
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    lat_is_n_to_s = len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]

    info = {
        "var_name": var_name,
        "lat_name": lat_name,
        "lon_name": lon_name,
        "lat_size": len(lat_vals),
        "lon_size": len(lon_vals),
        "lat_min": float(lat_vals.min()),
        "lat_max": float(lat_vals.max()),
        "lon_min": float(lon_vals.min()),
        "lon_max": float(lon_vals.max()),
        "lat_is_north_to_south": lat_is_n_to_s,
        "dims": sizes,
        "data_vars": list(ds.data_vars),
        "coords": list(ds.coords),
        "var_attrs": dict(ds[var_name].attrs),
        "global_attrs": dict(ds.attrs),
    }
    ds.close()

    log(f"  [检测结果] var='{var_name}', lat='{lat_name}'({info['lat_size']}), "
        f"lon='{lon_name}'({info['lon_size']}), "
        f"方向={'N→S' if lat_is_n_to_s else 'S→N'}, "
        f"lat=[{info['lat_min']:.2f},{info['lat_max']:.2f}], "
        f"lon=[{info['lon_min']:.2f},{info['lon_max']:.2f}]")
    return info


# ══════════════════════════════════════════════════════════════════════
#  坐标处理
# ══════════════════════════════════════════════════════════════════════

def normalize_longitude(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    """经度统一到 0-360，并确保单调递增。"""
    lon_vals = da[lon_name].values
    if np.any(lon_vals < 0):
        da = da.assign_coords({lon_name: (lon_vals + 360) % 360})
        da = da.sortby(lon_name)
    return da


def ensure_lat_increasing(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    """确保 latitude 单调递增。"""
    lat_vals = da[lat_name].values
    if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
        da = da.sortby(lat_name)
    return da


def normalize_coords(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """统一处理：lat 单调递增 + lon 转为 0-360。"""
    da = normalize_longitude(da, lon_name)
    da = ensure_lat_increasing(da, lat_name)
    return da


# ══════════════════════════════════════════════════════════════════════
#  插值
# ══════════════════════════════════════════════════════════════════════

def interpolate_to_1p5(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
    method: str = "linear",
) -> xr.DataArray:
    """
    双线性插值到全局 1.5° 网格。

    参数
    ----------
    da       : xr.DataArray  输入（须含 lat_name, lon_name 坐标）
    lat_name : str           纬度坐标名
    lon_name : str           经度坐标名
    method   : str           插值方法

    返回
    -------
    xr.DataArray  维度 (..., lat=121, lon=240)
    """
    da = da.sortby(lat_name).sortby(lon_name)
    da_renamed = da.rename({lat_name: "lat", lon_name: "lon"})

    da_interp = da_renamed.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method=method,
        kwargs={"fill_value": None},
    )
    return da_interp


# ══════════════════════════════════════════════════════════════════════
#  通用保存
# ══════════════════════════════════════════════════════════════════════

def save_dataset(
    da: xr.DataArray,
    output_path: Path,
    var_name: str = None,
    long_name: str = None,
    units: str = None,
    description: str = "",
    extra_attrs: dict = None,
):
    """
    保存 DataArray 为 NetCDF，自动添加变量/全局属性和压缩。

    参数
    ----------
    da          : xr.DataArray
    output_path : Path
    var_name    : str  输出变量名
    long_name   : str  变量 long_name
    units       : str  变量 units
    description : str  描述（同时写入变量和全局属性）
    extra_attrs : dict 额外的全局属性
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if var_name is None:
        var_name = da.name if da.name else "data"

    ds_out = da.to_dataset(name=var_name)

    # 变量属性
    if long_name:
        ds_out[var_name].attrs["long_name"] = long_name
    if units:
        ds_out[var_name].attrs["units"] = units
    if description:
        ds_out[var_name].attrs["description"] = description

    # 全局属性
    ds_out.attrs["source"] = "CRA Reanalysis / FDP Project"
    ds_out.attrs["var_name"] = var_name
    ds_out.attrs["created_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if description:
        ds_out.attrs["description"] = description
    if extra_attrs:
        ds_out.attrs.update(extra_attrs)

    encoding = {var_name: COMPRESSION}
    ds_out.to_netcdf(output_path, encoding=encoding)
    log(f"    保存 → {output_path}")


# ══════════════════════════════════════════════════════════════════════
#  第1步：CLIM — 计算逐日气候态（原始分辨率）
# ══════════════════════════════════════════════════════════════════════

def compute_daily_climatology(
    structure: dict,
    clim_years: list = None,
    output_dir: Path = None,
    overwrite: bool = False,
) -> dict:
    """
    计算 365 天逐日气候态（2004-2023 平均，排除 02-29）。

    文件缺失时自动用已有年份计算，并给出 WARNING。

    输出
    -------
    dict  {mmdd: Path}  365 个气候态文件路径
    """
    if clim_years is None:
        clim_years = CLIM_YEARS
    if output_dir is None:
        output_dir = CLIM_OUTPUT_DIR

    var_name = structure["var_name"]
    lat_name = structure["lat_name"]
    lon_name = structure["lon_name"]

    log(f"\n{'='*60}")
    log(f" 第1步 CLIM │ 逐日气候态 [{var_name}]  ({clim_years[0]}-{clim_years[-1]}, 原始分辨率)")
    log(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    skipped = 0

    for idx, mmdd in enumerate(ALL_MMDD):
        out_path = output_dir / f"{mmdd}.nc"

        if not overwrite and out_path.exists():
            generated[mmdd] = out_path
            skipped += 1
            if (idx + 1) % 50 == 0:
                log(f"  [{idx+1:>3}/365] {mmdd} — 已存在，跳过")
            continue

        # 收集所有年份该日数据
        arrays = []
        missing = []
        for year in clim_years:
            if mmdd == "0229":
                continue
            nc_path = find_nc_file(year, mmdd)
            if not nc_path.exists():
                missing.append(year)
                continue
            try:
                ds = xr.open_dataset(nc_path)
                arrays.append(ds[var_name].copy())
                ds.close()
            except Exception as e:
                log(f"  [WARN] 读取 {nc_path} 失败: {e}", "WARN")
                missing.append(year)

        if missing:
            log(f"  [WARN] {mmdd}: 缺失 {missing}, 用 {len(arrays)}/{len(clim_years)} 年", "WARN")

        if not arrays:
            log(f"  [ERROR] {mmdd}: 全部缺失！", "ERROR")
            continue

        # 对齐坐标后平均
        if len(arrays) > 1:
            ref = {lat_name: arrays[0][lat_name], lon_name: arrays[0][lon_name]}
            for i in range(1, len(arrays)):
                arrays[i] = arrays[i].assign_coords(ref)
        da_clim = xr.concat(arrays, dim="year").mean(dim="year")

        save_dataset(
            da_clim, out_path, var_name=var_name,
            long_name=f"{get_var_attr('long_name', var_name)} Daily Climatology",
            units=get_var_attr("units"),
            description=(
                f"{var_name} daily climatology for {mmdd}, "
                f"averaged over {clim_years[0]}-{clim_years[-1]} "
                f"({len(arrays)} years)."
            ),
            extra_attrs={
                "step": "CLIM",
                "climatology_period": f"{clim_years[0]}-{clim_years[-1]}",
                "mmdd": mmdd,
                "years_used": len(arrays),
                "years_missing": missing if missing else "none",
            },
        )
        generated[mmdd] = out_path

        if (idx + 1) % 50 == 0:
            log(f"  [{idx+1:>3}/365] {mmdd} — 完成")

    log(f"  CLIM 完成: 生成 {len(generated)-skipped}, 跳过 {skipped}")
    return generated


# ══════════════════════════════════════════════════════════════════════
#  第2步：DIFF — 计算逐日距平（原始分辨率）
# ══════════════════════════════════════════════════════════════════════

def compute_daily_diff(
    structure: dict,
    clim_files: dict,
    diff_years: list = None,
    output_dir: Path = None,
    overwrite: bool = False,
) -> dict:
    """
    计算逐日距平（DIFF = 原始值 - 气候态），保持在原始 0.25° 分辨率。

    排除 02-29；2024 年虽为闰年也不处理 02-29。

    输出
    -------
    dict  {year: {mmdd: Path}}
    """
    if diff_years is None:
        diff_years = DIFF_YEARS
    if output_dir is None:
        output_dir = DIFF_OUTPUT_DIR

    var_name = structure["var_name"]
    lat_name = structure["lat_name"]
    lon_name = structure["lon_name"]

    log(f"\n{'='*60}")
    log(f" 第2步 DIFF │ 逐日距平 [{var_name}]  ({diff_years}, 原始分辨率)")
    log(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = defaultdict(dict)

    for year in diff_years:
        log(f"\n  ── {year} ──")
        year_dir = output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        success = 0

        for idx, mmdd in enumerate(ALL_MMDD):
            out_path = year_dir / f"{mmdd}.nc"

            if not overwrite and out_path.exists():
                all_results[year][mmdd] = out_path
                if (idx + 1) % 50 == 0:
                    log(f"    [{idx+1:>3}/365] {mmdd} — 已存在")
                continue

            try:
                # 读取原始数据
                nc_path = find_nc_file(year, mmdd)
                if not nc_path.exists():
                    log(f"    [WARN] {year}{mmdd}: 原始文件缺失", "WARN")
                    continue

                ds = xr.open_dataset(nc_path)
                da_raw = ds[var_name].copy()
                ds.close()

                # 读取气候态
                if mmdd not in clim_files:
                    log(f"    [WARN] {year}{mmdd}: 气候态 {mmdd}.nc 缺失", "WARN")
                    continue

                ds_clim = xr.open_dataset(clim_files[mmdd])
                da_clim = ds_clim[var_name].copy()
                ds_clim.close()

                # DIFF
                da_diff = da_raw - da_clim

                # 坐标统一
                da_diff = normalize_coords(da_diff, lat_name, lon_name)

                save_dataset(
                    da_diff, out_path, var_name=var_name,
                    long_name=f"{get_var_attr('long_name', var_name)} Difference",
                    units=get_var_attr("units"),
                    description=(
                        f"{var_name} DIFF for {year}{mmdd} at original resolution. "
                        f"DIFF = raw({year}) - clim({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})."
                    ),
                    extra_attrs={
                        "step": "DIFF",
                        "method": f"daily({year}) - climatology({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})",
                        "year": year, "mmdd": mmdd,
                        "resolution": "0.25° (original)",
                    },
                )
                all_results[year][mmdd] = out_path
                success += 1

                if (idx + 1) % 50 == 0:
                    log(f"    [{idx+1:>3}/365] {mmdd} — 完成 ({success})")

            except Exception as e:
                log(f"    [ERROR] {year}{mmdd}: {e}", "ERROR")

        log(f"    年份 {year}: {success}/365 天")

    return dict(all_results)


# ══════════════════════════════════════════════════════════════════════
#  第3步：INTERP — DIFF 插值到 1.5°
# ══════════════════════════════════════════════════════════════════════

def interpolate_diff_to_1p5(
    structure: dict,
    diff_files: dict,
    output_dir: Path = None,
    overwrite: bool = False,
) -> dict:
    """
    将 DIFF 数据从原始 0.25° 双线性插值到全局 1.5° 网格。

    输出
    -------
    dict  {year: {mmdd: Path}}
    """
    if output_dir is None:
        output_dir = INTERP_OUTPUT_DIR

    var_name = structure["var_name"]
    lat_name = structure["lat_name"]
    lon_name = structure["lon_name"]

    log(f"\n{'='*60}")
    log(f" 第3步 INTERP │ DIFF 插值到 1.5° [{var_name}]")
    log(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = defaultdict(dict)

    for year in sorted(diff_files.keys()):
        log(f"\n  ── {year} ──")
        year_dir = output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        mmdd_list = sorted(diff_files[year].keys())
        success = 0

        for idx, mmdd in enumerate(mmdd_list):
            out_path = year_dir / f"{mmdd}.nc"

            if not overwrite and out_path.exists():
                all_results[year][mmdd] = out_path
                if (idx + 1) % 50 == 0:
                    log(f"    [{idx+1:>3}/{len(mmdd_list)}] {mmdd} — 已存在")
                continue

            try:
                ds = xr.open_dataset(diff_files[year][mmdd])
                da_diff = ds[var_name].copy()
                ds.close()

                da_interp = interpolate_to_1p5(da_diff, lat_name, lon_name)

                save_dataset(
                    da_interp, out_path, var_name=var_name,
                    long_name=f"{get_var_attr('long_name', var_name)} Difference (1.5°)",
                    units=get_var_attr("units"),
                    description=(
                        f"{var_name} DIFF for {year}{mmdd} interpolated to 1.5° grid. "
                        f"DIFF = raw({year}) - clim({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})."
                    ),
                    extra_attrs={
                        "step": "INTERP",
                        "method": f"daily({year}) - climatology({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})",
                        "year": year, "mmdd": mmdd,
                        "interpolation": "bilinear 0.25°→1.5°",
                        "grid": f"{TARGET_LAT_SIZE}×{TARGET_LON_SIZE} (lat×lon)",
                    },
                )
                all_results[year][mmdd] = out_path
                success += 1

                if (idx + 1) % 50 == 0:
                    log(f"    [{idx+1:>3}/{len(mmdd_list)}] {mmdd} — 完成 ({success})")

            except Exception as e:
                log(f"    [ERROR] {year}{mmdd}: {e}", "ERROR")

        log(f"    年份 {year}: {success}/{len(mmdd_list)} 天")

    return dict(all_results)


# ══════════════════════════════════════════════════════════════════════
#  第4步：COMBINE — 按 lead_time 合并距平数据
# ══════════════════════════════════════════════════════════════════════

def combine_anomaly_by_leadtime(
    interp_files: dict,
    lead_time_n: int = None,
    output_dir: Path = None,
    fill_missing_with_nan: bool = None,
    overwrite: bool = False,
) -> dict:
    """
    将 1.5° 距平数据按起报日合并未来 lead_time_n 天。

    输出维度：(lead_time, lat, lon) = (60, 121, 240)

    参数
    ----------
    interp_files          : dict  {year: {mmdd: Path}}
    lead_time_n           : int   未来天数
    output_dir            : Path
    fill_missing_with_nan : bool  True=NaN填充, False=跳过该起报日
    overwrite             : bool

    输出
    -------
    dict  {(year, mmdd): Path}
    """
    if lead_time_n is None:
        lead_time_n = LEAD_TIME_N
    if output_dir is None:
        output_dir = COMBINE_OUTPUT_DIR
    if fill_missing_with_nan is None:
        fill_missing_with_nan = FILL_MISSING_WITH_NAN

    log(f"\n{'='*60}")
    log(f" 第4步 COMBINE │ 合并 lead_time={lead_time_n} [{VAR_NAME}]")
    log(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建全局索引 yyyymmdd → Path
    all_daily = {}
    for year, files in interp_files.items():
        for mmdd, path in files.items():
            all_daily[f"{year}{mmdd}"] = path

    log(f"  共 {len(all_daily)} 天 1.5° 距平文件可用")

    # 生成起报日期列表
    init_dates = []
    for year in sorted(interp_files.keys()):
        for mmdd in ALL_MMDD:
            key = f"{year}{mmdd}"
            if key in all_daily:
                init_dates.append((year, mmdd))
            else:
                log(f"  [WARN] 起报日 {key} 缺失", "WARN")

    log(f"  有效起报日: {len(init_dates)}")

    # 辅助：日期偏移（正确处理跨年）
    def get_future_date(init_year: int, init_mmdd: str, offset: int):
        dt = datetime(2001, int(init_mmdd[:2]), int(init_mmdd[2:4]))
        target = dt + timedelta(days=offset)
        target_mmdd = target.strftime("%m%d")
        actual_year = init_year + 1 if dt + timedelta(days=offset) > datetime(2001, 12, 31) else init_year
        return actual_year, target_mmdd

    results = {}
    skipped_missing = 0
    failed = 0

    for idx, (year, mmdd) in enumerate(init_dates):
        out_path = output_dir / str(year) / f"{year}{mmdd}.nc"

        if not overwrite and out_path.exists():
            results[(year, mmdd)] = out_path
            if (idx + 1) % 100 == 0:
                log(f"  [{idx+1}/{len(init_dates)}] {year}{mmdd} — 已存在")
            continue

        # 收集未来 lead_time_n 天
        slices = []
        missing = []
        for lead in range(1, lead_time_n + 1):
            fy, fmmdd = get_future_date(year, mmdd, lead)
            fkey = f"{fy}{fmmdd}"
            if fkey in all_daily:
                try:
                    ds = xr.open_dataset(all_daily[fkey])
                    slices.append(ds[VAR_NAME].copy())
                    ds.close()
                except Exception as e:
                    log(f"  [WARN] 读取 {fkey}: {e}", "WARN")
                    missing.append(fkey)
                    if fill_missing_with_nan:
                        slices.append(None)
            else:
                missing.append(fkey)
                if fill_missing_with_nan:
                    slices.append(None)

        if missing:
            if fill_missing_with_nan:
                log(f"  [WARN] {year}{mmdd}: {len(missing)} 天缺失→NaN: {missing[:5]}...", "WARN")
                valid = next((s for s in slices if s is not None), None)
                if valid is None:
                    log(f"  [ERROR] {year}{mmdd}: 全部缺失！", "ERROR")
                    failed += 1
                    continue
                nan_slice = xr.full_like(valid, np.nan)
                slices = [s if s is not None else nan_slice for s in slices]
            else:
                log(f"  [WARN] {year}{mmdd}: {len(missing)} 天缺失，跳过", "WARN")
                skipped_missing += 1
                continue

        try:
            da_combined = xr.concat(slices, dim="lead_time")
            da_combined = da_combined.assign_coords(lead_time=np.arange(1, lead_time_n + 1))
        except Exception as e:
            log(f"  [ERROR] {year}{mmdd}: concat 失败: {e}", "ERROR")
            failed += 1
            continue

        save_dataset(
            da_combined, out_path, var_name=VAR_NAME,
            long_name=f"{get_var_attr('long_name', VAR_NAME)} Anomaly (1.5°, lead_time)",
            units=get_var_attr("units"),
            description=(
                f"{VAR_NAME} anomaly ensemble for init {year}{mmdd}. "
                f"lead_time=1..{lead_time_n} days, 1.5° grid."
            ),
            extra_attrs={
                "step": "COMBINE",
                "init_date": f"{year}{mmdd}", "init_year": year,
                "lead_time_n": lead_time_n, "lead_time_units": "days",
                "grid": f"{TARGET_LAT_SIZE}×{TARGET_LON_SIZE}",
                "method": f"daily - climatology({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})",
                "missing_dates": missing if missing else "none",
            },
        )
        results[(year, mmdd)] = out_path

        if (idx + 1) % 100 == 0:
            log(f"  [{idx+1}/{len(init_dates)}] {year}{mmdd} — 完成")

    log(f"  COMBINE 完成: {len(results)} 成功, {skipped_missing} 跳过, {failed} 失败")
    return results


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数：变量、年份、路径均可由编排器传入。"""
    parser = argparse.ArgumentParser(description="CRA v1.5 观测数据处理流水线")
    parser.add_argument("--var", type=str, default=VAR_NAME, help="变量名")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DATA_DIR), help="原始数据根目录")
    parser.add_argument("--clim-dir", type=str, default=str(CLIM_OUTPUT_DIR), help="气候态输出目录")
    parser.add_argument("--diff-dir", type=str, default=str(DIFF_OUTPUT_DIR), help="逐日距平输出目录")
    parser.add_argument("--interp-dir", type=str, default=str(INTERP_OUTPUT_DIR), help="1.5° 距平输出目录")
    parser.add_argument("--combine-dir", type=str, default=str(COMBINE_OUTPUT_DIR), help="合并 lead_time 输出目录")
    parser.add_argument("--clim-years", type=int, nargs=2, default=[CLIM_YEARS[0], CLIM_YEARS[-1]], help="气候态年份起止")
    parser.add_argument("--diff-years", type=int, nargs="+", default=list(DIFF_YEARS), help="距平年份列表")
    parser.add_argument("--lead-time", type=int, default=LEAD_TIME_N, help="lead_time 天数")
    return parser.parse_args()


def main():
    """主入口：按 CLIM → DIFF → INTERP → COMBINE 顺序执行。"""
    args = parse_args()
    global VAR_NAME, RAW_DATA_DIR, CLIM_OUTPUT_DIR, DIFF_OUTPUT_DIR, INTERP_OUTPUT_DIR, COMBINE_OUTPUT_DIR
    global CLIM_YEARS, DIFF_YEARS, LEAD_TIME_N
    VAR_NAME = args.var
    RAW_DATA_DIR = Path(args.raw_dir)
    CLIM_OUTPUT_DIR = Path(args.clim_dir)
    DIFF_OUTPUT_DIR = Path(args.diff_dir)
    INTERP_OUTPUT_DIR = Path(args.interp_dir)
    COMBINE_OUTPUT_DIR = Path(args.combine_dir)
    CLIM_YEARS = list(range(args.clim_years[0], args.clim_years[1] + 1))
    DIFF_YEARS = list(args.diff_years)
    LEAD_TIME_N = args.lead_time
    log("=" * 60)
    log(f"CRA v1.5 处理流水线: CLIM → DIFF → INTERP → COMBINE")
    log(f"  变量: {VAR_NAME}")
    log(f"  气候态: {CLIM_YEARS[0]}-{CLIM_YEARS[-1]}")
    log(f"  DIFF 年份: {DIFF_YEARS}")
    log(f"  lead_time: {LEAD_TIME_N}")
    log(f"  目标网格: {TARGET_LAT_SIZE}×{TARGET_LON_SIZE}")
    log(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # ── 0. 自动检测数据结构 ──
    log(f"\n── 数据结构自动检测 ──")

    # sample_file = None                                                        ###################################
    # for year in CLIM_YEARS + DIFF_YEARS:                                      ###################################
    #     f = find_nc_file(year, "0101")                                        ###################################
    #     if f.exists():                                                        ###################################
    #         sample_file = f                                                   ###################################
    #         break                                                             ###################################
    # if sample_file is None:                                                   ###################################
    #     log("无法找到示例 NC 文件！检查路径。", "ERROR")                         ###################################
    #     sys.exit(1)                                                           ###################################

    sample_file = None

    for year in CLIM_YEARS + DIFF_YEARS:
        year_dir = RAW_DATA_DIR / str(year)

        if not year_dir.is_dir():
            continue

        sample_files = sorted(year_dir.glob("*.nc"))
        if sample_files:
            sample_file = sample_files[0]
            break
    if sample_file is None:
        log(f"无法找到示例 NC 文件，原始数据目录: {RAW_DATA_DIR}", "ERROR")
        log("预期格式: <原始数据目录>/<年份>/<YYYYMMDD>.nc", "ERROR")
        sys.exit(1)
 #########################################################################################################################    

    log(f"  示例文件: {sample_file}")
    inspect_nc_file(sample_file)
    structure = auto_detect_structure(sample_file)

    if structure["var_name"] not in structure["data_vars"]:
        log(f"  [ERROR] 变量 '{structure['var_name']}' 不存在！", "ERROR")
        sys.exit(1)

    # ── 第1步：CLIM ──
    clim_files = compute_daily_climatology(structure, overwrite=False)

    # ── 第2步：DIFF ──
    diff_files = compute_daily_diff(structure, clim_files, overwrite=False)

    # ── 第3步：INTERP ──
    interp_files = interpolate_diff_to_1p5(structure, diff_files, overwrite=False)

    # ── 第4步：COMBINE ──
    combine_files = combine_anomaly_by_leadtime(interp_files, overwrite=False)

    # ── 汇总 ──
    log("\n" + "=" * 60)
    log("流水线全部完成！")
    log(f"  CLIM:    {len(clim_files)} 天")
    log(f"  DIFF:    {sum(len(v) for v in diff_files.values())} 天")
    log(f"  INTERP:  {sum(len(v) for v in interp_files.values())} 天")
    log(f"  COMBINE: {len(combine_files)} 个起报日")
    log("=" * 60)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════
#  使用说明（单独运行该代码时）
# ══════════════════════════════════════════════════════════════════════
#
# 【1. 运行方式】
#     python CRA_v1.5_process.py
#
# 【2. 修改变量】
#     修改 VAR_NAME： "tp" | "t2m" | "ssr" | "ssrd"
#     新增变量在 VAR_ATTRS 中加一行。
#
# 【3. 修改路径】
#     RAW_DATA_DIR       — 原始逐日 NC 根目录
#     CLIM_OUTPUT_DIR    — 气候态输出（0.25°）
#     DIFF_OUTPUT_DIR    — 距平输出（0.25°）
#     INTERP_OUTPUT_DIR  — 插值输出（1.5°）
#     COMBINE_OUTPUT_DIR — 合并 lead_time 输出（1.5°）
#
# 【4. 修改年份】
#     CLIM_YEARS — 气候态基准年（默认 2004-2023）
#     DIFF_YEARS — 距平年份（默认 [2024, 2025]）
#     LEAD_TIME_N — lead_time 长度（默认 60）
#
# 【5. 流水线四步输出维度】
#
#     步骤    目录                      维度                       示例
#     ─────  ────────────────────────  ────────────────────────  ────────────
#     CLIM    clim_1p5/{var}/          (lat=721, lon=1440)       0101.nc
#     DIFF    diff_1p5/{var}/YYYY/     (lat=721, lon=1440)       2024/0101.nc
#     INTERP  anom_1p5/{var}/YYYY/     (lat=121, lon=240)        2024/0101.nc
#     COMBINE anom_combine_1p5/{var}/YYYY/ (lead_time=60,        2024/20240101.nc
#                                           lat=121, lon=240)    → shape (60,121,240)
#
# 【6. lead_time 与有效日期】
#     init=YYYYmmdd, lead_time=L → valid_date = init_date + L 天（自动跨年）
#     例如: init=20240101, L=1 → 20240102
#          init=20241215, L=20 → 20250104
#
# 【7. 02-29 处理】
#     气候态和 DIFF 均排除 02-29；2024 年虽为闰年也不处理该日。
#
# 【8. 扩展其他变量】
#     1. VAR_NAME = "t2m"
#     2. 确认 VAR_ATTRS 中存在该变量条目
#     3. 无需修改任何函数
