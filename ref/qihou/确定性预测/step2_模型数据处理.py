"""
my_model — 逐日起报气候态 & 距平计算。

处理变量：t2m（可配置为 tp、ssr、ssrd 等）

流水线：
  1. 对每个 nc 文件 member 维度求平均 → (lead_time, lat, lon)
  2. 按起报日期 mmdd 计算 2004-2023 气候态 → 365 个气候态文件
  3. 计算 2024 年逐日起报距平 = 2024 member 均值 - 气候态 → 365 个距平文件

用法：
    python my_model_anom.py

作者：FDP Project
日期：2026-06-16
"""

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys
import argparse

# ══════════════════════════════════════════════════════════════════════════
#  日志配置
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  全局配置 —— 修改变量/路径/年份只需改这里（单独运行该代码时）
# ══════════════════════════════════════════════════════════════════════════

# ── 变量名 ──
VAR_NAME = "t2m"                # 可改为 "tp", "ssr", "ssrd" 等
OUTPUT_VAR_NAME = "t2m"         # 输出文件中的变量名，None 则沿用 VAR_NAME

# ── 路径 ──
RAW_DATA_DIR    = Path("/gpu/zhouchg/wangchp/FDP/Fengshun_v2.0/t2m")
CLIM_OUTPUT_DIR = Path("/gpu/zhouchg/liuzh/step2_output/clim_1p5/t2m")
ANOM_OUTPUT_DIR = Path("/gpu/zhouchg/liuzh/step2_output/anom_1p5/t2m")
# RAW_DATA_DIR    = Path("step2_data")
# CLIM_OUTPUT_DIR = Path(f"step2_data_22/clim_1p5/{OUTPUT_VAR_NAME}")
# ANOM_OUTPUT_DIR = Path(f"step2_data_22/anom_1p5/{OUTPUT_VAR_NAME}")

# ── 年份 ──
CLIM_START_YEAR = 2004
CLIM_END_YEAR   = 2023
ANOM_YEAR       = 2024

# ── 预期维度 ──
EXPECTED_MEMBER_N    = 11
EXPECTED_LEAD_TIME_N = 60
EXPECTED_LAT_N       = 121
EXPECTED_LON_N       = 240

# ── 压缩 ──
COMPRESSION = {"zlib": True, "complevel": 4}

# ── 变量属性表（新增变量在此添加一行） ──
VAR_ATTRS = {
    "tp":   {"long_name": "Total Precipitation",              "units": "m"},
    "t2m":  {"long_name": "2-Metre Temperature",              "units": "K"},
    "ssr":  {"long_name": "Surface Net Solar Radiation",      "units": "J m**-2"},
    "ssrd": {"long_name": "Surface Solar Radiation Downwards","units": "J m**-2"},
}


# ══════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════

def generate_dates(year: int) -> list:
    """
    生成全年 mmdd 列表，自动排除 02-29。

    参数
    ----------
    year : int  任意平年即可（如 2001），仅用于逐日迭代

    返回
    -------
    list[str]  e.g. ['0101', '0102', ..., '1231']，共 365 个
    """
    dates = []
    dt = datetime(year, 1, 1)
    for _ in range(365):
        if not (dt.month == 2 and dt.day == 29):
            dates.append(dt.strftime("%m%d"))
        dt += timedelta(days=1)
    return dates


# 全局 365 天 mmdd 列表
ALL_MMDD = generate_dates(2001)


def get_var_attr(key: str, default: str = "") -> str:
    """从 VAR_ATTRS 获取变量属性，不存在则返回默认值。"""
    return VAR_ATTRS.get(VAR_NAME, {}).get(key, default)


# ══════════════════════════════════════════════════════════════════════════
#  NC 文件自动检测
# ══════════════════════════════════════════════════════════════════════════

def inspect_nc_file(file_path: Path) -> dict:
    """
    检查 NetCDF 文件结构并返回关键信息。不依赖硬编码。

    返回
    -------
    dict  包含 dims, data_vars, coords, attrs, lat_name, lon_name,
          lead_time_name, member_name, var_name, lat_size, lon_size,
          lead_time_size, member_size, lat_is_descending, lon_min, lon_max 等
    """
    info = {}
    try:
        ds = xr.open_dataset(file_path)

        info["dims"] = dict(ds.sizes)
        info["data_vars"] = list(ds.data_vars)
        info["coords"] = list(ds.coords)
        info["attrs"] = dict(ds.attrs)

        # ── 坐标名自动探测 ──
        lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
        lon_candidates = ["lon", "longitude", "LON", "Longitude"]
        lead_candidates = ["lead_time", "leadtime", "lead", "LEAD_TIME", "step", "time"]
        member_candidates = ["member", "ensemble", "ens", "MEMBER", "realization"]

        for c in lat_candidates:
            if c in ds.coords or c in ds.dims:
                info["lat_name"] = c; break
        for c in lon_candidates:
            if c in ds.coords or c in ds.dims:
                info["lon_name"] = c; break
        for c in lead_candidates:
            if c in ds.coords or c in ds.dims:
                info["lead_time_name"] = c; break
        for c in member_candidates:
            if c in ds.coords or c in ds.dims:
                info["member_name"] = c; break

        # ── 变量名 ──
        var_name = VAR_NAME if VAR_NAME in ds.data_vars else list(ds.data_vars)[0]
        if var_name != VAR_NAME:
            log.warning("未找到变量 '%s'，回退到 '%s'", VAR_NAME, var_name)
        info["var_name"] = var_name

        # ── 经纬度信息 ──
        lat_name = info.get("lat_name")
        lon_name = info.get("lon_name")
        if lat_name:
            lat_vals = ds[lat_name].values
            info["lat_size"] = len(lat_vals)
            info["lat_min"] = float(lat_vals.min())
            info["lat_max"] = float(lat_vals.max())
            info["lat_is_descending"] = len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]
        if lon_name:
            lon_vals = ds[lon_name].values
            info["lon_size"] = len(lon_vals)
            info["lon_min"] = float(lon_vals.min())
            info["lon_max"] = float(lon_vals.max())

        info["var_attrs"] = dict(ds[info["var_name"]].attrs)

        ds.close()

        log.info("  文件: %s  |  维度: %s  |  变量: %s  |  坐标: %s",
                 file_path.name, info["dims"], info["data_vars"], info["coords"])
    except Exception as e:
        log.warning("无法打开 %s: %s", file_path.name, e)

    return info


def validate_structure(info: dict, file_path: Path) -> list:
    """
    验证数据结构是否符合预期，返回 warning 列表。

    参数
    ----------
    info : dict  由 inspect_nc_file 返回的信息字典

    返回
    -------
    list[str]  warning 消息
    """
    warnings_list = []
    fname = file_path.name

    lat_n = info.get("lat_size")
    lon_n = info.get("lon_size")
    lt_n  = info.get("lead_time_name") and info["dims"].get(info["lead_time_name"])
    mem_n = info.get("member_name") and info["dims"].get(info["member_name"])

    if lat_n and lat_n != EXPECTED_LAT_N:
        warnings_list.append(f"{fname}: lat={lat_n} (预期 {EXPECTED_LAT_N})")
    if lon_n and lon_n != EXPECTED_LON_N:
        warnings_list.append(f"{fname}: lon={lon_n} (预期 {EXPECTED_LON_N})")
    if lt_n and lt_n != EXPECTED_LEAD_TIME_N:
        warnings_list.append(f"{fname}: lead_time={lt_n} (预期 {EXPECTED_LEAD_TIME_N})")
    if mem_n and mem_n != EXPECTED_MEMBER_N:
        warnings_list.append(f"{fname}: member={mem_n} (预期 {EXPECTED_MEMBER_N})")

    lon_min = info.get("lon_min")
    if lon_min is not None and lon_min < 0:
        warnings_list.append(f"{fname}: lon 范围为 {lon_min}~{info.get('lon_max')}，含负值")

    lat_desc = info.get("lat_is_descending")
    if lat_desc:
        warnings_list.append(f"{fname}: lat 为降序 (N→S)")

    return warnings_list


# ══════════════════════════════════════════════════════════════════════════
#  维度归一化
# ══════════════════════════════════════════════════════════════════════════

def normalize_dimensions(da: xr.DataArray, info: dict) -> xr.DataArray:
    """
    将 DataArray 维度转置为 (member, lead_time, lat, lon) 标准顺序。

    同时确保 lat 升序、lon 在 0-360 范围。

    参数
    ----------
    da   : xr.DataArray  原始数据
    info : dict          由 inspect_nc_file 返回的结构信息

    返回
    -------
    xr.DataArray  维度顺序为 (member, lead_time, lat, lon)（若存在 member）
                  或 (lead_time, lat, lon)
    """
    lat_name  = info.get("lat_name", "lat")
    lon_name  = info.get("lon_name", "lon")
    lt_name   = info.get("lead_time_name", "lead_time")
    mem_name  = info.get("member_name", "member")

    # ── 经度：-180~180 → 0~360 ──
    if lon_name in da.coords:
        lon_vals = da[lon_name].values
        if np.any(lon_vals < 0):
            da = da.assign_coords({lon_name: (lon_vals + 360) % 360})
            da = da.sortby(lon_name)

    # ── 纬度：确保升序 (S→N) ──
    if lat_name in da.coords:
        lat_vals = da[lat_name].values
        if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            da = da.sortby(lat_name)

    # ── 重命名坐标到标准名 ──
    rename_map = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if lt_name != "lead_time":
        rename_map[lt_name] = "lead_time"
    if mem_name != "member" and mem_name in da.dims:
        rename_map[mem_name] = "member"
    if rename_map:
        da = da.rename(rename_map)

    # ── 转置为标准顺序 ──
    if "member" in da.dims:
        target_dims = ["member", "lead_time", "lat", "lon"]
    else:
        target_dims = ["lead_time", "lat", "lon"]
    existing = [d for d in target_dims if d in da.dims]
    da = da.transpose(*existing)

    return da


# ══════════════════════════════════════════════════════════════════════════
#  Member 平均
# ══════════════════════════════════════════════════════════════════════════

def compute_member_mean(da: xr.DataArray) -> xr.DataArray:
    """对 member 维度求平均，输出维度 (lead_time, lat, lon)。"""
    if "member" in da.dims:
        return da.mean(dim="member")
    return da


# ══════════════════════════════════════════════════════════════════════════
#  通用保存函数
# ══════════════════════════════════════════════════════════════════════════

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
    保存 DataArray 为 NetCDF，自动添加变量属性、坐标属性和全局属性。

    参数
    ----------
    da          : xr.DataArray
    output_path : Path
    var_name    : str  输出变量名，None 则使用 da.name
    long_name   : str
    units       : str
    description : str
    extra_attrs : dict  额外的全局属性
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if var_name is None:
        var_name = da.name if da.name else "data"

    ds_out = da.to_dataset(name=var_name)

    # ── 变量属性 ──
    if long_name:
        ds_out[var_name].attrs["long_name"] = long_name
    if units:
        ds_out[var_name].attrs["units"] = units
    if description:
        ds_out[var_name].attrs["description"] = description

    # ── 坐标属性 ──
    if "lead_time" in ds_out.coords:
        ds_out["lead_time"].attrs = {
            "long_name": "forecast lead time",
            "units": "days",
            "description": "Lead time in days from init_date",
        }
    if "lat" in ds_out.coords:
        ds_out["lat"].attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
        }
    if "lon" in ds_out.coords:
        ds_out["lon"].attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
        }

    # ── 全局属性 ──
    ds_out.attrs["source"] = "my_model Ensemble Forecast System / FDP Project"
    ds_out.attrs["created_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if description:
        ds_out.attrs["description"] = description
    if extra_attrs:
        ds_out.attrs.update(extra_attrs)

    encoding = {var_name: COMPRESSION}
    ds_out.to_netcdf(output_path, encoding=encoding)
    log.info("  保存 → %s", output_path)


# ══════════════════════════════════════════════════════════════════════════
#  第1步：计算逐日起报气候态
# ══════════════════════════════════════════════════════════════════════════

def compute_daily_climatology(
    info: dict,
    clim_start_year: int = CLIM_START_YEAR,
    clim_end_year: int = CLIM_END_YEAR,
    output_dir: Path = None,
) -> dict:
    """
    计算逐日起报日期气候态（2004-2023 年 member 平均后再多年平均）。

    对每个 mmdd：
      1. 收集 clim_start_year ~ clim_end_year 所有年份该起报日期的文件
      2. 逐个文件做 member 平均 → (lead_time, lat, lon)
      3. 沿 year 维度 concat 后再平均 → 气候态 (lead_time, lat, lon)

    输出 365 个文件，含完整坐标和属性。

    返回
    -------
    dict  {mmdd: Path}  365 个气候态文件路径
    """
    if output_dir is None:
        output_dir = CLIM_OUTPUT_DIR

    clim_years = list(range(clim_start_year, clim_end_year + 1))

    log.info("=" * 60)
    log.info("第1步 CLIM | 逐日起报气候态 [%s]  (%d-%d, %d 年)",
             VAR_NAME, clim_start_year, clim_end_year, len(clim_years))
    log.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    skipped = 0

    for idx, mmdd in enumerate(ALL_MMDD):
        out_path = output_dir / f"{mmdd}.nc"

        if out_path.exists():
            generated[mmdd] = out_path
            skipped += 1
            if (idx + 1) % 60 == 0:
                log.info("  [%3d/365] %s — 已存在，跳过", idx + 1, mmdd)
            continue

        # 收集所有年份该日起报数据（member 平均后）
        member_means = []
        missing_years = []
        for year in clim_years:
            nc_path = RAW_DATA_DIR / f"{year}{mmdd}.nc"
            if not nc_path.exists():
                missing_years.append(year)
                continue
            try:
                ds = xr.open_dataset(nc_path)
                da = ds[info["var_name"]].copy()
                ds.close()

                da = normalize_dimensions(da, info)
                da_mean = compute_member_mean(da)          # → (lead_time, lat, lon)
                member_means.append(da_mean)
            except Exception as e:
                log.warning("  读取 %s 失败: %s", nc_path.name, e)
                missing_years.append(year)

        if missing_years:
            log.warning("  [WARN] mmdd=%s: 缺失年份 %s, 使用 %d/%d 年",
                        mmdd, missing_years, len(member_means), len(clim_years))

        if not member_means:
            log.error("  [ERROR] mmdd=%s: 所有年份均缺失，跳过！", mmdd)
            continue

        # 多年平均
        if len(member_means) == 1:
            da_clim = member_means[0]
        else:
            da_clim = xr.concat(member_means, dim="clim_year").mean(dim="clim_year")

        out_var = OUTPUT_VAR_NAME if OUTPUT_VAR_NAME else info["var_name"]
        save_dataset(
            da_clim, out_path, var_name=out_var,
            long_name=f"{get_var_attr('long_name', VAR_NAME)} Daily Climatology",
            units=get_var_attr("units"),
            description=(
                f"{VAR_NAME} daily climatology for init date {mmdd}, "
                f"averaged over {clim_start_year}-{clim_end_year} "
                f"({len(member_means)} years, member-mean first)."
            ),
            extra_attrs={
                "step": "CLIM",
                "climatology_period": f"{clim_start_year}-{clim_end_year}",
                "mmdd": mmdd,
                "years_used": len(member_means),
                "years_missing": str(missing_years) if missing_years else "none",
            },
        )
        generated[mmdd] = out_path

        if (idx + 1) % 60 == 0:
            log.info("  [%3d/365] %s — 完成", idx + 1, mmdd)

    log.info("  CLIM 完成: 生成 %d, 跳过 %d", len(generated) - skipped, skipped)
    return generated


# ══════════════════════════════════════════════════════════════════════════
#  第2步：计算 2024 年逐日起报距平
# ══════════════════════════════════════════════════════════════════════════

def compute_anomaly_for_year(
    info: dict,
    clim_files: dict,
    anom_year: int = ANOM_YEAR,
    output_dir: Path = None,
) -> dict:
    """
    计算指定年份的逐日起报距平。

    对每个起报日期 mmdd：
      1. 读取 {anom_year}{mmdd}.nc → member 平均 → (lead_time, lat, lon)
      2. 读取气候态文件 clim_{mmdd}.nc
      3. 距平 = 2024 member 均值 - 气候态

    返回
    -------
    dict  {mmdd: Path}  365 个距平文件路径
    """
    if output_dir is None:
        output_dir = ANOM_OUTPUT_DIR

    log.info("=" * 60)
    log.info("第2步 ANOM | 逐日起报距平 [%s]  (%d)", VAR_NAME, anom_year)
    log.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    skipped = 0

    for idx, mmdd in enumerate(ALL_MMDD):
        out_path = output_dir / f"{anom_year}{mmdd}.nc"

        if out_path.exists():
            generated[mmdd] = out_path
            skipped += 1
            if (idx + 1) % 60 == 0:
                log.info("  [%3d/365] %s — 已存在，跳过", idx + 1, mmdd)
            continue

        # ── 读取 2024 年起报文件 ──
        fcst_path = RAW_DATA_DIR / f"{anom_year}{mmdd}.nc"
        if not fcst_path.exists():
            log.warning("  [WARN] %s 缺失，跳过", fcst_path.name)
            continue

        try:
            ds_fcst = xr.open_dataset(fcst_path)
            da_fcst = ds_fcst[info["var_name"]].copy()
            ds_fcst.close()

            da_fcst = normalize_dimensions(da_fcst, info)
            da_fcst_mean = compute_member_mean(da_fcst)     # → (lead_time, lat, lon)
        except Exception as e:
            log.error("  [ERROR] 读取预报 %s: %s", fcst_path.name, e)
            continue

        # ── 读取对应气候态 ──
        if mmdd not in clim_files:
            log.warning("  [WARN] mmdd=%s 气候态缺失，跳过 %s", mmdd, fcst_path.name)
            continue

        try:
            ds_clim = xr.open_dataset(clim_files[mmdd])
            clim_var = list(ds_clim.data_vars)[0]
            da_clim = ds_clim[clim_var].copy()
            ds_clim.close()
        except Exception as e:
            log.error("  [ERROR] 读取气候态 %s: %s", clim_files[mmdd].name, e)
            continue

        # ── 计算距平 ──
        da_anom = da_fcst_mean - da_clim

        out_var = OUTPUT_VAR_NAME if OUTPUT_VAR_NAME else info["var_name"]
        save_dataset(
            da_anom, out_path, var_name=out_var,
            long_name=f"{get_var_attr('long_name', VAR_NAME)} Anomaly",
            units=get_var_attr("units"),
            description=(
                f"{VAR_NAME} anomaly for init date {anom_year}{mmdd}. "
                f"Computed as {anom_year} forecast (member mean) "
                f"minus {CLIM_START_YEAR}-{CLIM_END_YEAR} daily climatology."
            ),
            extra_attrs={
                "step": "ANOM",
                "anomaly_method": f"forecast({anom_year})_member_mean - climatology({CLIM_START_YEAR}-{CLIM_END_YEAR})",
                "climatology_period": f"{CLIM_START_YEAR}-{CLIM_END_YEAR}",
                "forecast_year": anom_year,
                "init_date": f"{anom_year}{mmdd}",
            },
        )
        generated[mmdd] = out_path

        if (idx + 1) % 60 == 0:
            log.info("  [%3d/365] %s — 完成", idx + 1, mmdd)

    log.info("  ANOM 完成: 生成 %d, 跳过 %d", len(generated) - skipped, skipped)
    return generated


# ══════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数：变量、年份、路径均可由编排器传入。"""
    parser = argparse.ArgumentParser(description="my_model 模式数据处理")
    parser.add_argument("--var", type=str, default=VAR_NAME, help="变量名")
    parser.add_argument("--output-var", type=str, default=OUTPUT_VAR_NAME, help="输出变量名")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DATA_DIR), help="原始数据根目录")
    parser.add_argument("--clim-dir", type=str, default=str(CLIM_OUTPUT_DIR), help="气候态输出目录")
    parser.add_argument("--anom-dir", type=str, default=str(ANOM_OUTPUT_DIR), help="距平输出目录")
    parser.add_argument("--clim-start", type=int, default=CLIM_START_YEAR, help="气候态起始年")
    parser.add_argument("--clim-end", type=int, default=CLIM_END_YEAR, help="气候态结束年")
    parser.add_argument("--anom-year", type=int, default=ANOM_YEAR, help="距平年份")
    return parser.parse_args()


def main():
    """主入口：数据结构检测 → 气候态 → 距平。"""
    args = parse_args()
    global VAR_NAME, OUTPUT_VAR_NAME, RAW_DATA_DIR, CLIM_OUTPUT_DIR, ANOM_OUTPUT_DIR
    global CLIM_START_YEAR, CLIM_END_YEAR, ANOM_YEAR
    VAR_NAME = args.var
    OUTPUT_VAR_NAME = args.output_var
    RAW_DATA_DIR = Path(args.raw_dir)
    CLIM_OUTPUT_DIR = Path(args.clim_dir)
    ANOM_OUTPUT_DIR = Path(args.anom_dir)
    CLIM_START_YEAR = args.clim_start
    CLIM_END_YEAR = args.clim_end
    ANOM_YEAR = args.anom_year
    log.info("=" * 60)
    log.info("my_model  逐日起报气候态 & 距平计算")
    log.info("  变量:       %s", VAR_NAME)
    log.info("  原始数据:   %s", RAW_DATA_DIR)
    log.info("  气候态年份: %d-%d", CLIM_START_YEAR, CLIM_END_YEAR)
    log.info("  距平年份:   %d", ANOM_YEAR)
    log.info("  气候态输出: %s", CLIM_OUTPUT_DIR)
    log.info("  距平输出:   %s", ANOM_OUTPUT_DIR)
    log.info("  启动时间:   %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    # ── 0. 数据结构自动检测 ──
    log.info("\n── 数据结构自动检测 ──")

    # 查找第一个存在的文件作为样本
    sample_file = None
    for year in range(CLIM_START_YEAR, CLIM_END_YEAR + 1):
        f = RAW_DATA_DIR / f"{year}0101.nc"
        if f.exists():
            sample_file = f; break
    if sample_file is None:
        # 尝试 2024 年
        f = RAW_DATA_DIR / f"{ANOM_YEAR}0101.nc"
        if f.exists():
            sample_file = f

    if sample_file is None:
        log.error("无法找到示例 NC 文件！请检查 RAW_DATA_DIR 路径。")
        sys.exit(1)

    log.info("示例文件: %s", sample_file)
    info = inspect_nc_file(sample_file)

    if not info or "var_name" not in info:
        log.error("无法解析 NC 文件结构，退出。")
        sys.exit(1)

    # 验证结构
    warnings_list = validate_structure(info, sample_file)
    if warnings_list:
        for w in warnings_list:
            log.warning("  [结构异常] %s", w)
    else:
        log.info("  结构验证通过: member=%d, lead_time=%d, lat=%d, lon=%d",
                 info["dims"].get(info.get("member_name", "member"), 0),
                 info["dims"].get(info.get("lead_time_name", "lead_time"), 0),
                 info.get("lat_size", 0), info.get("lon_size", 0))

    # 打印变量属性
    log.info("  变量属性: %s", info.get("var_attrs", {}))
    log.info("  全局属性: %s", info.get("attrs", {}))

    # ── 第1步：计算气候态 ──
    clim_files = compute_daily_climatology(info)

    # ── 第2步：计算 2024 距平 ──
    anom_files = compute_anomaly_for_year(info, clim_files)

    # ── 汇总 ──
    log.info("\n" + "=" * 60)
    log.info("全部处理完成！")
    log.info("  气候态: %d 天 (→ %s)", len(clim_files), CLIM_OUTPUT_DIR)
    log.info("  距平:   %d 天 (→ %s)", len(anom_files), ANOM_OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════
#  使用说明（单独运行该代码时）
# ══════════════════════════════════════════════════════════════════════════
#
# 【1. 如何运行脚本】
#     python my_model_anom.py
#
# 【2. 如何修改变量名和路径】
#     修改顶部全局配置区：
#       VAR_NAME        — 变量名，如 "t2m" → "tp" / "ssr" / "ssrd"
#       OUTPUT_VAR_NAME — 输出变量名，None 则沿用 VAR_NAME
#       RAW_DATA_DIR    — 原始 NC 文件目录
#       CLIM_OUTPUT_DIR — 气候态输出目录
#       ANOM_OUTPUT_DIR — 距平输出目录
#       CLIM_START_YEAR / CLIM_END_YEAR — 气候态年份范围
#       ANOM_YEAR       — 距平年份
#     如需新增变量属性，在 VAR_ATTRS 字典中添加一行。
#
# 【3. 气候态文件的输出维度】
#     每个气候态文件 (e.g. 0101.nc):
#       lead_time : 60      (预报时效 1~60 天)
#       lat       : 121     (全球 1.5° 纬度)
#       lon       : 240     (全球 1.5° 经度)
#     变量名: t2m (或 OUTPUT_VAR_NAME 指定的名称)
#     共 365 个文件（排除 02-29）
#
# 【4. 距平文件的输出维度】
#     每个距平文件 (e.g. 20240101.nc):
#       lead_time : 60      (预报时效 1~60 天)
#       lat       : 121     (全球 1.5° 纬度)
#       lon       : 240     (全球 1.5° 经度)
#     变量名: t2m (或 OUTPUT_VAR_NAME 指定的名称)
#     共 365 个文件（排除 02-29）
#
# 【5. 缺失文件的默认处理方式】
#     - 气候态计算中：某个日期的某一年文件缺失 → WARNING，用已有年份计算平均
#     - 气候态计算中：某个日期所有年份均缺失 → ERROR，跳过该日期
#     - 距平计算中：2024 年起报文件缺失 → WARNING，跳过该日期
#     - 距平计算中：对应气候态文件缺失 → WARNING，跳过该日期
#     - 02-29 自动跳过，不参与任何计算
#     - 已存在的输出文件自动跳过（断点续跑）
#
# 【6. 处理逻辑说明】
#     ┌──────────────────────────────────────────────────┐
#     │ 原始数据：t2m(member=11, lead_time=60, lat=121, lon=240) │
#     │   ↓ member 平均                                    │
#     │ t2m_mean(lead_time=60, lat=121, lon=240)           │
#     │   ↓ 按 mmdd 分组，多年平均                          │
#     │ 气候态：t2m_clim_mmdd(lead_time=60, lat=121, lon=240) │
#     │   ↓ 2024 mmdd - 气候态 mmdd                        │
#     │ 距平：t2m_anom_mmdd(lead_time=60, lat=121, lon=240)  │
#     └──────────────────────────────────────────────────┘
