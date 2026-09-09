"""
CRA v1.5 MJO 数据完整处理流水线。

该脚本合并以下两个原脚本的功能，但不依赖也不修改原脚本：
  1. mjo_CRA_v1.5_process.py
  2. mjo_data_combine_CRA（略有不同）.py

处理流程：
  1. CLIM  — 计算 2004-2023 年逐日气候态
  2. ANOM  — 计算 2023、2024 年逐日距平
  3. MERGE — 插值到全球 2.5° 网格，沿 time 拼接并合并三个变量

输入变量：ttr、u（从 u 中分别提取 u200 和 u850）
最终输出：olr、u200、u850，维度通常为 (time, lat=73, lon=144)

注意：
  1. 默认排除 02-29，与原有两份脚本保持一致。
  2. u200/u850 默认按 isobaricInhPa 的位置索引选层，运行前必须确认索引正确。
  3. ttr 在合并阶段仅重命名为 olr，不取反，与原 CRA 合并脚本保持一致。
"""

from datetime import datetime, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  全局配置
# ══════════════════════════════════════════════════════════════════════

PROCESS_VARS = ["ttr", "u200", "u850"]

CLIM_YEARS = list(range(2004, 2024))
ANOM_YEARS = [2023, 2024]

VAR_CONFIG = {
    "ttr": {                                                        ##变量格式：ttr(latitude=121, longitude=240)，空间分辨率1.5°
        "raw_dir": "/gpu/zhouchg/zhouchg/CRA/1P5/netcdf/ttr",
        "nc_var": "ttr",
        "level": None,
        "expected_pressure": None,
        "long_name": "TOA Thermal Radiation",
        "units": "W m**-2",
    },
    "u200": {                                                       ##变量格式：u200(latitude=121, longitude=240)，空间分辨率1.5°
        "raw_dir": "/gpu/zhouchg/zhouchg/CRA/1P5/netcdf/u",
        "nc_var": "u",
        "level": 10,
        "expected_pressure": 200,
        "long_name": "Zonal Wind at 200 hPa",
        "units": "m s**-1",
    },
    "u850": {                                                       ##变量格式：u850(latitude=121, longitude=240)，空间分辨率1.5°
        "raw_dir": "/gpu/zhouchg/zhouchg/CRA/1P5/netcdf/u",
        "nc_var": "u",
        "level": 3,
        "expected_pressure": 850,
        "long_name": "Zonal Wind at 850 hPa",
        "units": "m s**-1",
    },
}

# 中间结果：各变量的气候态和逐日距平。
INTERMEDIATE_DIR = Path("/gpu/zhouchg/liuzh/MJO/CRA_V1.5")           ## 气候态输出格式：ttr(latitude=121, longitude=240)，空间分辨率1.5°
                                                                                    ## u200(latitude=121, longitude=240)，空间分辨率1.5°
                                                                                    ## u850(latitude=121, longitude=240)，空间分辨率1.5°
                                                                    ##距平输出格式（还未插值）：ttr(latitude=121, longitude=240)，空间分辨率1.5°
                                                                                    ## u200(latitude=121, longitude=240)，空间分辨率1.5°
                                                                                    ## u850(latitude=121, longitude=240)，空间分辨率1.5°
# 最终合并结果，可直接供 A_00_cal_S2S-ecmf_mjo_20240920_optimized.py 使用。
FINAL_OUTPUT_FILE = Path(
    "/gpu/zhouchg/liuzh/MJO/CRA_V1.5_output/"
    "CRA_anom_daily_var3_2p5_2023_2024.nc"
)                                                                       ##olr(time=730, lat=73, lon=144)，空间分辨率2.5°，时间分辨率1天
                                                                        ##u200(time=730, lat=73, lon=144)，空间分辨率2.5°，时间分辨率1天
                                                                        ##u850(time=730, lat=73, lon=144)，空间分辨率2.5°，时间分辨率1天

TARGET_LAT = np.arange(90.0, -90.1, -2.5)
TARGET_LON = np.arange(0.0, 360.0, 2.5)

OVERWRITE_INTERMEDIATE = False
OVERWRITE_FINAL = True

INTERMEDIATE_COMPRESSION = {"zlib": True, "complevel": 4}
FINAL_COMPRESSION = {
    "dtype": "float32",
    "zlib": True,
    "complevel": 5,
}


# ══════════════════════════════════════════════════════════════════════
#  通用工具
# ══════════════════════════════════════════════════════════════════════

def log(message: str, level: str = "INFO") -> None:
    """输出带时间戳的日志。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def generate_non_leap_mmdd_list(start_year: int = 2001) -> list[str]:
    """生成不含 02-29 的 365 天月日列表。"""
    result = []
    current = datetime(start_year, 1, 1)
    for _ in range(365):
        result.append(current.strftime("%m%d"))
        current += timedelta(days=1)
    return result


ALL_MMDD = generate_non_leap_mmdd_list()


def find_nc_file(var_name: str, year: int, mmdd: str) -> Path:
    """构造一个 CRA 原始日文件的路径。"""
    return Path(VAR_CONFIG[var_name]["raw_dir"]) / str(year) / f"{year}{mmdd}.nc"


def read_raw_data(var_name: str, year: int, mmdd: str) -> xr.DataArray:
    """读取二维日场；风场按配置的位置索引提取气压层。"""
    nc_path = find_nc_file(var_name, year, mmdd)
    if not nc_path.exists():
        raise FileNotFoundError(f"文件不存在: {nc_path}")

    config = VAR_CONFIG[var_name]
    with xr.open_dataset(nc_path) as dataset:
        data = dataset[config["nc_var"]]
        if config["level"] is not None:
            level_dim = "isobaricInhPa"
            if level_dim not in data.dims:
                raise KeyError(f"{nc_path} 中缺少维度 {level_dim}")

            level_index = config["level"]
            if level_dim in data.coords:
                actual_pressure = data[level_dim].values[level_index]
                expected_pressure = config["expected_pressure"]
                if expected_pressure is not None and not np.isclose(
                    float(actual_pressure), float(expected_pressure)
                ):
                    log(
                        f"{var_name}: {level_dim}[{level_index}]={actual_pressure}，"
                        f"预期 {expected_pressure} hPa，请检查层索引",
                        "WARN",
                    )

            data = data.isel({level_dim: level_index})

        return data.load()


def save_dataset(
    data: xr.DataArray,
    output_path: Path,
    var_name: str,
    long_name: str,
    units: str,
    description: str,
    extra_attrs: dict | None = None,
) -> None:
    """保存单变量 NetCDF 中间文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = data.to_dataset(name=var_name)
    dataset[var_name].attrs.update({
        "long_name": long_name,
        "units": units,
        "description": description,
    })
    dataset.attrs.update({
        "source": "CRA Reanalysis / FDP Project",
        "var_name": var_name,
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
    })
    if extra_attrs:
        dataset.attrs.update(extra_attrs)
    dataset.to_netcdf(
        output_path,
        encoding={var_name: INTERMEDIATE_COMPRESSION},
    )


# ══════════════════════════════════════════════════════════════════════
#  第1步：CLIM — 逐日气候态
# ══════════════════════════════════════════════════════════════════════

def compute_daily_climatology(var_name: str) -> dict[str, Path]:
    """计算 2004-2023 年同月日算术平均，输出 365 个气候态文件。"""
    config = VAR_CONFIG[var_name]
    output_dir = INTERMEDIATE_DIR / var_name / "clim"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    log(f"第1步 CLIM [{var_name}]：{CLIM_YEARS[0]}-{CLIM_YEARS[-1]}")
    for index, mmdd in enumerate(ALL_MMDD, start=1):
        output_path = output_dir / f"{mmdd}.nc"
        if output_path.exists() and not OVERWRITE_INTERMEDIATE:
            generated[mmdd] = output_path
            continue

        yearly_data = []
        missing_years = []
        for year in CLIM_YEARS:
            try:
                yearly_data.append(read_raw_data(var_name, year, mmdd))
            except FileNotFoundError:
                missing_years.append(year)

        if not yearly_data:
            log(f"{var_name} {mmdd}: 所有气候态年份均缺失", "ERROR")
            continue
        if missing_years:
            log(f"{var_name} {mmdd}: 缺失年份 {missing_years}", "WARN")

        climatology = xr.concat(yearly_data, dim="year").mean("year")
        description = (
            f"{var_name} daily climatology for {mmdd}, averaged over "
            f"{CLIM_YEARS[0]}-{CLIM_YEARS[-1]} ({len(yearly_data)} years)."
        )
        save_dataset(
            climatology,
            output_path,
            var_name,
            f"{config['long_name']} Daily Climatology",
            config["units"],
            description,
            {
                "step": "CLIM",
                "climatology_period": f"{CLIM_YEARS[0]}-{CLIM_YEARS[-1]}",
                "mmdd": mmdd,
                "years_used": len(yearly_data),
                "years_missing": str(missing_years) if missing_years else "none",
            },
        )
        generated[mmdd] = output_path

        if index % 50 == 0:
            log(f"{var_name} CLIM: {index}/365")

    return generated


# ══════════════════════════════════════════════════════════════════════
#  第2步：ANOM — 逐日距平
# ══════════════════════════════════════════════════════════════════════

def compute_daily_anomaly(
    var_name: str,
    climatology_files: dict[str, Path],
) -> dict[str, Path]:
    """计算指定年份的逐日距平：原始日值减去同月日气候态。"""
    config = VAR_CONFIG[var_name]
    output_dir = INTERMEDIATE_DIR / var_name / "anom"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    log(f"第2步 ANOM [{var_name}]：{ANOM_YEARS}")
    for year in ANOM_YEARS:
        for index, mmdd in enumerate(ALL_MMDD, start=1):
            date_key = f"{year}{mmdd}"
            output_path = output_dir / f"{date_key}.nc"
            if output_path.exists() and not OVERWRITE_INTERMEDIATE:
                generated[date_key] = output_path
                continue
            if mmdd not in climatology_files:
                log(f"{date_key}: 缺少 {var_name} 气候态", "WARN")
                continue

            try:
                raw_data = read_raw_data(var_name, year, mmdd)
            except FileNotFoundError:
                log(f"{date_key}: 缺少 {var_name} 原始文件", "WARN")
                continue

            with xr.open_dataset(climatology_files[mmdd]) as dataset:
                climatology = dataset[var_name].load()
            anomaly = raw_data - climatology

            description = (
                f"{var_name} anomaly for {date_key}. ANOM = raw({year}) - "
                f"climatology({CLIM_YEARS[0]}-{CLIM_YEARS[-1]})."
            )
            save_dataset(
                anomaly,
                output_path,
                var_name,
                f"{config['long_name']} Anomaly",
                config["units"],
                description,
                {
                    "step": "ANOM",
                    "year": year,
                    "mmdd": mmdd,
                    "date": date_key,
                },
            )
            generated[date_key] = output_path

            if index % 50 == 0:
                log(f"{var_name} ANOM {year}: {index}/365")

    return generated


# ══════════════════════════════════════════════════════════════════════
#  第3步：MERGE — 2.5°插值、时间拼接和三变量合并
# ══════════════════════════════════════════════════════════════════════

def normalize_spatial_coordinates(data: xr.DataArray) -> xr.DataArray:
    """将空间维统一为 lat/lon，并将经度统一到 0-360。"""
    rename_map = {}
    if "latitude" in data.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in data.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        data = data.rename(rename_map)

    if "lat" not in data.dims or "lon" not in data.dims:
        raise KeyError(f"找不到 lat/lon 维度，实际维度为 {data.dims}")

    if np.any(data["lon"].values < 0):
        data = data.assign_coords(lon=(data["lon"].values + 360.0) % 360.0)
    return data.sortby("lat").sortby("lon")


def merge_anomaly_files(
    anomaly_files_by_var: dict[str, dict[str, Path]],
) -> Path:
    """将三个变量的逐日距平插值到2.5°并合并为一个时间序列文件。"""
    expected_dates = [f"{year}{mmdd}" for year in ANOM_YEARS for mmdd in ALL_MMDD]
    common_dates = [
        date_key
        for date_key in expected_dates
        if all(date_key in anomaly_files_by_var[var] for var in PROCESS_VARS)
    ]

    missing_count = len(expected_dates) - len(common_dates)
    if missing_count:
        log(f"三个变量共同可用 {len(common_dates)}/{len(expected_dates)} 天", "WARN")
    if not common_dates:
        raise RuntimeError("没有三个变量均完整的日期，无法生成最终合并文件")

    final_variables = {}
    for var_name in PROCESS_VARS:
        log(f"第3步 MERGE [{var_name}]：插值到2.5°并沿 time 拼接")
        daily_arrays = []
        for index, date_key in enumerate(common_dates, start=1):
            with xr.open_dataset(anomaly_files_by_var[var_name][date_key]) as dataset:
                data = dataset[var_name].load()

            data = normalize_spatial_coordinates(data)
            data = data.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")

            # 原始日文件可能带有随日期变化的 valid_time 等非维度坐标。
            # 新的 time 坐标由 date_key 统一创建，这些辅助坐标必须先删除，
            # 否则 xr.concat 会因不同日期的 valid_time 值冲突而失败。
            data = data.reset_coords(drop=True)
            
            # 风场选层后可能残留标量气压坐标，合并变量前删除以免坐标冲突。
            for coord_name in ("level", "isobaricInhPa"):
                if coord_name in data.coords:
                    data = data.drop_vars(coord_name)

            output_name = "olr" if var_name == "ttr" else var_name
            data = data.rename(output_name)
            valid_time = pd.to_datetime(date_key, format="%Y%m%d")
            data = data.expand_dims(time=[valid_time])
            daily_arrays.append(data)

            if index % 100 == 0:
                log(f"{var_name} MERGE: {index}/{len(common_dates)}")

        output_name = "olr" if var_name == "ttr" else var_name
        final_variables[output_name] = xr.concat(daily_arrays, dim="time")

    final_dataset = xr.Dataset(final_variables)
    final_dataset.attrs.update({
        "description": "CRA anomaly data interpolated to 2.5 degree grid",
        "source": "CRA v1.5",
        "variables": "olr, u200, u850",
        "time_range": f"{ANOM_YEARS[0]}-{ANOM_YEARS[-1]}",
        "grid": "2.5 degree, lat=73, lon=144",
        "ttr_to_olr": "renamed only; sign unchanged",
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    FINAL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if FINAL_OUTPUT_FILE.exists() and not OVERWRITE_FINAL:
        log(f"最终文件已存在，跳过: {FINAL_OUTPUT_FILE}")
        return FINAL_OUTPUT_FILE

    encoding = {name: FINAL_COMPRESSION for name in final_variables}
    final_dataset.to_netcdf(FINAL_OUTPUT_FILE, encoding=encoding)
    log(f"最终文件已保存: {FINAL_OUTPUT_FILE}")
    return FINAL_OUTPUT_FILE


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    """依次执行 CLIM、ANOM 和 MERGE 三个阶段。"""
    log("=" * 70)
    log("CRA v1.5 MJO 完整流水线: CLIM → ANOM → MERGE 2.5°")
    log(f"处理变量: {PROCESS_VARS}")
    log(f"气候态年份: {CLIM_YEARS[0]}-{CLIM_YEARS[-1]}")
    log(f"距平年份: {ANOM_YEARS}")
    log(f"中间目录: {INTERMEDIATE_DIR}")
    log(f"最终文件: {FINAL_OUTPUT_FILE}")
    log("=" * 70)

    anomaly_files_by_var = {}
    for var_name in PROCESS_VARS:
        log("\n" + "█" * 60)
        log(f"开始处理变量: {var_name}")
        log("█" * 60)

        sample_file = find_nc_file(var_name, CLIM_YEARS[0], "0101")
        if not sample_file.exists():
            raise FileNotFoundError(f"示例文件不存在: {sample_file}")

        climatology_files = compute_daily_climatology(var_name)
        anomaly_files = compute_daily_anomaly(var_name, climatology_files)
        anomaly_files_by_var[var_name] = anomaly_files
        log(
            f"{var_name} 完成: CLIM={len(climatology_files)}, "
            f"ANOM={len(anomaly_files)}"
        )

    output_file = merge_anomaly_files(anomaly_files_by_var)
    log("=" * 70)
    log("全部处理完成")
    log(f"可供 MJO 主程序读取: {output_file}")
    log("=" * 70)


if __name__ == "__main__":
    main()
