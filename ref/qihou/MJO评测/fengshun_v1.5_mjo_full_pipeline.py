"""
Fengshun v1.5 MJO 三变量完整处理流水线。

处理流程：
  1. CLIM  — 对每个文件做 member 平均，再计算 2004-2023 年模式气候态
  2. ANOM  — 2024 年 member 平均预报减去对应起报日的模式气候态
  3. MERGE — 将距平插值到 2.5°，添加 time，并合并 olr/u200/u850

最终输出维度：
  olr(time, lead_time=60, lat=73, lon=144)
  u200(time, lead_time=60, lat=73, lon=144)
  u850(time, lead_time=60, lat=73, lon=144)

注意：
  1. 三个变量 ttr、u200、u850 会在一次运行中全部处理。
  2. ttr 在最终合并阶段按原脚本约定执行 olr = -ttr。
  3. 默认排除 02-29，与原有脚本保持一致。
  4. 中间气候态和距平文件已存在时默认跳过，支持断点续跑。
"""

from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr


# ══════════════════════════════════════════════════════════════════════
#  日志配置
# ══════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  全局配置 —— 迁移服务器或调整年份时主要修改这里
# ══════════════════════════════════════════════════════════════════════

# 固定同时处理 MJO 所需的三个变量。
VAR_NAMES = ["ttr", "u200", "u850"]

# 原始集合预报目录结构：RAW_DATA_BASE_DIR/<var>/yyyymmdd.nc
RAW_DATA_BASE_DIR = Path(                                               ##变量格式：ttr(member=11, lead_time=60, lat=121, lon=240)，空间分辨率1.5°
    "/gpu/zhouchg/wangchp/fengshun_v1.5_u200_u850_ttr_2004-2024"              ##  u200(member=11, lead_time=60, lat=121, lon=240)，空间分辨率1.5°
)                                                                             ##  u850(member=11, lead_time=60, lat=121, lon=240)，空间分辨率1.5°

# 中间文件目录结构：
#   INTERMEDIATE_OUTPUT_DIR/<var>/clim_1p5/mmdd.nc
#   INTERMEDIATE_OUTPUT_DIR/<var>/anom_1p5/yyyymmdd.nc
INTERMEDIATE_OUTPUT_DIR = Path("/gpu/zhouchg/liuzh/MJO/fengshun_v1.5")      ##变量格式（member已平均，还未插值）：ttr(lead_time=60, lat=121, lon=240)，空间分辨率1.5°
                                                                            ## u200(lead_time=60, lat=121, lon=240)，空间分辨率1.5°
                                                                            ## u850(lead_time=60, lat=121, lon=240)，空间分辨率1.5°

# 最终三变量合并文件，可供 A_00_cal_S2S-ecmf_mjo_20240920_optimized.py 读取。
FINAL_OUTPUT_FILE = Path("/gpu/zhouchg/liuzh/MJO/fengshun_v1.5_output/fengshun1.5_anom_daily_var3_2p5_2024.nc")    
                                                                            ##输出格式：olr(time=365, lead_time=60, lat=73, lon=144)，空间分辨率2.5°
                                                                                    ## u200(time=365, lead_time=60, lat=73, lon=144)，空间分辨率2.5°
                                                                                    ## u850(time=365, lead_time=60, lat=73, lon=144)，空间分辨率2.5°

CLIM_START_YEAR = 2004
CLIM_END_YEAR = 2023
ANOM_YEAR = 2024

# 原始文件的预期结构；不匹配时给出警告。
EXPECTED_MEMBER_N = 11
EXPECTED_LEAD_TIME_N = 60
EXPECTED_LAT_N = 121
EXPECTED_LON_N = 240

# 最终输出的全球 2.5° 网格。
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

VAR_ATTRS = {
    "ttr": {
        "long_name": "Top-of-Atmosphere Thermal Radiation",
        "units": "W m**-2",
    },
    "u200": {
        "long_name": "Zonal Wind at 200 hPa",
        "units": "m s**-1",
    },
    "u850": {
        "long_name": "Zonal Wind at 850 hPa",
        "units": "m s**-1",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  日期和文件工具
# ══════════════════════════════════════════════════════════════════════

def generate_dates(year: int = 2001) -> list[str]:
    """生成平年365天的 mmdd 列表，不包含02-29。"""
    dates = []
    current = datetime(year, 1, 1)
    for _ in range(365):
        dates.append(current.strftime("%m%d"))
        current += timedelta(days=1)
    return dates


ALL_MMDD = generate_dates()


def find_sample_file(raw_data_dir: Path) -> Path | None:
    """优先从气候态年份中寻找一个01月01日文件作为结构样本。"""
    for year in range(CLIM_START_YEAR, CLIM_END_YEAR + 1):
        candidate = raw_data_dir / f"{year}0101.nc"
        if candidate.exists():
            return candidate

    candidate = raw_data_dir / f"{ANOM_YEAR}0101.nc"
    return candidate if candidate.exists() else None


# ══════════════════════════════════════════════════════════════════════
#  NC结构检测与维度归一化
# ══════════════════════════════════════════════════════════════════════

def inspect_nc_file(file_path: Path, expected_var: str) -> dict:
    """检测变量名、空间维、时效维和成员维。"""
    with xr.open_dataset(file_path) as dataset:
        info = {
            "dims": dict(dataset.sizes),
            "data_vars": list(dataset.data_vars),
            "coords": list(dataset.coords),
            "attrs": dict(dataset.attrs),
        }

        candidates = {
            "lat_name": ["lat", "latitude", "LAT", "Latitude"],
            "lon_name": ["lon", "longitude", "LON", "Longitude"],
            "lead_time_name": ["lead_time", "leadtime", "lead", "LEAD_TIME", "step", "time"],
            "member_name": ["member", "ensemble", "ens", "MEMBER", "realization"],
        }
        for key, names in candidates.items():
            for name in names:
                if name in dataset.coords or name in dataset.dims:
                    info[key] = name
                    break

        actual_var = expected_var if expected_var in dataset.data_vars else None
        if actual_var is None:
            if not dataset.data_vars:
                raise ValueError(f"{file_path} 中没有数据变量")
            actual_var = list(dataset.data_vars)[0]
            log.warning(
                "未找到变量 '%s'，示例文件中暂时使用 '%s'",
                expected_var,
                actual_var,
            )
        info["var_name"] = actual_var
        info["var_attrs"] = dict(dataset[actual_var].attrs)

        lat_name = info.get("lat_name")
        lon_name = info.get("lon_name")
        if lat_name:
            lat_values = dataset[lat_name].values
            info["lat_size"] = len(lat_values)
            info["lat_is_descending"] = (
                len(lat_values) > 1 and lat_values[0] > lat_values[-1]
            )
        if lon_name:
            lon_values = dataset[lon_name].values
            info["lon_size"] = len(lon_values)
            info["lon_min"] = float(lon_values.min())
            info["lon_max"] = float(lon_values.max())

    log.info(
        "  文件: %s | 维度: %s | 变量: %s | 坐标: %s",
        file_path.name,
        info["dims"],
        info["data_vars"],
        info["coords"],
    )
    return info


def validate_structure(info: dict, file_path: Path) -> None:
    """检查样本结构；异常只警告，真正无法处理时由后续代码报错。"""
    checks = [
        ("member_name", EXPECTED_MEMBER_N, "member"),
        ("lead_time_name", EXPECTED_LEAD_TIME_N, "lead_time"),
    ]
    for key, expected_size, label in checks:
        dim_name = info.get(key)
        actual_size = info["dims"].get(dim_name) if dim_name else None
        if actual_size != expected_size:
            log.warning(
                "%s: %s=%s，预期%s",
                file_path.name,
                label,
                actual_size,
                expected_size,
            )

    for key, expected_size, label in [
        ("lat_size", EXPECTED_LAT_N, "lat"),
        ("lon_size", EXPECTED_LON_N, "lon"),
    ]:
        if info.get(key) != expected_size:
            log.warning(
                "%s: %s=%s，预期%s",
                file_path.name,
                label,
                info.get(key),
                expected_size,
            )


def normalize_dimensions(data: xr.DataArray, info: dict) -> xr.DataArray:
    """统一为 member、lead_time、lat、lon，并规范经纬度坐标。"""
    lat_name = info.get("lat_name", "lat")
    lon_name = info.get("lon_name", "lon")
    lead_name = info.get("lead_time_name", "lead_time")
    member_name = info.get("member_name", "member")

    rename_map = {}
    if lat_name != "lat" and lat_name in data.dims:
        rename_map[lat_name] = "lat"
    if lon_name != "lon" and lon_name in data.dims:
        rename_map[lon_name] = "lon"
    if lead_name != "lead_time" and lead_name in data.dims:
        rename_map[lead_name] = "lead_time"
    if member_name != "member" and member_name in data.dims:
        rename_map[member_name] = "member"
    if rename_map:
        data = data.rename(rename_map)

    required_dims = {"lead_time", "lat", "lon"}
    missing_dims = required_dims.difference(data.dims)
    if missing_dims:
        raise KeyError(f"缺少必要维度 {sorted(missing_dims)}，实际维度为 {data.dims}")

    if np.any(data["lon"].values < 0):
        data = data.assign_coords(lon=(data["lon"].values + 360.0) % 360.0)

    data = data.sortby("lat").sortby("lon")
    target_dims = ["member", "lead_time", "lat", "lon"]
    ordered_dims = [name for name in target_dims if name in data.dims]
    extra_dims = [name for name in data.dims if name not in ordered_dims]
    return data.transpose(*(ordered_dims + extra_dims))


def read_forecast(file_path: Path, var_name: str, info: dict) -> xr.DataArray:
    """读取、加载并标准化一个集合预报文件。"""
    with xr.open_dataset(file_path) as dataset:
        actual_var = var_name if var_name in dataset.data_vars else info["var_name"]
        data = dataset[actual_var].load()
    return normalize_dimensions(data, info)


def compute_member_mean(data: xr.DataArray) -> xr.DataArray:
    """对member维做算术平均，输出 (lead_time, lat, lon)。"""
    return data.mean("member") if "member" in data.dims else data


# ══════════════════════════════════════════════════════════════════════
#  中间文件保存
# ══════════════════════════════════════════════════════════════════════

def save_intermediate_dataset(
    data: xr.DataArray,
    output_path: Path,
    var_name: str,
    description: str,
    extra_attrs: dict,
) -> None:
    """保存单变量气候态或距平文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = data.to_dataset(name=var_name)
    dataset[var_name].attrs.update({
        "long_name": VAR_ATTRS[var_name]["long_name"],
        "units": VAR_ATTRS[var_name]["units"],
        "description": description,
    })
    dataset.attrs.update({
        "source": "Fengshun v1.5 Ensemble Forecast System",
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        **extra_attrs,
    })
    dataset.to_netcdf(
        output_path,
        encoding={var_name: INTERMEDIATE_COMPRESSION},
    )


# ══════════════════════════════════════════════════════════════════════
#  第1步：CLIM — 逐日起报模式气候态
# ══════════════════════════════════════════════════════════════════════

def compute_daily_climatology(
    var_name: str,
    info: dict,
    raw_data_dir: Path,
) -> dict[str, Path]:
    """先做成员平均，再计算2004-2023年同一起报月日的多年平均。"""
    output_dir = INTERMEDIATE_OUTPUT_DIR / var_name / "clim_1p5"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    clim_years = list(range(CLIM_START_YEAR, CLIM_END_YEAR + 1))

    log.info("第1步 CLIM [%s]：%d-%d", var_name, CLIM_START_YEAR, CLIM_END_YEAR)
    for index, mmdd in enumerate(ALL_MMDD, start=1):
        output_path = output_dir / f"{mmdd}.nc"
        if output_path.exists() and not OVERWRITE_INTERMEDIATE:
            generated[mmdd] = output_path
            continue

        member_means = []
        missing_years = []
        for year in clim_years:
            input_path = raw_data_dir / f"{year}{mmdd}.nc"
            if not input_path.exists():
                missing_years.append(year)
                continue
            try:
                forecast = read_forecast(input_path, var_name, info)
                member_means.append(compute_member_mean(forecast))
            except Exception as error:
                log.warning("读取 %s 失败: %s", input_path, error)
                missing_years.append(year)

        if not member_means:
            log.error("%s %s: 所有气候态年份均缺失", var_name, mmdd)
            continue
        if missing_years:
            log.warning(
                "%s %s: 缺失年份%s，使用%d/%d年",
                var_name,
                mmdd,
                missing_years,
                len(member_means),
                len(clim_years),
            )

        climatology = xr.concat(member_means, dim="clim_year").mean("clim_year")
        climatology = climatology.assign_coords(
            lead_time=np.arange(1, climatology.sizes["lead_time"] + 1)
        )
        description = (
            f"{var_name} daily forecast climatology for init date {mmdd}, "
            f"member mean followed by {CLIM_START_YEAR}-{CLIM_END_YEAR} mean."
        )
        save_intermediate_dataset(
            climatology,
            output_path,
            var_name,
            description,
            {
                "step": "CLIM",
                "climatology_period": f"{CLIM_START_YEAR}-{CLIM_END_YEAR}",
                "mmdd": mmdd,
                "years_used": len(member_means),
                "years_missing": str(missing_years) if missing_years else "none",
            },
        )
        generated[mmdd] = output_path

        if index % 60 == 0:
            log.info("%s CLIM: %d/365", var_name, index)

    return generated


# ══════════════════════════════════════════════════════════════════════
#  第2步：ANOM — 2024逐日起报距平
# ══════════════════════════════════════════════════════════════════════

def compute_anomaly_for_year(
    var_name: str,
    info: dict,
    raw_data_dir: Path,
    climatology_files: dict[str, Path],
) -> dict[str, Path]:
    """计算2024年集合平均预报减去对应起报日模式气候态。"""
    output_dir = INTERMEDIATE_OUTPUT_DIR / var_name / "anom_1p5"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    log.info("第2步 ANOM [%s]：%d", var_name, ANOM_YEAR)
    for index, mmdd in enumerate(ALL_MMDD, start=1):
        output_path = output_dir / f"{ANOM_YEAR}{mmdd}.nc"
        if output_path.exists() and not OVERWRITE_INTERMEDIATE:
            generated[mmdd] = output_path
            continue

        forecast_path = raw_data_dir / f"{ANOM_YEAR}{mmdd}.nc"
        if not forecast_path.exists():
            log.warning("缺少预报文件: %s", forecast_path)
            continue
        if mmdd not in climatology_files:
            log.warning("%s %s: 缺少气候态文件", var_name, mmdd)
            continue

        try:
            forecast = read_forecast(forecast_path, var_name, info)
            forecast_mean = compute_member_mean(forecast)
            forecast_mean = forecast_mean.assign_coords(
                lead_time=np.arange(1, forecast_mean.sizes["lead_time"] + 1)
            )
            with xr.open_dataset(climatology_files[mmdd]) as dataset:
                climatology = dataset[var_name].load()
            anomaly = forecast_mean - climatology
        except Exception as error:
            log.error("计算 %s %s 距平失败: %s", var_name, mmdd, error)
            continue

        description = (
            f"{var_name} anomaly for init date {ANOM_YEAR}{mmdd}: "
            f"{ANOM_YEAR} member mean minus {CLIM_START_YEAR}-{CLIM_END_YEAR} "
            "forecast climatology."
        )
        save_intermediate_dataset(
            anomaly,
            output_path,
            var_name,
            description,
            {
                "step": "ANOM",
                "anomaly_method": "forecast_member_mean - forecast_climatology",
                "climatology_period": f"{CLIM_START_YEAR}-{CLIM_END_YEAR}",
                "forecast_year": ANOM_YEAR,
                "init_date": f"{ANOM_YEAR}{mmdd}",
            },
        )
        generated[mmdd] = output_path

        if index % 60 == 0:
            log.info("%s ANOM: %d/365", var_name, index)

    return generated


# ══════════════════════════════════════════════════════════════════════
#  第3步：MERGE — 插值到2.5°并合并三变量
# ══════════════════════════════════════════════════════════════════════

def interpolate_and_merge(
    anomaly_files_by_var: dict[str, dict[str, Path]],
) -> Path:
    """将共同可用的起报日插值到2.5°，沿time拼接并合并三个变量。"""
    common_dates = [
        mmdd
        for mmdd in ALL_MMDD
        if all(mmdd in anomaly_files_by_var[var] for var in VAR_NAMES)
    ]
    if not common_dates:
        raise RuntimeError("没有三个变量均完整的起报日期，无法生成最终文件")
    if len(common_dates) != len(ALL_MMDD):
        log.warning(
            "三个变量共同可用的起报日为%d/365，最终文件只包含共同日期",
            len(common_dates),
        )

    merged_variables = {}
    for var_name in VAR_NAMES:
        log.info("第3步 MERGE [%s]：1.5° → 2.5°", var_name)
        daily_arrays = []
        for index, mmdd in enumerate(common_dates, start=1):
            input_path = anomaly_files_by_var[var_name][mmdd]
            with xr.open_dataset(input_path) as dataset:
                data = dataset[var_name].load()

            data = data.sortby("lat").sortby("lon")
            data = data.interp(
                lat=TARGET_LAT,
                lon=TARGET_LON,
                method="linear",
            )

            # u200/u850可能带有标量层坐标，变量名已经区分层次，可安全删除。
            for coord_name in ("level", "isobaricInhPa"):
                if coord_name in data.coords:
                    data = data.drop_vars(coord_name)

            output_name = "olr" if var_name == "ttr" else var_name
            if var_name == "ttr":
                data = -data
            data = data.rename(output_name)
            init_time = pd.to_datetime(f"{ANOM_YEAR}{mmdd}", format="%Y%m%d")
            data = data.expand_dims(time=[init_time])
            daily_arrays.append(data)

            if index % 100 == 0:
                log.info("%s MERGE: %d/%d", var_name, index, len(common_dates))

        output_name = "olr" if var_name == "ttr" else var_name
        merged_variables[output_name] = xr.concat(daily_arrays, dim="time")

    final_dataset = xr.Dataset(merged_variables)
    final_dataset.attrs.update({
        "description": (
            "Fengshun v1.5 merged anomaly data; TTR converted to OLR and "
            "interpolated to 2.5 degree grid"
        ),
        "source": "Fengshun v1.5 Ensemble Forecast System",
        "variables": "olr, u200, u850",
        "forecast_year": str(ANOM_YEAR),
        "climatology_period": f"{CLIM_START_YEAR}-{CLIM_END_YEAR}",
        "grid": "2.5 degree, lat=73, lon=144",
        "ttr_to_olr": "olr = -ttr",
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    FINAL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if FINAL_OUTPUT_FILE.exists() and not OVERWRITE_FINAL:
        log.info("最终文件已存在，跳过: %s", FINAL_OUTPUT_FILE)
        return FINAL_OUTPUT_FILE

    encoding = {name: FINAL_COMPRESSION for name in merged_variables}
    final_dataset.to_netcdf(FINAL_OUTPUT_FILE, encoding=encoding)
    log.info("最终文件已保存: %s", FINAL_OUTPUT_FILE)
    return FINAL_OUTPUT_FILE


# ══════════════════════════════════════════════════════════════════════
#  单变量流程与主入口
# ══════════════════════════════════════════════════════════════════════

def process_variable(var_name: str) -> tuple[dict[str, Path], dict[str, Path]]:
    """执行一个变量的结构检测、气候态和距平计算。"""
    raw_data_dir = RAW_DATA_BASE_DIR / var_name
    log.info("\n%s", "*" * 70)
    log.info("开始处理变量: %s", var_name)
    log.info("原始数据目录: %s", raw_data_dir)
    log.info("%s", "*" * 70)

    sample_file = find_sample_file(raw_data_dir)
    if sample_file is None:
        raise FileNotFoundError(f"找不到示例文件，请检查目录: {raw_data_dir}")

    info = inspect_nc_file(sample_file, var_name)
    validate_structure(info, sample_file)
    climatology_files = compute_daily_climatology(var_name, info, raw_data_dir)
    anomaly_files = compute_anomaly_for_year(
        var_name,
        info,
        raw_data_dir,
        climatology_files,
    )
    log.info(
        "%s完成: CLIM=%d，ANOM=%d",
        var_name,
        len(climatology_files),
        len(anomaly_files),
    )
    return climatology_files, anomaly_files


def main() -> None:
    """一次运行完成三个变量的CLIM、ANOM和最终合并。"""
    log.info("=" * 70)
    log.info("Fengshun v1.5 MJO完整流水线: CLIM → ANOM → MERGE 2.5°")
    log.info("处理变量: %s", VAR_NAMES)
    log.info("气候态年份: %d-%d", CLIM_START_YEAR, CLIM_END_YEAR)
    log.info("距平年份: %d", ANOM_YEAR)
    log.info("原始数据根目录: %s", RAW_DATA_BASE_DIR)
    log.info("中间输出目录: %s", INTERMEDIATE_OUTPUT_DIR)
    log.info("最终输出文件: %s", FINAL_OUTPUT_FILE)
    log.info("=" * 70)

    anomaly_files_by_var = {}
    for var_name in VAR_NAMES:
        _, anomaly_files = process_variable(var_name)
        anomaly_files_by_var[var_name] = anomaly_files

    output_file = interpolate_and_merge(anomaly_files_by_var)
    log.info("=" * 70)
    log.info("全部处理完成")
    log.info("可供MJO主程序读取: %s", output_file)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
