#!/usr/bin/env python3
#  fengshun v2.0-->my model        fengshun v1.5/ecs2s--->other model
"""
独立计算 TCC、RMSE 并绘制区域柱状图。

处理流程：
1. 读取 CRA_v1.5、my_model 和 other_model 的逐日起报数据；
2. 以 CRA_v1.5 为参考，沿 init_date 维计算 TCC 和 RMSE；
3. 保存格点指标 NetCDF 和区域加权平均 CSV；
4. 分别为 tp 和 t2m 绘制 TCC/RMSE 区域柱状图。

数据形状：
- 单日文件: (week, lat, lon)
- 全年合并: (init_date, week, lat, lon)
- 指标结果: (week, lat, lon)

说明：1 kg/m² 液态水层与 1 mm 数值等价，tp 不使用缩放因子。
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


LOGGER = logging.getLogger("tcc_rmse_bar")
PROJECT_ROOT = Path(__file__).resolve().parent
# DATA_ROOT = Path("M:/FDP")  # / "data"                                ####################
DATA_ROOT = Path("/gpu/zhouchg/liuzh") 

@dataclass(frozen=True)
class Region:
    """柱状图的空间统计区域。"""

    name: str
    abbr: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# ==============================
# 可修改配置
# ==============================
YEAR = 2024  
# VARIABLES = ("tp", "t2m")                                           ####################
VARIABLES = ("t2m",)

# REFERENCE_SOURCE = "CRA_v1.5"                                         #################
# MODEL_SOURCES = ("fengshun_v2.0", "fengshun_v1.5")                ##########################
REFERENCE_SOURCE = "CRA_v1.5"
MODEL_SOURCES = ("my_model","other_model")  # 这里的标签用于绘图和 CSV，实际数据源路径在 SOURCE_ROOTS 中指定


# SOURCE_ROOTS = {
#     # 项目内实际目录名是 Fenghsun_v2.0，这里保留该拼写以匹配数据路径。                                        
#     "fengshun_v2.0": DATA_ROOT / "Fengshun_v2.0" / "anom_combine_1p5_week",                            #################
#     "fengshun_v1.5": DATA_ROOT / "Fengshun_v1.5" / "anom_combine_1p5_week",                            #################
#     "CRA_v1.5": DATA_ROOT / "CRA_v1.5" / "anom_combine_1p5_week",                                      #################
# }
SOURCE_ROOTS = {
    # 项目内实际目录名是 Fenghsun_v2.0，这里保留该拼写以匹配数据路径。
    "my_model": DATA_ROOT /"step3_output"/"my_model"/"anom_combine_1p5_week",
    "other_model": DATA_ROOT /"module1_other_model"/"anom_combine_1p5_week",
    "CRA_v1.5": DATA_ROOT /"step3_output"/"CRA_1.5"/"anom_combine_1p5_week",
}


EXPECTED_DIMS = {"week": 11, "lat": 121, "lon": 240}
PLOT_WEEKS = (3, 4, 5, 6, 34, 56)
MISSING_FILE = "nan"  # 可选 "nan" 或 "skip"
# OUTPUT_DIR = Path("M:/FDP") / "plot_results" / "tcc_rmse_bars_2024_JanJun"                            #################
OUTPUT_DIR = Path("/gpu/zhouchg/liuzh/step4_output")


REGIONS = (
    Region("Global", "GL", -90, 90, 0, 360),
    Region("Tropics", "TP", -20, 20, 0, 360),
    Region("Northern Extratropics", "NET", 20, 90, 0, 360),
    Region("Southern Extratropics", "SET", -90, -20, 0, 360),
    Region("East Asia", "EA", -20, 50, 90, 150),
    Region("South Asia", "SA", -10, 30, 60, 130),
)


# def generate_init_dates(year: int, end_month: int = 12) -> list[str]:     #########################改######################
#     """生成起报日期，自动排除 2 月 29 日。"""

#     current = date(year, 1, 1)
#     # 计算 end_month 的最后一天
#     if end_month == 12:
#         end = date(year, 12, 31)
#     else:
#         # 取 end_month+1 的第 1 天，往前推 1 天
#         end = date(year, end_month + 1, 1) - timedelta(days=1)
#     values: list[str] = []
#     while current <= end:
#         if not (current.month == 2 and current.day == 29):
#             values.append(current.strftime("%Y%m%d"))
#         current += timedelta(days=1)
#     return values

def generate_init_dates(
    year: int,
    start_month: int = 9,
    start_day: int = 23,
    end_month: int = 12,
    end_day: int = 31,
) -> list[str]:
    """生成指定日期范围的起报日期，自动排除 2 月 29 日。"""

    current = date(year, start_month, start_day)
    end = date(year, end_month, end_day)

    values = []
    while current <= end:
        if not (current.month == 2 and current.day == 29):
            values.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return values
###############################################################################################################

def parse_week(value: object) -> int:
    """将 3、'week3'、'w3' 等 week 坐标统一解析为整数。"""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"无法解析 week 坐标: {value!r}")
    return int(match.group())


def _find_variable(ds: xr.Dataset, requested: str) -> str:
    """优先匹配目标变量，必要时自动识别唯一的三维变量。"""

    if requested in ds.data_vars:
        return requested
    case_matches = [name for name in ds.data_vars if name.lower() == requested.lower()]
    if len(case_matches) == 1:
        LOGGER.warning("使用大小写匹配变量 %s 代替 %s", case_matches[0], requested)
        return case_matches[0]
    candidates = [name for name, da in ds.data_vars.items() if da.ndim >= 3]
    if len(candidates) == 1:
        LOGGER.warning("自动识别变量 %s 为 %s", candidates[0], requested)
        return candidates[0]
    raise ValueError(f"无法识别变量 {requested!r}，文件中变量为 {list(ds.data_vars)}")


def _infer_dim(da: xr.DataArray, target: str) -> str:
    """根据常见名称或维度长度识别 week/lat/lon。"""

    aliases = {
        "week": ("week", "lead", "lead_time", "step", "forecast_week"),
        "lat": ("lat", "latitude", "y"),
        "lon": ("lon", "longitude", "x"),
    }
    for name in aliases[target]:
        if name in da.dims:
            return name
    size_matches = [dim for dim in da.dims if da.sizes[dim] == EXPECTED_DIMS[target]]
    if len(size_matches) == 1:
        LOGGER.warning("根据长度推断 %s 维为 %s", target, size_matches[0])
        return size_matches[0]
    raise ValueError(f"无法识别 {target} 维：{dict(da.sizes)}")


def standardize_dataarray(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """统一变量名、维度名和维度顺序。"""

    da = ds[_find_variable(ds, variable)].squeeze(drop=True)
    rename = {}
    for target in ("week", "lat", "lon"):
        actual = _infer_dim(da, target)
        if actual != target:
            rename[actual] = target
    if rename:
        da = da.rename(rename)
    da = da.transpose("week", "lat", "lon")

    # 没有坐标时使用位置索引；有坐标时保留真实经纬度。
    for coord in ("week", "lat", "lon"):
        if coord not in da.coords:
            da = da.assign_coords({coord: np.arange(da.sizes[coord])})
    da.name = variable
    return da


def load_one_file(source: str, variable: str, init_date: str) -> xr.DataArray | None:
    """读取单日数据；缺失或 0 字节文件返回 None。"""

    file_path = Path(SOURCE_ROOTS[source].format(var=variable, yyyymmdd=init_date))
    if not file_path.exists() or file_path.stat().st_size == 0:
        LOGGER.warning("缺失或空文件: %s", file_path)
        return None
    with xr.open_dataset(file_path) as ds:
        da = standardize_dataarray(ds, variable).load()
    # 三套数据直接使用原始值，这里不乘任何缩放因子。
    return da.expand_dims(init_date=[init_date])


def load_year(source: str, variable: str, init_dates: list[str]) -> xr.DataArray:
    """读取一个数据源的全年数据，返回 (init_date, week, lat, lon)。"""

    sample = next(
        (load_one_file(source, variable, d) for d in init_dates
        if Path(SOURCE_ROOTS[source].format(var=variable, yyyymmdd=d)).exists()),
        None,
    )
    if sample is None:
        raise FileNotFoundError(f"未找到可用数据: {source}/{variable}")
    template = sample.isel(init_date=0, drop=True)

    arrays: list[xr.DataArray] = []
    for init_date in init_dates:
        da = load_one_file(source, variable, init_date)
        if da is not None:
            arrays.append(da)
        elif MISSING_FILE == "nan":
            arrays.append(xr.full_like(template, np.nan).expand_dims(init_date=[init_date]))
        elif MISSING_FILE != "skip":
            raise ValueError("MISSING_FILE 只能设为 'nan' 或 'skip'")

    if not arrays:
        raise FileNotFoundError(f"全年均无可用数据: {source}/{variable}")
    return xr.concat(arrays, dim="init_date")


def compute_tcc_rmse(reference: xr.DataArray, model: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    沿起报日维计算逐格点 TCC 和 RMSE。

    xarray 会先根据 init_date/week/lat/lon 坐标对齐数据；
    join='inner' 可避免 skip 策略下两套数据的日期错位。
    """

    reference, model = xr.align(reference, model, join="inner")
    valid = reference.notnull() & model.notnull()
    reference_valid = reference.where(valid)
    model_valid = model.where(valid)

    tcc = xr.corr(reference_valid, model_valid, dim="init_date")
    rmse = np.sqrt(((model_valid - reference_valid) ** 2).mean("init_date", skipna=True))
    tcc.name = "tcc"
    rmse.name = "rmse"
    return {"tcc": tcc, "rmse": rmse}


def select_plot_weeks(da: xr.DataArray) -> xr.DataArray:
    """选取 week3~6；week34/56 不存在时由单周平均派生。"""

    lookup = {parse_week(value): value for value in da.week.values}
    selected: list[xr.DataArray] = []
    for week in PLOT_WEEKS:
        if week in lookup:
            item = da.sel(week=lookup[week])
        elif week == 34 and 3 in lookup and 4 in lookup:
            item = da.sel(week=[lookup[3], lookup[4]]).mean("week")
        elif week == 56 and 5 in lookup and 6 in lookup:
            item = da.sel(week=[lookup[5], lookup[6]]).mean("week")
        else:
            raise ValueError(f"无法选取或派生 week{week}，可用周次: {sorted(lookup)}")
        selected.append(item.expand_dims(week=[week]))
    return xr.concat(selected, dim="week")


def spatial_weighted_mean(da: xr.DataArray, region: Region) -> xr.DataArray:
    """对指定区域做 cos(lat) 面积加权平均。"""

    lat_ascending = bool(da.lat.values[0] < da.lat.values[-1])
    lat_slice = (
        slice(region.lat_min, region.lat_max)
        if lat_ascending
        else slice(region.lat_max, region.lat_min)
    )
    subset = da.sel(lat=lat_slice)

    lon_min, lon_max = region.lon_min, region.lon_max
    if float(subset.lon.min()) < 0 and lon_max > 180:
        lon_min = (lon_min + 180) % 360 - 180
        lon_max = (lon_max + 180) % 360 - 180
    if lon_min <= lon_max:
        subset = subset.sel(lon=slice(lon_min, lon_max))
    else:
        subset = subset.where((subset.lon >= lon_min) | (subset.lon <= lon_max), drop=True)

    weights = np.cos(np.deg2rad(subset.lat))
    weights.name = "latitude_weights"
    return subset.weighted(weights).mean(("lat", "lon"))


def make_region_table(
    results: dict[str, dict[str, xr.DataArray]], variable: str, metric: str
) -> pd.DataFrame:
    """把格点指标转换成便于保存和绘图的长表。"""

    labels = ("week3", "week4", "week5", "week6", "week3-4", "week5-6")
    rows = []
    for source in MODEL_SOURCES:
        selected = select_plot_weeks(results[source][metric])
        for region in REGIONS:
            values = spatial_weighted_mean(selected, region).values
            for label, value in zip(labels, values):
                rows.append(
                    {
                        "variable": variable,
                        "metric": metric,
                        "region": region.name,
                        "region_abbr": region.abbr,
                        "source": source,
                        "reference_source": REFERENCE_SOURCE,
                        "week": label,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def save_metric(da: xr.DataArray, variable: str, metric: str, source: str, year: int) -> Path:
    """保存逐格点 TCC/RMSE 结果。"""

    output_dir = OUTPUT_DIR / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{variable}_{metric}_{source}_vs_{REFERENCE_SOURCE}_{year}.nc"
    output = da.astype("float32")
    output.name = variable
    output.attrs.update({"metric": metric, "model_source": source, "reference_source": REFERENCE_SOURCE})
    output.to_dataset().to_netcdf(output_file)
    return output_file


def plot_region_bars(table: pd.DataFrame, variable: str, metric: str, year: int) -> Path:
    """绘制 6 个区域的双数据源对比柱状图。"""

    labels = ["week3", "week4", "week5", "week6", "week3-4", "week5-6"]
    tick_labels = ["3", "4", "5", "6", "3-4", "5-6"]
    colors = {"my_model": "tab:red", "other_model": "tab:blue"}
    output_dir = OUTPUT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{variable}_{metric}_regional_bars_{year}.png"

    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    x = np.arange(len(labels))
    for index, (ax, region) in enumerate(zip(axes.flat, REGIONS)):
        for offset, source in zip((-0.18, 0.18), MODEL_SOURCES):
            values = (
                table[(table.region_abbr == region.abbr) & (table.source == source)]
                .set_index("week")
                .loc[labels, "value"]
                .to_numpy()
            )
            ax.bar(
                x + offset,
                values,
                width=0.36,
                color=colors.get(source, "gray"),
                alpha=0.85,
                label=source if index == len(REGIONS) - 1 else None,
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x, tick_labels)
        ax.set_title(region.name)
        ax.set_ylabel(metric.upper())
        if index >= 4:
            ax.set_xlabel("Lead time (weeks)")

    handles, legend_labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{variable} {metric.upper()} regional weighted mean ({year})", fontsize=16)
    fig.subplots_adjust(top=0.91, bottom=0.10, wspace=0.25, hspace=0.32)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def process_variable(variable: str, year: int) -> None:
    """完成一个变量的计算、保存和柱状图绘制。"""

    # init_dates = generate_init_dates(year, end_month=6)                   ##############################改####################
    init_dates = generate_init_dates(year,start_month=9,start_day=23,end_month=12,end_day=31,)

    LOGGER.info("读取参考数据 %s/%s", REFERENCE_SOURCE, variable)
    reference = load_year(REFERENCE_SOURCE, variable, init_dates)

    results: dict[str, dict[str, xr.DataArray]] = {}
    for source in MODEL_SOURCES:
        LOGGER.info("读取并计算 %s vs %s (%s)", source, REFERENCE_SOURCE, variable)
        model = load_year(source, variable, init_dates)
        results[source] = compute_tcc_rmse(reference, model)
        for metric, da in results[source].items():
            save_metric(da, variable, metric, source, year)

    for metric in ("tcc", "rmse"):
        table = make_region_table(results, variable, metric)
        table_dir = OUTPUT_DIR / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_dir / f"{variable}_{metric}_regional_weighted_mean_{year}.csv", index=False)
        plot_region_bars(table, variable, metric, year)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算 TCC/RMSE 并绘制区域柱状图")
    parser.add_argument("--variables", nargs="+", default=list(VARIABLES))
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model-sources", nargs="+", default=list(MODEL_SOURCES))
    parser.add_argument("--reference-source", type=str, default=REFERENCE_SOURCE)
    # 改成三条完整路径模板（像 step5）########################################################################################单独运行时记得修改这里！！！！！！！！##########################
    parser.add_argument(
        "--reference-path", type=str,
        default=r"/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
        help="参考场路径模板",
    )
    parser.add_argument(
        "--my-model-path", type=str,
        default=r"/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
        help="my_model 路径模板",
    )
    parser.add_argument(
        "--other-model-path", type=str,
        default=r"/gpu/zhouchg/liuzh/module1_other_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
        help="other_model 路径模板",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global SOURCE_ROOTS, MODEL_SOURCES, OUTPUT_DIR, REFERENCE_SOURCE
    REFERENCE_SOURCE = args.reference_source
    MODEL_SOURCES = tuple(args.model_sources)
    # SOURCE_ROOTS 里现在存的是“完整模板字符串”
    SOURCE_ROOTS = {
        "my_model":    args.my_model_path,
        "other_model": args.other_model_path,
        REFERENCE_SOURCE: args.reference_path,
    }
    OUTPUT_DIR = Path(args.output_dir)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    for variable in args.variables:
        process_variable(variable, args.year)


if __name__ == "__main__":
    main()
