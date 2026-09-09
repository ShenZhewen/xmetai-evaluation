"""
================================================================================
CRA_v1.5 作为观测场，评测 my_model 和 other_model 预报变量性能
================================================================================

【任务说明】
  以 CRA_v1.5 周平均距平数据作为观测参考（真值），对比评估：
    1. my_model 预报系统的预报技巧
    2. other_model 预报系统的预报技巧

  评估变量：ssrd, w10, w100, tp, t2m 等多个变量

【处理步骤】
  Step 0: 配置检查 —— 验证路径、变量、年份等参数
  Step 1: 生成起报日期 —— 平年奇数日（0101, 0103, 0105, ..., 1231），排除 0229
           每年约 183 天，2020-2024 共 5 年约 915 个起报日
  Step 2: 读取数据 —— 分别读取三套数据的周平均距平文件，并合并为统一形状
           - CRA_v1.5:      1.5° 分辨率，单位 J/m²，作为观测参考
           - my_model: 0.25° 分辨率，单位 W/m² → ×86400 转为 J/m² 后再双线性插值到 1.5°
           - other_model:         1.5° 分辨率，单位 J/m²
           每套数据合并后形状: (init_date, week, lat, lon)
  Step 3: 计算统计量 —— 分别计算 my_model vs CRA 和 other_model vs CRA 的:
           - TCC (Temporal Correlation Coefficient, 时间相关系数)
           - p-value (统计显著性)
           - RMSE (Root Mean Square Error, 均方根误差)
           统计结果形状: (week, lat, lon)，保存为 NetCDF 文件
  Step 4: 区域平均 —— 计算多个区域的纬度加权平均 TCC 和 RMSE
           纬度权重 = cos(lat)，用于补偿经线汇聚效应
           输出 CSV 表格
  Step 5: 可视化 ——
           a) 全球 TCC/RMSE 空间对比图（Robinson 投影）
              每行一个预报系统，每列一个 week；调整行间距减少留白
           b) 区域纬度加权平均柱状图
              每个区域一张子图，对比两个预报系统在各 week 的表现

【数据路径模板】
  CRA_v1.5:      M:\CRA_v1.5\anom_1p5_week\{var}\{yyyymmdd}.nc       (1.5°)
  my_model: M:\my_model\anom_1p5_week\{var}\{yyyymmdd}.nc  (0.25°, W/m²→J/m²→插值到1.5°)
  other_model:         M:\other_model\{var}\anom_week\{var}_{yyyymmdd}.nc         (1.5°)

【核心数据形状约定】
  - 单个 nc 文件:  variable(week, lat, lon)  或  variable(week, latitude, longitude)
  - 合并起报日后: variable(init_date, week, lat, lon)
  - 统计结果:     metric(week, lat, lon)

【时间范围】
  - 年份: 2020-2024（5 年）
  - 起报日: 平年奇数日（day-of-year 为奇数的日期），排除 0229
  - 约 183 天/年 × 5 年 = 915 个起报日

【用法示例】
  python CRA_v1.5_process.py                              # 默认处理所有变量
  python CRA_v1.5_process.py --var tp                      # 处理单个变量
  python CRA_v1.5_process.py --vars tp t2m w10             # 批量处理
  python CRA_v1.5_process.py --vars tp --steps read stats  # 只读取+统计
  python CRA_v1.5_process.py --vars tp --steps plot        # 只绘图（需已有统计文件）
  python CRA_v1.5_process.py --vars tp --year-start 2021 --year-end 2023

【灵活修改指南】
  所有可修改参数集中在代码开头的 "USER CONFIG" 区域，包括:
  - 变量列表、年份范围
  - 三套数据的路径模板
  - 目标网格分辨率
  - 评估区域定义
  - 绘图参数（色标、图幅、字体等）
  - 输出目录
  修改这些参数即可适配不同的数据路径、变量、年份或评估需求，
  无需改动业务逻辑代码。

作者：FDP Project
日期：2026-06-25
================================================================================
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          USER  CONFIG（按需修改）   （单独运行该代码时）                         ║
# ║   切换变量 / 路径 / 年份 / 区域 / 绘图参数 —— 只改这里，函数无须改动         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 1. 处理的变量列表 ─────────────────────────────────────────────────────────
# 命令行 --var / --vars 可覆盖此默认值
# DEFAULT_VARS = ["w10m"]                                                   ##############################
DEFAULT_VARS = ["t2m"]

# ── 2. 年份范围 ───────────────────────────────────────────────────────────────
YEAR_START = 2020
YEAR_END   = 2024

# ── 3. 数据路径模板 ───────────────────────────────────────────────────────────
# {var}       → 变量名（如 tp, t2m, w10）
# {yyyymmdd}  → 起报日期（如 20200101）
# CRA_PATH_TEMPLATE      = r"M:\CRA_v1.5\anom_1p5_week\{var}\{yyyymmdd}.nc"                              ##############################
# FENGSHUN_PATH_TEMPLATE = r"M:\fengshun_v1.5_w_ssr_ssrd\{var}\week\{yyyymmdd}.nc"                     ##############################
# ECS2S_PATH_TEMPLATE    = r"M:\ECS2S\{var}\anom_week\{var}_{yyyymmdd}.nc"                              ##############################
CRA_PATH_TEMPLATE      = r"/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/{var}/{yyyymmdd}.nc"
MY_MODEL_PATH_TEMPLATE = r"/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc"
OTHER_MODEL_PATH_TEMPLATE    = r"/gpu/zhouchg/liuzh/module1_other_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc"

# ── 4. 数据源标签（用于输出文件命名和图例） ───────────────────────────────────
# REFERENCE_SOURCE = "CRA_v1.5"       # 观测参考名称                                                             ##############################
# MODEL_SOURCES = {                    # 待评估的预报系统                                                        ##############################
#     "Fengshun_v1.5": {                                                                                          ##############################
#         "template": FENGSHUN_PATH_TEMPLATE,                                                                      ##############################
#         "resolution": 0.25,          # 原始分辨率（度），需插值到 1.5°                                           ##############################
#         "label": "Fengshun v1.5",                                                                              ##############################
#         "unit_conversion": 86400,    # ssrd: W/m² → J/m² (× 每日秒数)，非 ssrd 变量自动跳过                     ##############################
#         "path_var": {"w10m": "w10"}, # 路径中变量名映射（Fengshun 中 w10m 目录名为 w10）                            ##############################
#     },                                                                                                        ##############################
#     "ECS2S": {                                                                                                     ##############################
#         "template": ECS2S_PATH_TEMPLATE,                                                                          ##############################
#         "resolution": 1.5,           # 原始分辨率已是 1.5°，无需插值                                                ##############################
#         "label": "ECS2S",                                                                                         ##############################
#         "unit_conversion": None,     # 无需转换，已为 J/m²                                                      ##############################
#         "path_var": {"w10m": "w10"}, # 路径中变量名映射（ECS2S 中 w10m 命名为 w10）                           ##############################
#     },
# }
REFERENCE_SOURCE = "CRA_v1.5" 
MODEL_SOURCES = {                    # 待评估的预报系统
    "my_model": {
        "template": MY_MODEL_PATH_TEMPLATE,
        "resolution": 1.5,          # 原始分辨率（度），需插值到 1.5°
        "label": "My Model",
        "unit_conversion": None,    # ssrd: W/m² → J/m² (× 每日秒数)，非 ssrd 变量自动跳过
        "path_var": {}, # 路径中变量名映射（my_model 中 w10m 目录名为 w10）
    },
    "other_model": {
        "template": OTHER_MODEL_PATH_TEMPLATE,
        "resolution": 1.5,           # 原始分辨率已是 1.5°，无需插值
        "label": "Other Model",
        "unit_conversion": None,     # 无需转换，已为 J/m²
        "path_var": {}, # 路径中变量名映射（Other Model 中 w10m 命名为 w10）
    },
}

REFERENCE_TEMPLATE = CRA_PATH_TEMPLATE
# CRA 参考数据路径变量名映射（如 w10m → ws10m）
# REFERENCE_PATH_VAR = {"w10m": "ws10m"}                                                        ##############################
REFERENCE_PATH_VAR = {}

# ── 5. 目标网格（1.5° 等经纬度） ──────────────────────────────────────────────
TARGET_LAT = np.arange(90.0, -91.0, -1.5)   # N→S, 121 点
TARGET_LON = np.arange(0.0, 360.0, 1.5)     # 0→360, 240 点

# ── 6. 缺失文件策略 ───────────────────────────────────────────────────────────
# "nan":  缺失文件用 NaN 填充，保持 init_date 维度长度一致
# "skip": 跳过该起报日（各数据源的 init_date 维度长度可能不同）
MISSING_FILE_MODE = "nan"

# ── 7. NetCDF 压缩参数 ────────────────────────────────────────────────────────
COMPRESSION = {"zlib": True, "complevel": 4}

# ── 8. 输出目录 ───────────────────────────────────────────────────────────────
# OUTPUT_DIR = Path(r"e:\WCP\Fengshun_u_v_ssr_1.5\output")                                      ###################################输出目录！！！！！！###################
OUTPUT_DIR = Path(r"/gpu/zhouchg/liuzh/step5_output")
STATS_DIR  = OUTPUT_DIR / "stats"       # 格点统计 NetCDF
REGION_DIR = OUTPUT_DIR / "regional"    # 区域平均 CSV
FIGURE_DIR = OUTPUT_DIR / "figures"     # 空间图 & 柱状图

# ── 9. 需要绘制的 week（None=全部；可指定子集如 [3, 4, 5, 6, 34, 56]） ──────
PLOT_WEEKS = ["week3", "week4", "week5", "week6", "week34", "week56"]

# ── 10. 评估区域定义 ──────────────────────────────────────────────────────────
# 格式: {区域名: {"lat": (min, max), "lon": (min, max)}}
# lon 使用 0-360 体系（若数据经度为 -180~180，代码内部会自动转换）
REGIONS = {
    "Global":                 {"lat": (-90, 90),  "lon": (0, 360)},
    "Tropics":                {"lat": (-20, 20),  "lon": (0, 360)},
    "Northern Extratropics":  {"lat": (20, 90),   "lon": (0, 360)},
    "Southern Extratropics":  {"lat": (-90, -20), "lon": (0, 360)},
    "East Asia":              {"lat": (-20, 50),  "lon": (90, 150)},
    "South Asia":             {"lat": (-10, 30),  "lon": (60, 130)},
}

# ── 11. 绘图参数 ──────────────────────────────────────────────────────────────
MAP_FIG_DPI       = 300      # 空间图 DPI
BAR_FIG_DPI       = 300      # 柱状图 DPI
MAP_CMAP_TCC      = "RdBu_r" # TCC 色标
MAP_CMAP_RMSE     = "YlOrRd" # RMSE 色标（正值为主）
FIGURE_FORMAT     = "png"    # 输出图片格式

# ── 12. 运行控制 ──────────────────────────────────────────────────────────────
DEFAULT_STEPS = ["read", "stats", "regional", "plot"]
# 可选步骤: read（读取+合并）、stats（计算统计量）、regional（区域平均）、plot（绘图）
# 例如只绘图: --steps plot


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          INTERNAL（业务逻辑代码）                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 工具函数 ───────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    """统一日志输出，带时间戳。"""
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def generate_init_dates_odd_days(
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
) -> List[Tuple[int, str]]:
    """
    生成平年奇数日起报日期列表。

    以非闰年（2001）为模板，取 day-of-year 为奇数的日期
    （1月1日=第1天, 1月3日=第3天, ..., 12月31日=第365天），
    显式排除 0229（其在平年中不存在，但作为安全检查）。
    每年 183 天。

    返回
    -------
    List[Tuple[int, str]]
        [(year, mmdd), ...]  例如 [(2020, "0101"), (2020, "0103"), ...]
    """
    # 用非闰年生成模板日期
    template_year = 2001  # 平年
    start = date(template_year, 1, 1)
    end = date(template_year, 12, 31)

    mmdd_list = []
    current = start
    day_idx = 1
    while current <= end:
        # 取 day-of-year 为奇数的日期，且排除 0229
        if day_idx % 2 == 1 and not (current.month == 2 and current.day == 29):
            mmdd_list.append(current.strftime("%m%d"))
        current += timedelta(days=1)
        day_idx += 1

    log(f"  模板 mmdd: {len(mmdd_list)} 天 (平年奇数日)")

    # 扩展到所有年份
    init_dates = []
    for year in range(year_start, year_end + 1):
        for mmdd in mmdd_list:
            # 跳过闰年的 0229（0229 本就不在 mmdd_list 中，但作为双重保险）
            if mmdd == "0229":
                continue
            init_dates.append((year, mmdd))

    log(f"  起报日期总数: {len(init_dates)} ({year_start}-{year_end}, {len(mmdd_list)} 天/年)")
    return init_dates


def build_file_path(
    path_template: str,
    var: str,
    year: int,
    mmdd: str,
    path_var: Optional[Dict[str, str]] = None,
) -> Path:
    """根据路径模板构造完整文件路径。

    path_var: 可选，变量名映射字典。若 var 在映射中，
              则路径中的 {var} 使用映射后的名称。
              例: path_var={"w10m": "w10"} 时，var="w10m"
              生成的路径使用 "w10"。
    """
    yyyymmdd = f"{year}{mmdd}"
    effective_var = var
    if path_var and var in path_var:
        effective_var = path_var[var]
    return Path(path_template.format(var=effective_var, yyyymmdd=yyyymmdd))


# ── 数据结构检测 ──────────────────────────────────────────────────────────────

def detect_dim_names(ds: xr.Dataset, var_name: str) -> dict:
    """
    自动检测变量名、经/纬度维度名、week 维度名。

    返回
    -------
    dict  keys: var_name, lat_dim, lon_dim, week_dim, lat_size, lon_size, ...
    """
    da = ds[var_name]

    # ── 经/纬度维度名探测 ──
    lat_candidates = ["lat", "latitude", "LAT", "Latitude", "y"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude", "x"]
    week_candidates = ["week", "lead_time", "lead", "time", "step", "forecast_week"]

    dims_in_data = list(da.dims)

    lat_dim = next((d for d in lat_candidates if d in dims_in_data), None)
    lon_dim = next((d for d in lon_candidates if d in dims_in_data), None)
    week_dim = next((d for d in week_candidates if d in dims_in_data), None)

    # 按长度兜底推断
    if lat_dim is None:
        for d in dims_in_data:
            if da.sizes[d] == 121:
                lat_dim = d; break
    if lon_dim is None:
        for d in dims_in_data:
            if da.sizes[d] == 240:
                lon_dim = d; break
    if week_dim is None:
        # week 一般是除 lat/lon 外的维度
        for d in dims_in_data:
            if d not in (lat_dim, lon_dim):
                week_dim = d; break

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"无法识别经纬度维度名。可用维度: {dims_in_data}, "
            f"候选: lat={lat_candidates}, lon={lon_candidates}"
        )

    lat_vals = da[lat_dim].values if lat_dim in da.coords else np.arange(da.sizes[lat_dim])
    lon_vals = da[lon_dim].values if lon_dim in da.coords else np.arange(da.sizes[lon_dim])

    return {
        "var_name": var_name,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
        "week_dim": week_dim,
        "lat_size": int(da.sizes[lat_dim]),
        "lon_size": int(da.sizes[lon_dim]),
        "week_size": int(da.sizes[week_dim]) if week_dim else 0,
        "lat_min": float(np.min(lat_vals)),
        "lat_max": float(np.max(lat_vals)),
        "lon_min": float(np.min(lon_vals)),
        "lon_max": float(np.max(lon_vals)),
        "lat_is_increasing": bool(lat_vals[0] < lat_vals[-1]) if len(lat_vals) > 1 else True,
        "all_dims": dims_in_data,
        "all_data_vars": list(ds.data_vars),
    }


def find_data_variable(ds: xr.Dataset, expected_var: str) -> str:
    """
    在 NC 文件中定位真实变量名。

    优先精确匹配 expected_var；其次大小写不敏感匹配；
    最后取第一个 ≥3 维的 data_var。
    """
    if expected_var in ds.data_vars:
        return expected_var

    # 大小写不敏感
    lower_matches = [v for v in ds.data_vars if v.lower() == expected_var.lower()]
    if len(lower_matches) == 1:
        log(f"  [WARN] 变量名大小写不匹配: 使用 '{lower_matches[0]}' 代替 '{expected_var}'", "WARN")
        return lower_matches[0]

    # 取第一个有 ≥3 维的变量（week/lat/lon = 3 维）
    for v in ds.data_vars:
        if len(ds[v].dims) >= 3:
            log(f"  [WARN] 自动选择变量 '{v}' (期望 '{expected_var}')", "WARN")
            return v

    raise ValueError(
        f"无法在文件中找到变量 '{expected_var}'。"
        f"可用变量: {list(ds.data_vars)}"
    )


# ── 插值函数 ───────────────────────────────────────────────────────────────────

def interpolate_to_1p5(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    """
    将任意分辨率的数据双线性插值到全局 1.5° 目标网格。

    参数
    ----------
    da      : xr.DataArray  输入数据
    lat_dim : str           纬度维度名
    lon_dim : str           经度维度名

    返回
    -------
    xr.DataArray  维度中 lat/lon 已替换为 1.5° 目标网格
    """
    # 重命名为标准名称
    rename = {}
    if lat_dim != "lat":
        rename[lat_dim] = "lat"
    if lon_dim != "lon":
        rename[lon_dim] = "lon"
    if rename:
        da = da.rename(rename)

    # 确保坐标单调递增（interp 要求）
    da = da.sortby("lat").sortby("lon")

    # 双线性插值
    da_interp = da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": None},  # 允许小幅外推
    )
    return da_interp


# ── 坐标标准化 ────────────────────────────────────────────────────────────────

def standardize_coords(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    week_dim: str = None,
) -> xr.DataArray:
    """
    统一坐标系统:
      - 重命名 lat/lon 维度为标准名
      - lat 单调递增（S→N）
      - lon 转为 0-360
    """
    rename = {}
    if lat_dim != "lat":
        rename[lat_dim] = "lat"
    if lon_dim != "lon":
        rename[lon_dim] = "lon"
    if rename:
        da = da.rename(rename)

    # lat 单调递增
    lat_vals = da["lat"].values
    if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
        da = da.sortby("lat")

    # lon → 0-360
    lon_vals = da["lon"].values
    if np.any(lon_vals < 0):
        da = da.assign_coords(lon=(lon_vals + 360) % 360)
        da = da.sortby("lon")

    return da


# ── 数据读取与合并 ────────────────────────────────────────────────────────────

def read_one_file(
    file_path: Path,
    var: str,
    need_interpolate: bool = False,
    unit_conversion: Optional[float] = None,
) -> Optional[xr.DataArray]:
    """
    读取单个 NC 文件并返回标准化的 DataArray。

    形状: (week, lat, lon)，其中 lat/lon 为目标网格坐标。

    参数
    ----------
    file_path        : Path   文件路径
    var              : str    期望的变量名
    need_interpolate : bool   是否需要插值到 1.5°（my_model 0.25° 数据）
    unit_conversion  : float  单位转换因子（乘性，在插值前执行）;
                              例: 86400 = W/m² → J/m²

    返回
    -------
    xr.DataArray 或 None（文件缺失/读取失败时）
    """
    if not file_path.exists():
        return None
    if file_path.stat().st_size == 0:
        log(f"  [WARN] 文件为空: {file_path.name}", "WARN")
        return None

    try:
        ds = xr.open_dataset(file_path, chunks=None)
    except Exception as e:
        log(f"  [ERROR] 无法打开 {file_path.name}: {e}", "ERROR")
        return None

    try:
        actual_var = find_data_variable(ds, var)
        info = detect_dim_names(ds, actual_var)
        da = ds[actual_var].squeeze(drop=True).copy()

        # 如果存在多余的单值维度（如 member, height），squeeze 掉
        extra_dims = [
            d for d in da.dims
            if d not in (info["lat_dim"], info["lon_dim"], info["week_dim"])
        ]
        for d in extra_dims:
            if da.sizes.get(d, 1) == 1:
                da = da.squeeze(d, drop=True)

        # 单位转换（在插值前执行，保持原始网格上的物理量一致性）
        # my_model: W/m² → J/m² (×86400 秒/天)
        if unit_conversion is not None:
            da = da * unit_conversion

        # 插值到 1.5°（仅 my_model 数据需要）
        if need_interpolate:
            da = interpolate_to_1p5(da, info["lat_dim"], info["lon_dim"])
            # 插值后维度名已标准化为 lat/lon
            actual_lat_dim = "lat"
            actual_lon_dim = "lon"
        else:
            actual_lat_dim = info["lat_dim"]
            actual_lon_dim = info["lon_dim"]

        # 坐标标准化
        da = standardize_coords(da, actual_lat_dim, actual_lon_dim)

        da.name = var
        da = da.load()  # 加载到内存
        return da

    except Exception as e:
        log(f"  [ERROR] 处理 {file_path.name} 失败: {e}", "ERROR")
        return None
    finally:
        ds.close()


def read_and_merge_source(
    path_template: str,
    var: str,
    init_dates: List[Tuple[int, str]],
    need_interpolate: bool = False,
    source_label: str = "",
    unit_conversion: Optional[float] = None,
    path_var: Optional[Dict[str, str]] = None,
) -> xr.DataArray:
    """
    读取单个数据源的全部起报日文件并合并为 (init_date, week, lat, lon)。

    参数
    ----------
    path_template    : str   路径模板
    var              : str   变量名
    init_dates       : list  起报日期列表 [(year, mmdd), ...]
    need_interpolate : bool  是否需要插值（仅 my_model 0.25°）
    source_label     : str   数据源名称（用于日志）
    unit_conversion  : float 单位转换因子（乘性，在插值前执行）
    path_var         : dict  可选，路径变量名映射（如 {"w10m": "w10"}）

    返回
    -------
    xr.DataArray  维度: (init_date, week, lat, lon)
    """
    log(f"\n{'─'*60}")
    log(f" 读取数据源: {source_label} [{var}]")
    log(f"  插值到 1.5°: {'是' if need_interpolate else '否'}")
    if unit_conversion is not None:
        log(f"  单位转换: ×{unit_conversion} (插值前)")
    log(f"{'─'*60}")

    arrays = []
    missing = []
    first_valid_da = None  # 第一个有效数据的引用，用于生成 NaN 模板

    pbar = tqdm(init_dates, desc=f"读取 {source_label}", unit="file")
    for year, mmdd in pbar:
        file_path = build_file_path(path_template, var, year, mmdd, path_var=path_var)
        da = read_one_file(file_path, var, need_interpolate=need_interpolate,
                           unit_conversion=unit_conversion)

        if da is None:
            missing.append(f"{year}{mmdd}")
            if MISSING_FILE_MODE == "nan":
                if first_valid_da is not None:
                    # 用第一个有效数据构造 NaN 占位
                    nan_da = xr.full_like(first_valid_da, np.nan)
                    nan_da = nan_da.expand_dims(init_date=[f"{year}{mmdd}"])
                    arrays.append(nan_da)
                # 若 first_valid_da 仍为 None（前面全是缺失），先跳过，
                # 等找到第一个有效文件后再回补
            elif MISSING_FILE_MODE == "skip":
                pass
            pbar.set_postfix({"miss": len(missing), "ok": len(arrays) - len(missing) if MISSING_FILE_MODE == "nan" else len(arrays)})
            continue

        # 记录第一个有效数据作为 NaN 模板（不含 init_date 维度）
        if first_valid_da is None:
            first_valid_da = da

            # 回补之前因无模板而跳过的缺失文件
            if MISSING_FILE_MODE == "nan":
                for prev_year, prev_mmdd in init_dates:
                    prev_date_str = f"{prev_year}{prev_mmdd}"
                    if prev_date_str in missing:
                        nan_da = xr.full_like(first_valid_da, np.nan)
                        nan_da = nan_da.expand_dims(init_date=[prev_date_str])
                        arrays.append(nan_da)

        # 扩展 init_date 维度
        da = da.expand_dims(init_date=[f"{year}{mmdd}"])
        arrays.append(da)
        pbar.set_postfix({"miss": len(missing), "ok": len(arrays) - len(missing) if MISSING_FILE_MODE == "nan" else len(arrays)})

    if not arrays:
        raise FileNotFoundError(
            f"数据源 {source_label}/{var}: 所有文件均缺失！"
        )

    # 按日期排序（确保 init_date 维度顺序正确）
    arrays.sort(key=lambda x: str(x.init_date.values[0]))

    # 合并
    log(f"  合并 {len(arrays)} 个起报日...")
    combined = xr.concat(arrays, dim="init_date")
    combined.attrs["source"] = source_label
    combined.attrs["variable"] = var
    combined.attrs["n_init_dates"] = len(arrays)
    combined.attrs["missing_dates"] = ",".join(missing) if missing else "none"
    combined.attrs["resolution"] = "1.5°"

    log(f"  完成: 形状 {dict(combined.sizes)}, 缺失 {len(missing)} 天")
    return combined


# ── 统计量计算 ────────────────────────────────────────────────────────────────

def compute_statistics(
    forecast: xr.DataArray,
    reference: xr.DataArray,
    model_label: str = "",
) -> Dict[str, xr.DataArray]:
    """
    计算模式预报相对参考场的 TCC、p-value、RMSE。

    参数
    ----------
    forecast : xr.DataArray  预报数据 (init_date, week, lat, lon)
    reference: xr.DataArray  参考数据 (init_date, week, lat, lon)
    model_label: str         模式名称（用于日志）

    返回
    -------
    dict
        {"tcc": DataArray(week, lat, lon),
         "p_value": DataArray(week, lat, lon),
         "rmse": DataArray(week, lat, lon)}
    """
    log(f"  计算统计量: {model_label} vs CRA_v1.5 ...")

    # ── TCC (时间相关系数，沿 init_date 维度) ──
    tcc = xr.corr(forecast, reference, dim="init_date")

    # ── 有效样本数 ──
    n_valid = forecast.notnull() & reference.notnull()
    n = n_valid.sum(dim="init_date")

    # ── p-value (基于 t 分布) ──
    try:
        from scipy.stats import t as t_dist

        safe_r = tcc.clip(min=-0.999999, max=0.999999)
        t_stat = safe_r * np.sqrt((n - 2) / (1.0 - safe_r ** 2))
        # 双侧检验
        p_value = xr.apply_ufunc(
            lambda t_val, df_val: 2.0 * t_dist.sf(np.abs(t_val), df_val),
            t_stat,
            n - 2,
            vectorize=True,
            dask="allowed",
        )
        # 无效样本设为 NaN
        p_value = p_value.where(n > 2)
    except ImportError:
        log("  [WARN] scipy 不可用，p-value 设为 NaN", "WARN")
        p_value = xr.full_like(tcc, np.nan)

    # ── RMSE ──
    rmse = np.sqrt(((forecast - reference) ** 2).mean(dim="init_date", skipna=True))

    # ── 属性 ──
    for da, name, desc in [
        (tcc, "TCC", "Temporal Correlation Coefficient"),
        (p_value, "p_value", "p-value of TCC"),
        (rmse, "RMSE", "Root Mean Square Error"),
    ]:
        da.attrs["metric"] = name
        da.attrs["long_name"] = desc
        da.attrs["model"] = model_label
        da.attrs["reference"] = "CRA_v1.5"

    log(f"    TCC 范围: [{float(tcc.min()):.4f}, {float(tcc.max()):.4f}]")
    log(f"    RMSE 范围: [{float(rmse.min()):.4f}, {float(rmse.max()):.4f}]")

    return {"tcc": tcc, "p_value": p_value, "rmse": rmse}


# ── 保存统计结果 ──────────────────────────────────────────────────────────────

def save_stats_netcdf(
    stats: Dict[str, xr.DataArray],
    var: str,
    model_name: str,
):
    """保存格点统计结果为 NetCDF。每个 metric 一个文件。"""
    stats_dir = STATS_DIR / var
    stats_dir.mkdir(parents=True, exist_ok=True)

    for metric, da in stats.items():
        out_path = stats_dir / f"{var}_{metric}_{model_name}_vs_CRA_v1.5.nc"
        ds_out = da.to_dataset(name=var)
        ds_out.attrs["metric"] = metric
        ds_out.attrs["model"] = model_name
        ds_out.attrs["reference"] = "CRA_v1.5"
        ds_out.attrs["variable"] = var
        ds_out.attrs["created"] = str(date.today())

        encoding = {var: COMPRESSION}
        ds_out.to_netcdf(out_path, encoding=encoding)
        log(f"    保存 → {out_path}")


def load_stats_netcdf(var: str, model_name: str) -> Dict[str, xr.DataArray]:
    """读取已保存的统计结果 NetCDF。"""
    stats_dir = STATS_DIR / var
    result = {}
    for metric in ["tcc", "p_value", "rmse"]:
        fpath = stats_dir / f"{var}_{metric}_{model_name}_vs_CRA_v1.5.nc"
        if fpath.exists():
            ds = xr.open_dataset(fpath, chunks=None)
            result[metric] = ds[var].load()
            ds.close()
        else:
            log(f"  [WARN] 统计文件不存在: {fpath}", "WARN")
    return result


# ── 纬度加权区域平均 ──────────────────────────────────────────────────────────

def lat_weighted_regional_mean(
    da: xr.DataArray,
    region: dict,
) -> xr.DataArray:
    """
    计算指定区域的 cos(lat) 加权平均值。

    参数
    ----------
    da     : xr.DataArray  含 lat/lon 维度的数据
    region : dict          {"lat": (min, max), "lon": (min, max)}

    返回
    -------
    xr.DataArray  区域平均后的数据（lat/lon 维度已约简）
    """
    lat_min, lat_max = region["lat"]
    lon_min, lon_max = region["lon"]

    # ── 纬度切片（处理 N→S 和 S→N 两种情况） ──
    lat_vals = da["lat"].values
    lat_ascending = bool(lat_vals[0] < lat_vals[-1])
    if lat_ascending:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    subset = da.sel(lat=lat_slice)

    # ── 经度切片（兼容 0-360 和 -180-180） ──
    lon_vals = subset["lon"].values
    # 如果数据经度为 -180~180 而区域定义为 0-360
    if lon_vals.min() < 0 and lon_max > 180:
        _lon_min = ((lon_min + 180) % 360) - 180
        _lon_max = ((lon_max + 180) % 360) - 180
    else:
        _lon_min, _lon_max = lon_min, lon_max

    if _lon_min <= _lon_max:
        subset = subset.sel(lon=slice(_lon_min, _lon_max))
    else:
        # 跨越 0° 经线的情况
        subset = subset.where(
            (subset.lon >= _lon_min) | (subset.lon <= _lon_max), drop=True
        )

    # ── cos(lat) 权重 ──
    weights = np.cos(np.deg2rad(subset["lat"]))
    weights.name = "weights"

    # 纬度加权平均
    weighted_mean = subset.weighted(weights).mean(dim=("lat", "lon"))
    return weighted_mean


def compute_all_regional_stats(
    all_stats: Dict[str, Dict[str, xr.DataArray]],
    var: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    对所有模式、所有区域、所有 week 计算纬度加权平均 TCC 和 RMSE。

    返回
    -------
    dict 结构:
        {model_name: {
            region_name: {
                "tcc_week1": float,
                "tcc_week2": float, ...
                "rmse_week1": float,
                "rmse_week2": float, ...
            }
        }}
    """
    log(f"\n{'─'*60}")
    log(f" 区域纬度加权平均 [{var}]")
    log(f"{'─'*60}")

    import pandas as pd

    all_rows = []

    for model_name, stats in all_stats.items():
        for metric in ["tcc", "rmse"]:
            if metric not in stats:
                continue
            da = stats[metric]  # (week, lat, lon)
            week_values = da["week"].values
            for week_idx, week_name in enumerate(week_values):
                da_week = da.isel(week=week_idx)
                for region_name, region_bounds in REGIONS.items():
                    try:
                        mean_val = float(lat_weighted_regional_mean(da_week, region_bounds))
                    except Exception as e:
                        log(f"  [WARN] {region_name} week={week_name}: {e}", "WARN")
                        mean_val = np.nan
                    all_rows.append({
                        "variable": var,
                        "model": model_name,
                        "metric": metric,
                        "week": str(week_name),
                        "region": region_name,
                        "value": mean_val,
                    })

    df = pd.DataFrame(all_rows)

    # 保存 CSV
    region_dir = REGION_DIR / var
    region_dir.mkdir(parents=True, exist_ok=True)
    csv_path = region_dir / f"{var}_regional_weighted_mean.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log(f"    保存 → {csv_path}")

    # 转换为嵌套 dict
    result: Dict[str, Dict] = {}
    for model_name in all_stats:
        result[model_name] = {}
        model_df = df[df["model"] == model_name]
        for region_name in REGIONS:
            region_df = model_df[model_df["region"] == region_name]
            region_dict = {}
            for _, row in region_df.iterrows():
                key = f"{row['metric']}_{row['week']}"
                region_dict[key] = row["value"]
            result[model_name][region_name] = region_dict

    return result


# ── 绘图：全球空间对比图 ──────────────────────────────────────────────────────


def _week_label(week_val) -> str:
    """将 week 坐标转为显示标签。"""
    return str(week_val)


def plot_global_maps(
    all_stats: Dict[str, Dict[str, xr.DataArray]],
    var: str,
):
    """
    绘制 TCC 全球空间对比图。

    布局: 3 行 × N 列 (weeks)，使用 GridSpec 控制行间距
      - 第1行: my_model TCC
      - 第2行: other_model TCC
      - 第3行: 差值 (my_model − other_model)

    使用 Robinson 投影，Cartopy 绘制海岸线。
    """
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ##########################################################################添加↓#############################################
        import cartopy

        cartopy.config["pre_existing_data_dir"] = Path(
            "/home/nmic/project/cartopy"
        )
        cartopy.config["data_dir"] = Path(
            "/home/nmic/project/cartopy"
        )
        ##########################################################################################################################
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        log(f"  [WARN] 缺少绘图库 ({e})，跳过空间图", "WARN")
        return

    model_names = list(all_stats.keys())
    if len(model_names) < 2:
        log("  [WARN] 需要至少2个模式才能绘制差值图，跳过空间图", "WARN")
        return

    metric = "tcc"

    # 取第一个模式的 week 坐标
    first_model = model_names[0]
    all_weeks = list(all_stats[first_model][metric]["week"].values)

    if PLOT_WEEKS is not None:
        plot_weeks = [w for w in all_weeks if str(w) in PLOT_WEEKS or w in PLOT_WEEKS]
    else:
        plot_weeks = all_weeks

    n_weeks = len(plot_weeks)
    week_labels = [_week_label(w) for w in plot_weeks]
    proj = ccrs.Robinson()

    # ── 确保模型顺序: my_model 在上, other_model 居中 ──
    ordered_models = []
    for mn in ["my_model", "other_model"]:  # 替换为实际模型名称
        if mn in model_names:
            ordered_models.append(mn)
    for mn in model_names:
        if mn not in ordered_models:
            ordered_models.append(mn)

    # ── 构建差值数据 (my_model − other_model) ──
    m0, m1 = ordered_models[0], ordered_models[1]
    da_diff = all_stats[m0][metric] - all_stats[m1][metric]  # (week, lat, lon)
    # ── 提取各 week 数据为列表（适配 GridSpec 逐图绘制） ──
    t2m1_weeks = [all_stats[ordered_models[0]][metric].sel(week=w) for w in plot_weeks]
    t2m2_weeks = [all_stats[ordered_models[1]][metric].sel(week=w) for w in plot_weeks]
    diff_weeks = [da_diff.sel(week=w) for w in plot_weeks]

    model_label_1 = MODEL_SOURCES.get(ordered_models[0], {}).get("label", ordered_models[0])
    model_label_2 = MODEL_SOURCES.get(ordered_models[1], {}).get("label", ordered_models[1])

    # ── 设置不同的colorbar范围 ──
    vmin_common, vmax_common = -1.0, 1.0  # 前两行范围
    vmin_diff, vmax_diff = -0.5, 0.5      # 第三行范围

    log(f"  绘制 TCC 全球空间图 (3行×{n_weeks}列) ...")

    # 创建图形
    fig = plt.figure(figsize=(20, 16))

    # 使用GridSpec定义三行图片的位置
    gs1 = GridSpec(1, 6, top=0.90, bottom=0.55, hspace=0.25, wspace=0.1)
    gs2 = GridSpec(1, 6, top=0.78, bottom=0.43, hspace=0.25, wspace=0.1)
    gs3 = GridSpec(1, 6, top=0.66, bottom=0.20, hspace=0.25, wspace=0.1)

    # 绘制第一行：模式1 数据
    for i in range(n_weeks):
        ax = fig.add_subplot(gs1[0, i], projection=proj)
        im1 = ax.pcolormesh(t2m1_weeks[i].lon, t2m1_weeks[i].lat, t2m1_weeks[i],
                           vmin=vmin_common, vmax=vmax_common, cmap='RdBu_r',
                           transform=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
        ax.set_title(f'{model_label_1} - {week_labels[i]}', fontsize=12, pad=10)
        ax.set_global()

    # 绘制第二行：模式2 数据
    for i in range(n_weeks):
        ax = fig.add_subplot(gs2[0, i], projection=proj)
        im2 = ax.pcolormesh(t2m2_weeks[i].lon, t2m2_weeks[i].lat, t2m2_weeks[i],
                           vmin=vmin_common, vmax=vmax_common, cmap='RdBu_r',
                           transform=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
        ax.set_title(f'{model_label_2} - {week_labels[i]}', fontsize=12, pad=10)
        ax.set_global()

    # 绘制第三行：差值图
    for i in range(n_weeks):
        ax = fig.add_subplot(gs3[0, i], projection=proj)
        im_diff = ax.pcolormesh(diff_weeks[i].lon, diff_weeks[i].lat, diff_weeks[i],
                               vmin=vmin_diff, vmax=vmax_diff, cmap='RdBu_r',
                               transform=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
        ax.set_title(f'Difference - {week_labels[i]}', fontsize=12, pad=10)
        ax.set_global()

    # 添加两个colorbar
    # 第一行和第二行的colorbar
    cbar_ax1 = fig.add_axes([0.25, 0.53, 0.5, 0.015])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('TCC', fontsize=12, labelpad=10)

    # 第三行的colorbar
    cbar_ax2 = fig.add_axes([0.25, 0.35, 0.5, 0.015])
    cbar2 = fig.colorbar(im_diff, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('TCC-Difference', fontsize=12, labelpad=10)

    # ── 总标题 ──
    fig.suptitle(
        f"{var}  TCC  comparison against CRA_v1.5",
        fontsize=16, y=0.96,
    )

    fig_dir = FIGURE_DIR / var
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"{var}_TCC_global_maps.{FIGURE_FORMAT}"
    fig.savefig(out_path, dpi=MAP_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"    保存 → {out_path}")


# ── 绘图：区域柱状图 ──────────────────────────────────────────────────────────

def plot_regional_bars(
    all_regional: Dict[str, Dict[str, Dict[str, float]]],
    var: str,
):
    """
    绘制区域纬度加权平均的 TCC 和 RMSE 柱状图。

    采用 3×2 布局（6 个区域），每张子图中两个模式并排对比各 week。
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except ImportError:
        log("  [WARN] 缺少 matplotlib，跳过柱状图", "WARN")
        return

    model_names = list(all_regional.keys())
    if not model_names:
        log("  [WARN] 无模式数据，跳过柱状图", "WARN")
        return

    # 提取 week 标签，按数字排序（3,4,5,6,34,56）
    import re
    def _num_sort(w: str) -> int:
        m = re.search(r'(\d+)', str(w))
        return int(m.group(1)) if m else 0

    first_region = list(REGIONS.keys())[0]
    sample_keys = list(all_regional[model_names[0]].get(first_region, {}).keys())
    week_keys = sorted(set(
        k.split("_", 1)[1] for k in sample_keys if k.startswith("tcc_")
    ), key=_num_sort)
    if not week_keys:
        week_keys = sorted(set(
            k.split("_", 1)[1] for k in sample_keys if k.startswith("rmse_")
        ), key=_num_sort)

    # 只保留 PLOT_WEEKS 中指定的 week（保持 PLOT_WEEKS 的原始顺序）
    if PLOT_WEEKS is not None:
        week_keys = [w for w in PLOT_WEEKS if w in week_keys or str(w) in [str(k) for k in week_keys]]

    # week 显示标签（去掉 "week" 前缀）
    week_labels = [
        w[4:] if w.lower().startswith("week") else w
        for w in week_keys
    ]

    colors = {
        model_names[0]: "tab:red",
        model_names[1]: "tab:blue",
    } if len(model_names) >= 2 else {}

    n_regions = len(REGIONS)

    for metric in ["tcc", "rmse"]:
        log(f"  绘制区域 {metric.upper()} 柱状图 ...")

        fig = plt.figure(figsize=(11, 9))
        axes = []

        for idx, (region_name, region_bounds) in enumerate(REGIONS.items()):
            ax = fig.add_subplot(3, 2, idx + 1)
            axes.append(ax)

            x_positions = np.arange(len(week_labels))

            for offset_idx, model_name in enumerate(model_names):
                values = []
                region_data = all_regional[model_name].get(region_name, {})
                for wk in week_keys:
                    key = f"{metric}_{wk}"
                    values.append(region_data.get(key, np.nan))

                offset = -0.18 if offset_idx == 0 else 0.18
                ax.bar(
                    x_positions + offset, values, 0.36,
                    bottom=0,
                    color=colors.get(model_name, "gray"),
                    alpha=0.85,
                    label=MODEL_SOURCES.get(model_name, {}).get("label", model_name)
                    if idx == n_regions - 1 else "",
                )

            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(week_labels)
            ax.set_title(region_name)
            ax.set_ylabel(metric.upper())
            if metric == "tcc":
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            elif var == "ssrd":
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0), useMathText=True)
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if idx >= 4:
                ax.set_xlabel("Lead time (weeks)")

        # 隐藏多余子图（当 n_regions < 6 时）
        for idx in range(n_regions, 6):
            if idx < len(axes):
                axes[idx].set_visible(False)

        # 图例
        handles, legend_labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels,
            loc="lower center", ncol=len(model_names),
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.suptitle(
            f"{var}  {metric.upper()}  regional weighted mean",
            fontsize=16,
        )

        fig.subplots_adjust(
            top=0.91, bottom=0.10,
            wspace=0.25, hspace=0.32,
        )

        fig_dir = FIGURE_DIR / var
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / f"{var}_{metric}_regional_bars.{FIGURE_FORMAT}"
        fig.savefig(out_path, dpi=BAR_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        log(f"    保存 → {out_path}")


# ── 单变量主流程 ──────────────────────────────────────────────────────────────

def process_variable(
    var: str,
    init_dates: List[Tuple[int, str]],
    steps: List[str],
):
    """
    处理单个变量的完整评估流程。

    参数
    ----------
    var        : str   变量名
    init_dates : list  起报日期列表
    steps      : list  执行步骤 ["read", "stats", "regional", "plot"]
    """
    log(f"\n{'#'*60}")
    log(f"# 处理变量: {var}")
    log(f"# 起报日期: {len(init_dates)} 个")
    log(f"# 步骤: {steps}")
    log(f"{'#'*60}")

    all_stats = {}
    all_data = {}    # 保存合并后的数据以便后续使用
    all_regional = {}  # 区域平均结果

    # ── Step: read ──
    if "read" in steps:
        # 读取参考数据 (CRA_v1.5)
        ref_da = read_and_merge_source(
            REFERENCE_TEMPLATE, var, init_dates,
            need_interpolate=False,
            source_label="CRA_v1.5 (Reference)",
            path_var=REFERENCE_PATH_VAR,
        )

        # 读取各模式数据
        model_das = {}
        for model_name, model_cfg in MODEL_SOURCES.items():
            need_interp = (model_cfg["resolution"] < 1.0)
            model_da = read_and_merge_source(
                model_cfg["template"], var, init_dates,
                need_interpolate=need_interp,
                source_label=model_name,
                unit_conversion=model_cfg.get("unit_conversion") if var == "ssrd" else None,
                path_var=model_cfg.get("path_var"),
            )
            model_das[model_name] = model_da

        # 保存中间合并数据
        all_data["reference"] = ref_da
        all_data["models"] = model_das

    # ── Step: stats ──
    if "stats" in steps:
        if all_data.get("reference") is None or not all_data.get("models"):
            # 需要重新读取数据
            log("  自动读取数据以计算统计量...")
            ref_da = read_and_merge_source(
                REFERENCE_TEMPLATE, var, init_dates,
                need_interpolate=False,
                source_label="CRA_v1.5 (Reference)",
                path_var=REFERENCE_PATH_VAR,
            )
            model_das = {}
            for model_name, model_cfg in MODEL_SOURCES.items():
                need_interp = (model_cfg["resolution"] < 1.0)
                model_da = read_and_merge_source(
                    model_cfg["template"], var, init_dates,
                    need_interpolate=need_interp,
                    source_label=model_name,
                    unit_conversion=model_cfg.get("unit_conversion") if var == "ssrd" else None,
                )
                model_das[model_name] = model_da
            all_data["reference"] = ref_da
            all_data["models"] = model_das

        ref_da = all_data.get("reference")
        model_das = all_data.get("models", {})

        if ref_da is None or not model_das:
            log("  [ERROR] 无可用数据，无法计算统计量", "ERROR")
            return

        for model_name, model_da in model_das.items():
            log(f"\n  计算 {model_name} vs CRA_v1.5 ...")
            stats = compute_statistics(model_da, ref_da, model_label=model_name)
            all_stats[model_name] = stats
            save_stats_netcdf(stats, var, model_name)

    # ── Step: regional ──
    if "regional" in steps or "plot" in steps:
        if not all_stats:
            # 尝试从文件加载统计结果
            log("  尝试从文件加载已有统计结果...")
            for model_name in MODEL_SOURCES:
                loaded = load_stats_netcdf(var, model_name)
                if loaded:
                    all_stats[model_name] = loaded

        if not all_stats:
            log("  [ERROR] 无统计结果可用，无法计算区域平均", "ERROR")
            return

        all_regional = compute_all_regional_stats(all_stats, var)

    # ── Step: plot ──
    if "plot" in steps:
        if not all_stats:
            log("  [ERROR] 无统计结果可用，无法绘图", "ERROR")
            return

        # 全球空间图
        plot_global_maps(all_stats, var)

        # 区域柱状图
        if all_regional:
            plot_regional_bars(all_regional, var)
        else:
            log("  [WARN] 无区域平均数据，跳过柱状图", "WARN")

    log(f"\n  变量 {var} 处理完成！")
    return all_stats


# ── 命令行参数 ─────────────────────────────────────────────────────────────────

def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description=(
            "CRA_v1.5 作为观测场，评测 my_model 和 other_model 预报性能。\n"
            "计算 TCC/p-value/RMSE，区域纬度加权平均，绘制全球空间图和区域柱状图。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python CRA_v1.5_process.py                                       # 默认所有变量
  python CRA_v1.5_process.py --var tp                              # 单变量
  python CRA_v1.5_process.py --vars tp t2m w10                     # 批量处理
  python CRA_v1.5_process.py --vars tp --steps read stats           # 只读取+统计
  python CRA_v1.5_process.py --vars tp --steps plot                 # 只绘图（需已有统计文件）
  python CRA_v1.5_process.py --vars t2m --year-start 2021 --year-end 2024
        """,
    )
    p.add_argument(
        "--var", type=str, default=None,
        help=f"处理的变量名（默认: {DEFAULT_VARS[0]}）",
    )
    p.add_argument(
        "--vars", type=str, nargs="+", default=None,
        help=f"批量处理多个变量（默认: {DEFAULT_VARS}）",
    )
    p.add_argument(
        "--year-start", type=int, default=YEAR_START,
        help=f"起始年份（默认: {YEAR_START}）",
    )
    p.add_argument(
        "--year-end", type=int, default=YEAR_END,
        help=f"结束年份（默认: {YEAR_END}）",
    )
    p.add_argument(
        "--steps", type=str, nargs="+",
        default=DEFAULT_STEPS,
        choices=["read", "stats", "regional", "plot"],
        help=f"执行步骤（默认: {DEFAULT_STEPS}）",
    )
    p.add_argument(
        "--missing-mode", type=str, default=MISSING_FILE_MODE,
        choices=["nan", "skip"],
        help="缺失文件策略: nan=NaN填充, skip=跳过（默认: nan）",
    )
    p.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"输出根目录（默认: {OUTPUT_DIR}）",
    )
    p.add_argument(
        "--cra-path", type=str, default=CRA_PATH_TEMPLATE,
        help="CRA 参考数据路径模板",
    )
    p.add_argument(
        "--my-model-path", type=str, default=MY_MODEL_PATH_TEMPLATE,
        help="my_model 模式路径模板",
    )
    p.add_argument(
        "--other-model-path", type=str, default=OTHER_MODEL_PATH_TEMPLATE,
        help="Other model 模式路径模板",
    )
    return p.parse_args()


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main():
    """主入口：解析参数 → 生成起报日期 → 逐变量处理。"""
    args = parse_args()

    # 全局变量更新
    global MISSING_FILE_MODE, OUTPUT_DIR, STATS_DIR, REGION_DIR, FIGURE_DIR
    global CRA_PATH_TEMPLATE, MY_MODEL_PATH_TEMPLATE, OTHER_MODEL_PATH_TEMPLATE, REFERENCE_TEMPLATE, MODEL_SOURCES
    MISSING_FILE_MODE = args.missing_mode
    OUTPUT_DIR = Path(args.output_dir)
    STATS_DIR = OUTPUT_DIR / "stats"
    REGION_DIR = OUTPUT_DIR / "regional"
    FIGURE_DIR = OUTPUT_DIR / "figures"

    # 路径模板由编排器传入后，同步更新 MODEL_SOURCES 里的 template
    old_my_model = MY_MODEL_PATH_TEMPLATE
    old_other_model = OTHER_MODEL_PATH_TEMPLATE
    for cfg in MODEL_SOURCES.values():
        if cfg.get("template") == old_my_model:
            cfg["template"] = args.my_model_path
        elif cfg.get("template") == old_other_model:
            cfg["template"] = args.other_model_path
    CRA_PATH_TEMPLATE = args.cra_path
    MY_MODEL_PATH_TEMPLATE = args.my_model_path
    OTHER_MODEL_PATH_TEMPLATE = args.other_model_path
    REFERENCE_TEMPLATE = args.cra_path

    # 变量列表
    if args.vars:
        var_list = args.vars
    elif args.var:
        var_list = [args.var]
    else:
        var_list = DEFAULT_VARS

    log("=" * 60)
    log("CRA_v1.5 观测场评测系统")
    log(f"  变量: {var_list}")
    log(f"  年份: {args.year_start}-{args.year_end}")
    log(f"  步骤: {args.steps}")
    log(f"  缺失策略: {MISSING_FILE_MODE}")
    log(f"  输出目录: {OUTPUT_DIR}")
    log("=" * 60)

    # 生成起报日期
    log("\n── 生成起报日期 ──")
    init_dates = generate_init_dates_odd_days(args.year_start, args.year_end)
    log(f"  起报日期示例: {init_dates[:3]} ... {init_dates[-3:]}")
    log(f"  总计: {len(init_dates)} 个起报日")

    # 逐变量处理
    success = 0
    for var in var_list:
        try:
            process_variable(var, init_dates, args.steps)
            success += 1
        except Exception as e:
            log(f"\n  [ERROR] 变量 {var} 处理失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    log(f"\n{'='*60}")
    log(f"全部完成: {success}/{len(var_list)} 个变量成功。")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
