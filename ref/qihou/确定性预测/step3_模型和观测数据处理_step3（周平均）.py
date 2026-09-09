#!/usr/bin/env python3
"""
NetCDF 气象预报距平数据：将 lead_time 维度聚合为周平均。

适用范围：tp, t2m, ssr, ssrd 等变量。支持同时处理多个输入源（观测 + 模式）。
使用方式：修改配置区 TASKS 列表（每个任务显式指定 input_dir/output_dir）后直接运行。
"""

from __future__ import annotations

import logging
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

# ============================================================================
# 配置区 —— 添加/修改变量时主要修改此处（单独运行该代码时）
# ============================================================================

@dataclass
class TaskConfig:
    """一个周平均处理任务：读取某个输入目录的 lead_time 数据，输出到指定目录。"""
    name: str                           # nc 文件中的变量名，如 "t2m"
    input_dir: Path                     # 输入目录（已含变量/年份子目录）
    output_dir: Path                    # 输出目录
    year_filter: str = "2024"           # glob 文件名前缀
    output_name: str | None = None      # 输出变量名，None 则与 name 相同

    @property
    def out_name(self) -> str:
        return self.output_name or self.name


def build_tasks_from_args() -> list[TaskConfig]:
    """根据命令行参数构造任务列表（变量、年份、路径均来自编排器）。"""
    parser = argparse.ArgumentParser(description="lead_time → week 周平均")
    parser.add_argument("--var", type=str, default="t2m", help="变量名")
    parser.add_argument("--year", type=str, default="2024", help="年份")
    parser.add_argument("--obs-input", type=str, default="/gpu/zhouchg/liuzh/CRA_1.5/anom_combine_1p5/t2m/2024",
                        help="观测 lead_time 输入目录")
    parser.add_argument("--obs-output", type=str, default="/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/t2m",
                        help="观测周平均输出目录")
    parser.add_argument("--model-input", type=str, default="/gpu/zhouchg/liuzh/step2_output/anom_1p5/t2m",
                        help="模式 lead_time 输入目录")
    parser.add_argument("--model-output", type=str, default="/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/t2m",
                        help="模式周平均输出目录")
    args = parser.parse_args()

    return [
        # 观测：step1 输出 → CRA_v1.5 周平均
        TaskConfig(
            name=args.var,
            input_dir=Path(args.obs_input),
            output_dir=Path(args.obs_output),
            year_filter=args.year,
        ),
        # 模式：step2 输出 → my_model 周平均
        TaskConfig(
            name=args.var,
            input_dir=Path(args.model_input),
            output_dir=Path(args.model_output),
            year_filter=args.year,
        ),
    ]

# 维度名预期（若 nc 文件命名不一致，脚本会自动检测）
LEAD_TIME_DIM: str = "lead_time"
LAT_DIM: str = "lat"
LON_DIM: str = "lon"

# 周分组规则：(标签, 起始天索引0-based, 结束天索引0-based 闭区间)
# 例如 (0, 6) 表示第 1–7 天 → week1
WEEK_GROUPS: list[tuple[str, int, int]] = [
    ("week1",   0,  6),
    ("week2",   7, 13),
    ("week3",  14, 20),
    ("week4",  21, 27),
    ("week5",  28, 34),
    ("week6",  35, 41),
    ("week7",  42, 48),
    ("week8",  49, 55),
    ("week9",  56, 59),   # 最后一周不足 7 天
    ("week34", 14, 27),   # week3 + week4 合并
    ("week56", 28, 41),   # week5 + week6 合并
]

# ============================================================================
# 日志
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("weeklymean")


# ============================================================================
# 工具函数
# ============================================================================

def inspect_nc_file(file_path: Path) -> dict:
    """检查 nc 文件元信息并返回摘要 dict。"""
    ds = xr.open_dataset(file_path, decode_times=False)
    info: dict = {
        "variables": list(ds.data_vars),
        "dims": dict(ds.sizes),
        "coords": list(ds.coords),
        "var_attrs": {},
        "global_attrs": dict(ds.attrs),
    }
    for vname in ds.data_vars:
        info["var_attrs"][vname] = dict(ds[vname].attrs)
        info.setdefault("var_dims", {})[vname] = list(ds[vname].dims)
        info.setdefault("var_shape", {})[vname] = dict(zip(ds[vname].dims, ds[vname].shape))
    ds.close()
    return info


def find_variable(ds: xr.Dataset, expected: str) -> str:
    """在数据集中定位目标变量名，支持自动匹配。"""
    if expected in ds.data_vars:
        return expected
    # 大小写不敏感匹配
    lower = expected.lower()
    for v in ds.data_vars:
        if v.lower() == lower:
            logger.info("使用变量 '%s'（预期 '%s'）", v, expected)
            return v
    raise KeyError(f"找不到变量 '{expected}'，数据集变量: {list(ds.data_vars)}")


def find_dim(da: xr.DataArray, candidates: list[str]) -> str:
    """从候选名中确定实际维度名。"""
    for c in candidates:
        if c in da.dims:
            return c
    raise KeyError(f"找不到维度（候选: {candidates}），实际维度: {list(da.dims)}")


def normalize_dimensions(da: xr.DataArray, dim_map: dict[str, str]) -> xr.DataArray:
    """将维度名统一为规范名称。"""
    renames = {v: k for k, v in dim_map.items() if v in da.dims and v != k}
    if renames:
        da = da.rename(renames)
    return da


def infer_lead_day_index(da: xr.DataArray, lead_dim: str) -> np.ndarray:
    """
    推断 lead_time 对应的「第几天」索引（0-based）。

    返回 shape (lead_time_size,) 的整数数组，值 = 0, 1, 2, ...

    情况 1：有坐标值 1–60   → 返回坐标值 - 1
    情况 2：有坐标值 0–59   → 返回坐标值
    情况 3：无坐标          → 返回 range(0, size)
    """
    coord = da.coords.get(lead_dim)

    if coord is not None and len(coord.values) > 0:
        vals = np.asarray(coord.values, dtype=float)
        if np.all(np.isfinite(vals)):
            vmin, vmax = vals.min(), vals.max()
            if vmin >= 1 and vmax <= 120:
                # 假定为 1-based
                logger.info(
                    "lead_time 坐标值范围 [%.0f, %.0f]，判定为 1-based（第 %d-%d 天）",
                    vmin, vmax, int(vmin), int(vmax),
                )
                return (vals - 1).astype(int)
            elif vmin >= 0:
                logger.info(
                    "lead_time 坐标值范围 [%.0f, %.0f]，判定为 0-based（第 %d-%d 天）",
                    vmin, vmax, int(vmin) + 1, int(vmax) + 1,
                )
                return vals.astype(int)
            else:
                logger.warning("lead_time 坐标值异常（最小=%f），回退为按索引顺序", vmin)
                return np.arange(da.sizes[lead_dim])

    logger.info("lead_time 无坐标值，按索引顺序作为第 1–%d 天", da.sizes[lead_dim])
    return np.arange(da.sizes[lead_dim])


def build_week_groups(day_index: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    根据 day_index 构建周分组索引。

    返回 [(label, bool_mask), ...]，每个 mask 在 lead_time 轴上为 True/False。
    """
    groups: list[tuple[str, np.ndarray]] = []
    for label, start, end in WEEK_GROUPS:
        mask = (day_index >= start) & (day_index <= end)
        if mask.sum() == 0:
            logger.warning("周分组 '%s'（第 %d–%d 天）无匹配的 lead_time", label, start + 1, end + 1)
        groups.append((label, mask))
    return groups


def compute_weekly_mean(
    da: xr.DataArray, lead_dim: str, lat_dim: str, lon_dim: str,
    groups: list[tuple[str, np.ndarray]],
) -> xr.DataArray:
    """
    将 lead_time 轴聚合为 week 轴。

    返回 DataArray，维度 (week, lat, lon)。
    """
    weekly_arrays: list[np.ndarray] = []
    week_labels: list[str] = []

    for label, mask in groups:
        if mask.sum() == 0:
            continue
        sliced = da.isel(**{lead_dim: mask})
        mean_2d = sliced.mean(dim=lead_dim, skipna=True)
        weekly_arrays.append(mean_2d.values)
        week_labels.append(label)

    week_data = np.stack(weekly_arrays, axis=0)

    week_coord = xr.DataArray(
        week_labels,
        dims="week",
        attrs={
            "description": (
                "week1-9: 逐周平均 (7天/周, week9为4天); "
                "week34: week3+week4合并 (第15-28天); "
                "week56: week5+week6合并 (第29-42天)"
            ),
        },
    )

    da_out = xr.DataArray(
        week_data,
        dims=("week", lat_dim, lon_dim),
        coords={
            "week": week_coord,
            lat_dim: da.coords[lat_dim],
            lon_dim: da.coords[lon_dim],
        },
    )
    return da_out


def build_output_dataset(
    weekly_da: xr.DataArray,
    original_ds: xr.Dataset,
    var_name: str,
    out_var_name: str,
) -> xr.Dataset:
    """组装输出 Dataset，保留坐标并添加元信息。"""
    # 变量属性
    new_attrs: dict[str, str] = {
        "description": f"{out_var_name} weekly mean aggregated from lead_time",
        "source": var_name,
        "lead_time_aggregation": (
            "week1:day1-7, week2:day8-14, week3:day15-21, week4:day22-28, "
            "week5:day29-35, week6:day36-42, week7:day43-49, week8:day50-56, "
            "week9:day57-60, week34:day15-28, week56:day29-42"
        ),
        "created_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "skipna": "True",
    }

    # 继承原始变量属性（不覆盖上述新增属性）
    if var_name in original_ds.data_vars:
        for k, v in original_ds[var_name].attrs.items():
            new_attrs.setdefault(k, v)

    weekly_da.attrs = new_attrs

    ds_out = weekly_da.to_dataset(name=out_var_name)

    # 全局属性
    ds_out.attrs = dict(original_ds.attrs)
    ds_out.attrs["description"] = (
        f"Weekly mean of {var_name} aggregated from lead_time dimension."
    )
    ds_out.attrs["processing"] = "lead_time → week mean (skipna=True)"
    ds_out.attrs["created_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ds_out


def save_weekly_dataset(ds_out: xr.Dataset, output_path: Path) -> None:
    """保存结果 NetCDF，自动创建父目录。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 编码：保持与输入一致的压缩设置
    encoding = {}
    for vname in ds_out.data_vars:
        encoding[vname] = {"zlib": True, "complevel": 4}

    ds_out.to_netcdf(output_path, encoding=encoding)
    logger.info("已保存: %s", output_path)


# ============================================================================
# 单文件处理
# ============================================================================

def process_one_file(file_path: Path, task: TaskConfig, output_dir: Path) -> None:
    """处理单个 nc 文件：检查 → 周平均 → 保存。"""
    logger.info("-" * 50)
    logger.info("[%s] 处理文件: %s", task.name, file_path.name)

    # 1. 检查
    info = inspect_nc_file(file_path)
    logger.info("[%s] 变量: %s", task.name, info["variables"])
    logger.info("[%s] 维度: %s", task.name, info["dims"])

    # 2. 打开
    ds = xr.open_dataset(file_path, decode_times=False)

    # 3. 定位变量 & 维度
    var = find_variable(ds, task.name)
    da = ds[var]

    lead_dim = find_dim(da, [LEAD_TIME_DIM, "leadtime", "step", "time", "forecast_time"])
    lat_dim = find_dim(da, [LAT_DIM, "latitude", "y"])
    lon_dim = find_dim(da, [LON_DIM, "longitude", "x"])
    logger.info("[%s] 维度映射: lead=%s, lat=%s, lon=%s", task.name, lead_dim, lat_dim, lon_dim)

    # 检查 lead_time 数量
    n_lead = da.sizes[lead_dim]
    if n_lead < 60:
        logger.warning("[%s] lead_time 长度=%d，不足 60，week9 可能不完整", task.name, n_lead)

    # 4. 推断 day index
    day_index = infer_lead_day_index(da, lead_dim)

    # 5. 构建周分组 & 计算周平均
    groups = build_week_groups(day_index)
    weekly_da = compute_weekly_mean(da, lead_dim, lat_dim, lon_dim, groups)
    logger.info("[%s] 输出维度: %s", task.name, dict(zip(weekly_da.dims, weekly_da.shape)))

    # 6. 组装 & 保存
    ds_out = build_output_dataset(weekly_da, ds, var, task.out_name)
    output_path = output_dir / file_path.name
    save_weekly_dataset(ds_out, output_path)

    ds.close()


# ============================================================================
# 主入口
# ============================================================================

def main() -> None:
    """主流程：遍历任务列表，逐任务批量处理。"""
    tasks = build_tasks_from_args()
    logger.info("待处理任务: %s", [t.name for t in tasks])

    total_success, total_fail = 0, 0

    for task in tasks:
        input_dir = task.input_dir
        output_dir = task.output_dir

        logger.info("=" * 60)
        logger.info("变量: %-6s | 输入: %s", task.name, input_dir)
        logger.info("变量: %-6s | 输出: %s", task.name, output_dir)

        if not input_dir.is_dir():
            logger.error("输入目录不存在，跳过变量 '%s': %s", task.name, input_dir)
            total_fail += 1
            continue

        files = sorted(input_dir.glob(f"{task.year_filter}*.nc"))
        if not files:
            logger.warning("在 %s 中未找到匹配 '%s*.nc' 的文件，跳过", input_dir, task.year_filter)
            continue

        logger.info("[%s] 找到 %d 个 nc 文件", task.name, len(files))

        success, fail = 0, 0
        for fp in files:
            try:
                process_one_file(fp, task, output_dir)
                success += 1
            except Exception:
                logger.exception("[%s] 处理失败: %s", task.name, fp.name)
                fail += 1

        logger.info("[%s] 完成: 成功 %d, 失败 %d", task.name, success, fail)
        total_success += success
        total_fail += fail

    logger.info("=" * 60)
    logger.info("全部完成: 成功 %d, 失败 %d", total_success, total_fail)



if __name__ == "__main__":
    main()
