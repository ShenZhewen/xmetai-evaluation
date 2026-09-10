#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一评估入口。

CLI 只负责加载配置和启动任务；数据读取、时间窗口、空间对齐和指标状态
都遵守 xmetai_evaluation 的核心契约。
"""
import argparse
import gc
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from xmetai_evaluation.configs.base import load_config
from xmetai_evaluation.core.contracts import DataIndex, DataRequest, EvaluationBatch
from xmetai_evaluation.logging_util import configure_logging

log = logging.getLogger(__name__)


def create_reader(reader_config: dict):
    """根据配置创建 Reader 和 Catalog。"""
    reader_type = reader_config["type"]
    if reader_type == "fuxi":
        from xmetai_evaluation.io.fuxi_reader import FuXiCatalog, FuXiReader

        step_hours = float(reader_config.get("step_hours", 6.0))
        return (
            FuXiReader(
                source_id=reader_config.get("source_id", "fuxi"),
                step_hours=step_hours,
            ),
            FuXiCatalog(reader_config["root_dir"], step_hours=step_hours),
            reader_config,
        )

    if reader_type == "station":
        from xmetai_evaluation.io.station_reader import (
            DiamondStationCatalog,
            DiamondStationReader,
        )

        return (
            DiamondStationReader(source_id=reader_config.get("source_id", "diamond_station")),
            DiamondStationCatalog(reader_config["root_dir"]),
            reader_config,
        )

    raise ValueError(f"未知的 reader 类型: {reader_type}")


def create_transform(transform_config: dict):
    """根据配置创建一个 Transform。"""
    transform_type = transform_config["type"]
    if transform_type == "grid_to_station":
        from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator

        return GridToStationInterpolator(transform_config.get("method", "bilinear"))
    if transform_type == "time_window_accumulator":
        from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator

        return TimeWindowAccumulator(
            window_hours=float(transform_config["window_hours"]),
            time_dim=transform_config.get("time_dim", "lead_time"),
        )
    raise ValueError(f"未知的 transform 类型: {transform_type}")


def create_metric(metric_config: dict):
    """根据配置创建 Metric，并规范化阈值格式。"""
    if metric_config["type"] != "ts_score":
        raise ValueError(f"未知的 metric 类型: {metric_config['type']}")

    from xmetai_evaluation.metrics.categorical import TSScore

    thresholds = []
    for item in metric_config.get("thresholds", []):
        if isinstance(item, (tuple, list)):
            thresholds.append((str(item[0]), float(item[1])))
        else:
            value = float(item)
            thresholds.append((f"≥{item:g}", value))
    return TSScore(thresholds=thresholds)


def _parse_file_time(path: Path) -> datetime:
    """解析 Diamond 文件的北京时间时刻。"""
    name = path.stem
    if len(name) == 10 and name.isdigit():
        return datetime.strptime(name, "%Y%m%d%H")
    if len(name) == 2 and name.isdigit() and path.parent.name.isdigit():
        return datetime.strptime(path.parent.name + name, "%Y%m%d%H")
    raise ValueError(f"无法解析观测文件时间: {path}")


def _make_request(source_id: str, variable: str, init_times=None) -> DataRequest:
    return DataRequest(source_id=source_id, variables=[variable], init_times=init_times)


def _select_observation_window(
    observation: xr.Dataset,
    end_time: datetime,
    window_hours: int,
) -> xr.DataArray:
    """按参考实现取北京时闭区间 [V-W+1h, V] 的完整观测窗口。"""
    expected = [end_time - timedelta(hours=i) for i in range(window_hours - 1, -1, -1)]
    available = pd.DatetimeIndex(observation.time.values)
    expected64 = np.asarray(expected, dtype="datetime64[ns]")
    positions = available.get_indexer(expected64)
    if np.any(positions < 0):
        missing = [str(expected[i]) for i, p in enumerate(positions) if p < 0]
        raise ValueError(f"观测窗口不完整，缺少 {len(missing)} 个时次（如 {missing[:3]}）")

    values = observation["precipitation"].isel(time=positions)
    # 缺任意小时即整站剔除；0 降水是有效记录。
    complete = np.isfinite(values).all(dim="time")
    return values.sum(dim="time", skipna=False).where(complete)


def _write_metric_results(
    metric_results: List[Tuple[float, Any]],
    output_dir: Path,
    window_hours: int,
) -> Path:
    """按参考输出格式写出逐窗口、逐时效的长表。"""
    rows = []
    for lead, result in metric_results:
        for grade, values in result.value.items():
            rows.append({
                "window_h": window_hours,
                "lead_h": int(lead) if float(lead).is_integer() else float(lead),
                "grade": grade,
                "threshold_mm": values["threshold"],
                "hits": values["hits"],
                "misses": values["misses"],
                "false_alarms": values["false_alarms"],
                "n_pairs": values["n_pairs"],
                "TS": values["TS"],
                "POD": values["POD"],
                "FAR": values["FAR"],
                "漏报率": values["miss_rate"],
                "BIAS": values["BIAS"],
            })
    if not rows:
        raise ValueError("没有可写出的指标结果")
    columns = [
        "window_h", "lead_h", "grade", "threshold_mm", "hits", "misses",
        "false_alarms", "n_pairs", "TS", "POD", "FAR", "漏报率", "BIAS",
    ]
    path = output_dir / "ts_fuxi.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def run_evaluation(cfg) -> int:
    """执行确定性预报与站点观测配对评估。"""
    started = perf_counter()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 80)
    log.info("开始评估任务: %s", cfg.name)
    log.info("描述: %s", cfg.description)

    forecast_reader, forecast_catalog, forecast_cfg = create_reader(cfg.forecast_reader)
    observation_reader, observation_catalog, observation_cfg = create_reader(cfg.observation_reader)

    start, end = cfg.parse_date_range()
    if start is None:
        raise ValueError("评估配置必须提供 start_date")
    if end is None:
        end = start

    # FuXiCatalog 的契约要求 request.init_times；先提交配置日期范围，
    # Catalog 会跳过不存在的日期，再从返回索引取得真实可用起报。
    requested_times = []
    current = start
    while current <= end:
        requested_times.append(current)
        current += timedelta(days=1)
    discovery_request = _make_request(
        forecast_cfg.get("source_id", "fuxi"),
        forecast_cfg.get("variable", "tp"),
        requested_times,
    )
    discovered_index = forecast_catalog.discover(discovery_request)
    if not discovered_index.available:
        raise ValueError("预报目录没有发现可用起报时间")
    discovered_by_init = discovered_index.available[0]
    init_times = sorted(discovered_by_init)
    if cfg.limit:
        init_times = init_times[: cfg.limit]
    if not init_times:
        raise ValueError(
            f"时间范围 {start} 到 {end} 内没有可用预报；"
            f"实际可用范围为 {min(discovered_by_init)} 到 {max(discovered_by_init)}"
        )
    log.info("起报时间: %s 到 %s，共 %d 个", init_times[0], init_times[-1], len(init_times))

    forecast_var = forecast_cfg.get("variable", "tp")
    obs_var = observation_cfg.get("variable", "precipitation")
    source_fcst = forecast_cfg.get("source_id", "fuxi")
    source_obs = observation_cfg.get("source_id", "diamond_station")

    # 预报按起报逐个读取，避免全年数据同时驻留内存。
    log.info("采用流式读取预报：每次处理一个起报")
    t0 = perf_counter()
    forecast_request = _make_request(source_fcst, forecast_var, init_times)
    forecast_index = forecast_catalog.discover(forecast_request)
    lead_max = max(
        len(paths) * float(getattr(forecast_reader, "step_hours", 6.0))
        for paths in forecast_index.available[0].values()
    )
    log.info("预报文件发现完成：%d 个起报，最大时效 %.0fh，耗时 %.2fs",
             len(init_times), lead_max, perf_counter() - t0)

    # 一次读取覆盖所有起报和时效的观测；后续窗口只做内存索引。
    lead_max = int(lead_max)
    window_hours = int(next(
        (t.get("window_hours", 24) for t in cfg.transforms
         if t.get("type") == "time_window_accumulator"),
        24,
    ))
    obs_start = min(init_times) + timedelta(hours=8) + timedelta(hours=1 - window_hours)
    obs_end = max(init_times) + timedelta(hours=8 + lead_max)
    obs_files = []
    log.info("正在扫描观测文件...")
    t0 = perf_counter()
    obs_request = _make_request(source_obs, obs_var)
    obs_index_all = observation_catalog.discover(obs_request)
    for path in obs_index_all.available:
        path_time = _parse_file_time(Path(path))
        if obs_start <= path_time <= obs_end:
            obs_files.append(path)
    obs_files.sort(key=lambda p: _parse_file_time(Path(p)))
    log.info("观测窗口覆盖 %s 到 %s，共 %d 个文件", obs_start, obs_end, len(obs_files))
    if not obs_files:
        raise ValueError("没有找到评估所需的观测文件")
    observation_index = DataIndex(source_id=source_obs, available=obs_files)
    log.info("正在读取观测数据...")
    observation_bundle = observation_reader.read(obs_request, observation_index)
    observation_ds = observation_bundle.payload
    log.info("观测读取完成: %s，耗时 %.2fs", observation_ds.sizes, perf_counter() - t0)

    forecast_accumulator = create_transform(next(
        t for t in cfg.transforms if t.get("type") == "time_window_accumulator"
    ))
    interpolator = create_transform(next(
        t for t in cfg.transforms if t.get("type") == "grid_to_station"
    ))
    metrics = [create_metric(m) for m in cfg.metrics]
    # 每个指标、每个有效预报时效分别维护状态；参考结果按 lead_h 输出，
    # 不能把不同 lead 的列联表合并后再写成一行。
    states: Dict[int, Dict[int, List[Any]]] = {
        metric_idx: {} for metric_idx in range(len(metrics))
    }
    station_lats = observation_ds["lat"].values
    station_lons = observation_ds["lon"].values

    # 24h 窗口按参考实现只评估完整窗口结束时效：24、48、72...。
    # accumulator 仍负责生成滑动窗口，执行器负责评估协议中的 lead 选择。
    window_leads = None
    processed = 0
    skipped = 0

    # 添加进度条
    pbar = tqdm(
        enumerate(init_times),
        total=len(init_times),
        desc="评估进度",
        unit="起报",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for init_idx, init_time in pbar:
        pbar.set_description(f"处理 {init_time.strftime('%Y-%m-%d')}")
        try:
            forecast_bundle = forecast_reader.read_one(forecast_request, forecast_index, init_time)
            forecast_ds = forecast_bundle.payload
            forecast_init = forecast_ds[forecast_var].isel(init_time=0)
            forecast_windows = forecast_accumulator.transform(forecast_init)
        except Exception as exc:
            skipped += 1
            log.exception("起报 %s 预报读取失败: %s", init_time, exc)
            continue
        if window_leads is None:
            window_leads = [
                float(lead) for lead in forecast_windows.lead_time.values
                if float(lead) >= window_hours
                and abs(float(lead) % window_hours) < 1e-6
            ]
            log.info("评估时效: %s", window_leads)
        for lead in window_leads:
            if lead not in forecast_windows.lead_time.values:
                continue
            valid_bjt = init_time + timedelta(hours=float(lead) + 8)
            try:
                obs_window = _select_observation_window(observation_ds, valid_bjt, window_hours)
                forecast_at_lead = forecast_windows.sel(lead_time=lead)
                forecast_at_station = interpolator.transform(
                    forecast_at_lead, station_lats, station_lons
                )
                valid_mask = xr.DataArray(
                    np.isfinite(forecast_at_station.values) & np.isfinite(obs_window.values),
                    dims=forecast_at_station.dims,
                    coords=forecast_at_station.coords,
                )
                n_valid = int(valid_mask.sum())
                if n_valid == 0:
                    skipped += 1
                    continue
                batch = EvaluationBatch(
                    forecast=forecast_at_station,
                    observation=obs_window,
                    sample_keys=[{"init": init_time.isoformat(), "lead": float(lead)}] * len(station_lats),
                    valid_mask=valid_mask,
                    alignment={
                        "method": "bilinear",
                        "observation_timezone": "Asia/Shanghai",
                        "window_hours": window_hours,
                        "valid_time_bjt": valid_bjt.isoformat(),
                    },
                )
                for metric_idx, metric in enumerate(metrics):
                    metric.validate(batch)
                    states[metric_idx].setdefault(int(lead), []).append(metric.accumulate(batch))
                processed += 1
            except Exception as exc:
                skipped += 1
                log.debug("  lead=%sh 失败: %s", lead, exc)

    pbar.close()
    del forecast_ds, forecast_bundle
    gc.collect()

    if processed == 0:
        log.error("没有成功处理任何评测批次")
        return 1

    for metric_idx, metric in enumerate(metrics):
        metric_results = []
        for lead, lead_states in sorted(states[metric_idx].items()):
            merged = metric.merge(lead_states)
            metric_results.append((lead, metric.finalize(merged)))
        path = _write_metric_results(metric_results, output_dir, window_hours)
        log.info("指标 %s 完成，结果写入 %s", metric.name, path)

    log.info("评估完成：成功批次=%d，跳过=%d，总耗时=%.1fs", processed, skipped, perf_counter() - started)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="xmetai-evaluation 气象模型评估框架")
    parser.add_argument("--config", required=True, help="配置名称或 Python 配置路径")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.start_date:
            cfg.start_date = args.start_date
        if args.end_date:
            cfg.end_date = args.end_date
        if args.limit is not None:
            cfg.limit = args.limit
        if args.log_level:
            cfg.log_level = args.log_level
        configure_logging(cfg.log_level, Path(args.log_file) if args.log_file else None)
        return run_evaluation(cfg)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("用户中断")
        return 130
    except Exception:
        logging.getLogger(__name__).exception("评估失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
