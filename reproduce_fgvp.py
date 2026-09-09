#!/usr/bin/env python
"""
FGVP 降水 TS 评测复现脚本

快速版：直接跑出结果，对比原脚本

用法：
    python reproduce_fgvp.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# 确保能导入新框架
sys.path.insert(0, '/workspace/szwCode/xmetai-evalation')

from xmetai_evaluation.io.fuxi_reader import FuXiReader, FuXiCatalog
from xmetai_evaluation.io.station_reader import DiamondStationReader, DiamondStationCatalog
from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator
from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator
from xmetai_evaluation.metrics.categorical import TSScore
from xmetai_evaluation.core.contracts import DataRequest, EvaluationBatch


def main():
    print("=" * 70)
    print("FGVP 降水 TS 评测 - 新框架复现")
    print("=" * 70)

    # 配置参数
    FORECAST_ROOT = Path("/workspace/data/shenzw/fgvp_output")
    STATION_DIR = Path("/workspace/data/worm/r0/2025")
    OUTPUT_CSV = Path("/workspace/szwCode/xmetai-evalation/results_fgvp_reproduce.csv")

    print(f"\n配置:")
    print(f"  预报根目录: {FORECAST_ROOT}")
    print(f"  站点目录: {STATION_DIR}")
    print(f"  输出文件: {OUTPUT_CSV}")

    # 扫描可用的起报日期
    print(f"\n扫描起报日期...")
    date_dirs = sorted([d for d in FORECAST_ROOT.iterdir()
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 8])

    if not date_dirs:
        print(f"✗ 未找到起报日期目录（格式 YYYYMMDD）")
        return 1

    print(f"  找到 {len(date_dirs)} 个起报日期")
    print(f"  范围: {date_dirs[0].name} 到 {date_dirs[-1].name}")

    # 转换为 datetime
    init_dates = [datetime.strptime(d.name, "%Y%m%d") for d in date_dirs[:5]]  # 先测试前5个
    print(f"\n本次测试: 前 5 个起报日期")
    for d in init_dates:
        print(f"    {d.strftime('%Y-%m-%d')}")

    # === 步骤 1: 初始化 Reader ===
    print(f"\n{'='*70}")
    print("步骤 1: 初始化数据读取器")
    print(f"{'='*70}")

    try:
        fgvp_reader, fgvp_catalog = FuXiReader.with_catalog(
            root_dir=FORECAST_ROOT,
            source_id="fgvp",
            step_hours=6.0,  # 假设 6h 步长，实际需要确认
        )
        print("  ✓ FGVP Reader 初始化完成")
    except Exception as e:
        print(f"  ✗ FGVP Reader 初始化失败: {e}")
        return 1

    try:
        station_reader, station_catalog = DiamondStationReader.with_catalog(
            station_dir=STATION_DIR,
            source_id="diamond_obs",
        )
        print("  ✓ 站点 Reader 初始化完成")
    except Exception as e:
        print(f"  ✗ 站点 Reader 初始化失败: {e}")
        return 1

    # === 步骤 2: 读取预报数据 ===
    print(f"\n{'='*70}")
    print("步骤 2: 读取预报数据")
    print(f"{'='*70}")

    forecast_request = DataRequest(
        source_id="fgvp",
        variables=["tp"],  # 或 "TP"，根据实际文件
        init_times=init_dates,
    )

    try:
        forecast_index = fgvp_catalog.discover(forecast_request)
        print(f"  ✓ 发现 {len(forecast_index.available[0])} 个起报时间的数据")

        forecast_bundle = fgvp_reader.read(forecast_request, forecast_index)
        print(f"  ✓ 预报数据读取完成")
        print(f"    形状: {forecast_bundle.payload['tp'].shape}")
        print(f"    维度: {list(forecast_bundle.payload['tp'].dims)}")
        print(f"    数据范围: {float(forecast_bundle.payload['tp'].min()):.2f} ~ {float(forecast_bundle.payload['tp'].max()):.2f}")
    except Exception as e:
        print(f"  ✗ 预报数据读取失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # === 步骤 3: 读取站点观测 ===
    print(f"\n{'='*70}")
    print("步骤 3: 读取站点观测")
    print(f"{'='*70}")

    obs_request = DataRequest(
        source_id="diamond_obs",
        variables=["precipitation"],
    )

    try:
        obs_index = station_catalog.discover(obs_request)
        print(f"  ✓ 发现 {len(obs_index.available)} 个站点文件")

        obs_bundle = station_reader.read(obs_request, obs_index)
        print(f"  ✓ 观测数据读取完成")
        print(f"    形状: {obs_bundle.payload['precipitation'].shape}")
        print(f"    站点数: {len(obs_bundle.payload.coords['station'])}")
        print(f"    时间范围: {obs_bundle.payload.coords['time'].values[0]} ~ {obs_bundle.payload.coords['time'].values[-1]}")
    except Exception as e:
        print(f"  ✗ 站点数据读取失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # === 步骤 4: 初始化转换器和指标 ===
    print(f"\n{'='*70}")
    print("步骤 4: 初始化转换器和指标")
    print(f"{'='*70}")

    interpolator = GridToStationInterpolator(method="bilinear")
    accumulator_24h = TimeWindowAccumulator(window_hours=24.0, time_dim="lead_time")

    thresholds = [
        ("≥0.1", 0.1),
        ("≥10", 10.0),
        ("≥25", 25.0),
        ("≥50", 50.0),
        ("≥100", 100.0),
        ("≥250", 250.0),
    ]
    ts_metric = TSScore(thresholds=thresholds)

    print(f"  ✓ 初始化完成")
    print(f"    插值方法: bilinear")
    print(f"    累积窗口: 24h")
    print(f"    阈值数量: {len(thresholds)}")

    # 提取站点坐标
    station_lats = obs_bundle.payload.coords['lat'].values
    station_lons = obs_bundle.payload.coords['lon'].values

    # === 步骤 5: 主循环处理 ===
    print(f"\n{'='*70}")
    print("步骤 5: 开始计算（逐起报×时效）")
    print(f"{'='*70}")

    all_states = []
    total_processed = 0
    total_skipped = 0

    for init_idx, init_time in enumerate(init_dates):
        print(f"\n[{init_idx+1}/{len(init_dates)}] 起报: {init_time.strftime('%Y-%m-%d %H:%M')}")

        # 提取该起报的预报
        forecast_init = forecast_bundle.payload['tp'].isel(init_time=init_idx)

        # 24h 累积
        try:
            forecast_24h = accumulator_24h.transform(forecast_init)
            available_leads = forecast_24h.coords['lead_time'].values
            print(f"  ✓ 24h累积完成，可用时效: {len(available_leads)} 个")
        except Exception as e:
            print(f"  ✗ 累积失败: {e}")
            continue

        # 只处理部分时效（加速测试）
        leads_to_process = available_leads[::2]  # 每隔一个
        print(f"  处理时效: {leads_to_process}")

        for lead_hour in leads_to_process:
            # 计算有效时刻（假设起报时间为 00 UTC）
            valid_time_utc = init_time + timedelta(hours=float(lead_hour))
            valid_time_bjt = valid_time_utc + timedelta(hours=8)

            # 在观测数据中查找最近的时刻
            try:
                # 使用最近邻匹配
                obs_times = obs_bundle.payload.coords['time'].values
                time_diffs = np.abs(obs_times - np.datetime64(valid_time_bjt))
                nearest_idx = np.argmin(time_diffs)
                obs_at_time = obs_bundle.payload['precipitation'].isel(time=nearest_idx)

                # 检查时间差是否在合理范围（24小时内）
                actual_time = pd.Timestamp(obs_times[nearest_idx])
                time_diff_hours = abs((actual_time - valid_time_bjt).total_seconds() / 3600)

                if time_diff_hours > 24:
                    print(f"    {lead_hour}h: 跳过（观测时间差 {time_diff_hours:.1f}h 过大）")
                    total_skipped += 1
                    continue

            except Exception as e:
                print(f"    {lead_hour}h: 跳过（无对应观测: {e}）")
                total_skipped += 1
                continue

            # 插值
            forecast_at_lead = forecast_24h.sel(lead_time=lead_hour)
            try:
                forecast_interp = interpolator.transform(
                    forecast_at_lead,
                    station_lats,
                    station_lons,
                )
            except Exception as e:
                print(f"    {lead_hour}h: 插值失败 - {e}")
                total_skipped += 1
                continue

            # 构建有效掩码
            valid_mask = xr.DataArray(
                np.isfinite(forecast_interp.values) & np.isfinite(obs_at_time.values),
                dims=["station"],
                coords={"station": forecast_interp.coords["station"]},
            )

            n_valid = int(valid_mask.sum().values)
            if n_valid < 10:  # 至少10个有效站点
                print(f"    {lead_hour}h: 跳过（有效站点太少: {n_valid}）")
                total_skipped += 1
                continue

            # 构建 Batch
            batch = EvaluationBatch(
                forecast=forecast_interp,
                observation=obs_at_time,
                sample_keys=[{"init": init_time, "lead": lead_hour}] * len(station_lats),
                valid_mask=valid_mask,
                alignment={"method": "bilinear", "time_diff_hours": time_diff_hours},
            )

            # 计算指标
            try:
                state = ts_metric.accumulate(batch)
                all_states.append(state)
                total_processed += 1
                print(f"    {lead_hour}h: ✓ 完成（{n_valid} 站点，时间差 {time_diff_hours:.1f}h）")
            except Exception as e:
                print(f"    {lead_hour}h: 计算失败 - {e}")
                total_skipped += 1

    # === 步骤 6: 合并结果 ===
    print(f"\n{'='*70}")
    print("步骤 6: 合并所有状态")
    print(f"{'='*70}")
    print(f"  成功: {total_processed} 个")
    print(f"  跳过: {total_skipped} 个")

    if not all_states:
        print("\n✗ 没有成功处理的数据")
        return 1

    merged_state = ts_metric.merge(all_states)
    final_result = ts_metric.finalize(merged_state)

    print(f"  ✓ 合并完成")
    print(f"  总有效样本: {final_result.n_valid}")

    # === 步骤 7: 输出结果 ===
    print(f"\n{'='*70}")
    print("步骤 7: 输出结果")
    print(f"{'='*70}")

    results_rows = []
    for threshold_name, metrics in final_result.value.items():
        print(f"\n{threshold_name}:")
        print(f"  TS:   {metrics['TS']:.6f}")
        print(f"  POD:  {metrics['POD']:.6f}")
        print(f"  FAR:  {metrics['FAR']:.6f}")
        print(f"  BIAS: {metrics['BIAS']:.6f}")
        print(f"  命中: {metrics['hits']}, 漏报: {metrics['misses']}, 空报: {metrics['false_alarms']}")

        results_rows.append({
            "window_h": 24,
            "lead_h": "AVG",
            "grade": threshold_name,
            "threshold_mm": metrics['threshold'],
            "hits": metrics['hits'],
            "misses": metrics['misses'],
            "false_alarms": metrics['false_alarms'],
            "n_pairs": metrics['n_pairs'],
            "TS": metrics['TS'],
            "POD": metrics['POD'],
            "FAR": metrics['FAR'],
            "miss_rate": metrics['miss_rate'],
            "BIAS": metrics['BIAS'],
        })

    # 保存
    df = pd.DataFrame(results_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ 结果已保存: {OUTPUT_CSV}")
    print(f"\n{'='*70}")
    print("完成！")
    print(f"{'='*70}")
    print(f"\n对比原结果:")
    print(f"  原代码结果: results2/en/ts_*.csv")
    print(f"  新框架结果: {OUTPUT_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
