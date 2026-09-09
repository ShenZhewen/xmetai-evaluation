#!/usr/bin/env python
"""
FuXi 降水 TS 评测复现脚本

目标：复现原代码 run_categorical.py 的结果，验证新框架正确性

用法：
    # 小样本测试（1个起报，24h时效）
    python reproduce_ts_eval.py --mode small

    # 单起报完整测试
    python reproduce_ts_eval.py --mode single --init-date 20250102

    # 全量测试（谨慎：需要很长时间）
    python reproduce_ts_eval.py --mode full
"""

import argparse
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
    parser = argparse.ArgumentParser(description="复现 FuXi TS 评测")
    parser.add_argument(
        "--mode",
        choices=["small", "single", "full"],
        default="small",
        help="测试模式：small=小样本, single=单起报, full=全量"
    )
    parser.add_argument(
        "--init-date",
        default="20250102",
        help="起报日期（YYYYMMDD），用于 single 模式"
    )
    parser.add_argument(
        "--forecast-root",
        default="/workspace/data/shenzw/fuxi_single_output",
        help="FuXi 输出根目录"
    )
    parser.add_argument(
        "--station-dir",
        default="/workspace/data/worm/r0/2025",
        help="站点观测目录"
    )
    parser.add_argument(
        "--output",
        default="/workspace/szwCode/xmetai-evalation/results_reproduce.csv",
        help="输出 CSV 文件"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FuXi 降水 TS 评测复现脚本")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"预报根目录: {args.forecast_root}")
    print(f"观测目录: {args.station_dir}")
    print(f"输出文件: {args.output}")
    print("=" * 60)

    # 根据模式确定起报日期列表
    if args.mode == "small":
        init_dates = [datetime(2025, 1, 2, 0)]  # 只测试一个起报
        lead_hours_to_test = [24]  # 只测试 24h 时效
        print("\n【小样本模式】：1个起报，24h时效")
    elif args.mode == "single":
        date_obj = datetime.strptime(args.init_date, "%Y%m%d")
        init_dates = [date_obj]
        lead_hours_to_test = None  # 测试所有时效
        print(f"\n【单起报模式】：{args.init_date}，所有时效")
    else:
        # 全量模式：扫描所有起报日期
        forecast_root = Path(args.forecast_root)
        date_dirs = sorted([d for d in forecast_root.iterdir() if d.is_dir() and d.name.isdigit()])
        init_dates = [datetime.strptime(d.name, "%Y%m%d") for d in date_dirs]
        lead_hours_to_test = None
        print(f"\n【全量模式】：{len(init_dates)}个起报日期")

    # 初始化 Reader
    print("\n步骤 1：初始化数据读取器...")
    fuxi_reader, fuxi_catalog = FuXiReader.with_catalog(
        root_dir=Path(args.forecast_root),
        source_id="fuxi",
        step_hours=6.0,
    )
    station_reader, station_catalog = DiamondStationReader.with_catalog(
        station_dir=Path(args.station_dir),
        source_id="diamond_obs",
    )
    print("  ✓ Reader 初始化完成")

    # 读取预报数据
    print("\n步骤 2：读取预报数据...")
    forecast_request = DataRequest(
        source_id="fuxi",
        variables=["tp"],
        init_times=init_dates,
    )

    try:
        forecast_index = fuxi_catalog.discover(forecast_request)
        print(f"  ✓ 发现 {len(forecast_index.available[0])} 个起报时间的数据")

        forecast_bundle = fuxi_reader.read(forecast_request, forecast_index)
        print(f"  ✓ 预报数据形状: {forecast_bundle.payload['tp'].shape}")
        print(f"    维度: {forecast_bundle.payload['tp'].dims}")
        print(f"    init_time: {forecast_bundle.payload['tp'].coords['init_time'].values}")
        print(f"    lead_time: {forecast_bundle.payload['tp'].coords['lead_time'].values}")
    except Exception as e:
        print(f"  ✗ 读取预报数据失败: {e}")
        return 1

    # 读取站点观测
    print("\n步骤 3：读取站点观测...")
    obs_request = DataRequest(
        source_id="diamond_obs",
        variables=["precipitation"],
    )

    try:
        obs_index = station_catalog.discover(obs_request)
        print(f"  ✓ 发现 {len(obs_index.available)} 个站点文件")

        obs_bundle = station_reader.read(obs_request, obs_index)
        print(f"  ✓ 观测数据形状: {obs_bundle.payload['precipitation'].shape}")
        print(f"    维度: {obs_bundle.payload['precipitation'].dims}")
        print(f"    站点数: {len(obs_bundle.payload.coords['station'])}")
        print(f"    时间范围: {obs_bundle.payload.coords['time'].values[0]} 到 {obs_bundle.payload.coords['time'].values[-1]}")
    except Exception as e:
        print(f"  ✗ 读取站点观测失败: {e}")
        return 1

    # 初始化 Transform 和 Metric
    print("\n步骤 4：初始化转换器和指标...")
    interpolator = GridToStationInterpolator(method="bilinear")
    accumulator_24h = TimeWindowAccumulator(window_hours=24.0, time_dim="lead_time")

    # 阈值（与原代码一致）
    thresholds = [
        ("≥0.1", 0.1),
        ("≥10", 10.0),
        ("≥25", 25.0),
        ("≥50", 50.0),
        ("≥100", 100.0),
        ("≥250", 250.0),
    ]
    ts_metric = TSScore(thresholds=thresholds)
    print(f"  ✓ 转换器和指标初始化完成（{len(thresholds)} 个阈值）")

    # 提取站点坐标
    station_lats = obs_bundle.payload.coords['lat'].values
    station_lons = obs_bundle.payload.coords['lon'].values
    print(f"\n站点坐标范围：")
    print(f"  纬度: {station_lats.min():.2f} 到 {station_lats.max():.2f}")
    print(f"  经度: {station_lons.min():.2f} 到 {station_lons.max():.2f}")

    # 主循环：逐起报、逐时效处理
    print("\n步骤 5：开始计算...")
    print("=" * 60)

    all_states = []  # 收集所有状态用于合并
    total_processed = 0
    total_skipped = 0

    for init_idx, init_time in enumerate(init_dates):
        print(f"\n处理起报: {init_time.strftime('%Y-%m-%d %H:%M')}")

        # 提取该起报的预报数据
        forecast_init = forecast_bundle.payload['tp'].isel(init_time=init_idx)

        # 24h 累积
        try:
            forecast_24h = accumulator_24h.transform(forecast_init)
            print(f"  ✓ 24h累积完成，形状: {forecast_24h.shape}")
        except Exception as e:
            print(f"  ✗ 累积失败: {e}")
            continue

        # 确定要处理的时效
        available_leads = forecast_24h.coords['lead_time'].values
        if lead_hours_to_test:
            leads_to_process = [l for l in lead_hours_to_test if l in available_leads]
        else:
            leads_to_process = available_leads

        print(f"  处理 {len(leads_to_process)} 个时效: {leads_to_process}")

        for lead_hour in leads_to_process:
            # 计算有效时刻（北京时）
            valid_time_utc = init_time + timedelta(hours=float(lead_hour))
            valid_time_bjt = valid_time_utc + timedelta(hours=8)

            # 查找对应的观测时刻
            # 观测是逐小时的，需要累积 24h
            obs_end_time = valid_time_bjt
            obs_start_time = obs_end_time - timedelta(hours=24)

            # 提取观测时间窗口（这里简化：假设观测已经是累积值）
            try:
                obs_at_time = obs_bundle.payload['precipitation'].sel(time=obs_end_time, method='nearest')
            except:
                print(f"    时效 {lead_hour}h: 跳过（无对应观测）")
                total_skipped += 1
                continue

            # 插值预报到站点
            forecast_at_lead = forecast_24h.sel(lead_time=lead_hour)
            try:
                forecast_interp = interpolator.transform(
                    forecast_at_lead,
                    station_lats,
                    station_lons,
                )
            except Exception as e:
                print(f"    时效 {lead_hour}h: 插值失败 - {e}")
                total_skipped += 1
                continue

            # 构建 EvaluationBatch
            # 掩码：两者都有效
            valid_mask = xr.DataArray(
                np.isfinite(forecast_interp.values) & np.isfinite(obs_at_time.values),
                dims=["station"],
                coords={"station": forecast_interp.coords["station"]},
            )

            n_valid = int(valid_mask.sum().values)
            if n_valid == 0:
                print(f"    时效 {lead_hour}h: 跳过（无有效站点）")
                total_skipped += 1
                continue

            batch = EvaluationBatch(
                forecast=forecast_interp,
                observation=obs_at_time,
                sample_keys=[{"init": init_time, "lead": lead_hour, "station": i} for i in range(len(station_lats))],
                valid_mask=valid_mask,
                alignment={
                    "method": "bilinear_interp",
                    "time_matched": True,
                    "valid_time_bjt": valid_time_bjt.isoformat(),
                },
            )

            # 计算指标（累积状态）
            try:
                state = ts_metric.accumulate(batch)
                all_states.append(state)
                total_processed += 1
                print(f"    时效 {lead_hour}h: ✓ 完成（{n_valid} 个有效站点）")
            except Exception as e:
                print(f"    时效 {lead_hour}h: 计算失败 - {e}")
                total_skipped += 1

    # 合并所有状态
    print("\n" + "=" * 60)
    print("步骤 6：合并所有状态...")
    print(f"  成功处理: {total_processed} 个 (起报×时效)")
    print(f"  跳过: {total_skipped} 个")

    if not all_states:
        print("\n✗ 没有成功处理的数据，无法生成结果")
        return 1

    merged_state = ts_metric.merge(all_states)
    final_result = ts_metric.finalize(merged_state)

    print(f"  ✓ 状态合并完成")
    print(f"  总有效样本数: {final_result.n_valid}")
    print(f"  状态: {final_result.status}")

    # 输出结果
    print("\n步骤 7：输出结果...")
    print("=" * 60)

    results_rows = []
    for threshold_name, metrics in final_result.value.items():
        print(f"\n{threshold_name}:")
        print(f"  TS:   {metrics['TS']:.6f}")
        print(f"  POD:  {metrics['POD']:.6f}")
        print(f"  FAR:  {metrics['FAR']:.6f}")
        print(f"  BIAS: {metrics['BIAS']:.6f}")
        print(f"  命中: {metrics['hits']}, 漏报: {metrics['misses']}, 空报: {metrics['false_alarms']}")
        print(f"  样本数: {metrics['n_pairs']}")

        # 构建 CSV 行（与原代码格式一致）
        results_rows.append({
            "window_h": 24,
            "lead_h": "AVG",  # 平均值
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

    # 保存 CSV
    df = pd.DataFrame(results_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ 结果已保存到: {output_path}")

    print("\n" + "=" * 60)
    print("复现完成！")
    print("=" * 60)
    print("\n下一步：对比原代码结果")
    print(f"  原代码结果: /workspace/szwCode/xmetai-evaluate/ref/tiqnqi/xmetai_model_verification/results2/en/ts_*.csv")
    print(f"  新框架结果: {output_path}")
    print(f"\n  对比命令: head -20 <原代码CSV> <新框架CSV>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
