# -*- coding: utf-8 -*-
"""2025 年暴雨过程个例检验（FuXi 确定性 × Diamond 站点）：43 个过程 × 两种口径。

口径 A 为逐日 08–08 北京时 24h 窗，口径 B 为过程累积窗；不分带，只出全球行。
"""
from datetime import datetime, timedelta

from xmetai_evaluation.configs.base import EvalConfig
from xmetai_evaluation.rainstorm_catalog import process_tuples

#: 降水 TS 阈值（mm）：模板 5 档 + ≥250 补档。
TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]

#: 冒烟开关：True 只跑 202521；False 跑全部 43 个过程。
SMOKE = False

#: (编号, 开始日, 结束日)，日期 YYYYMMDD，0–0 日历口径。
PROCESSES = process_tuples()

#: 数据源与输出根。
FUXI_ROOT = "/workspace/data/shenzw/fuxi_single_output"
STATION_ROOT = "/workspace/data/worm/r0/2025"
STATION_LIST = "/workspace/data/worm/r0/zd_sta_10285.dat"
OUTPUT_ROOT = "/workspace/szwCode/evaluation_results/rainstorm_ts_single_fuxi"

def _dense_leads(max_hours):
    """按 6h 密排到 max_hours（24h 窗是 4 个连续 6h 步加出来的，必须密排）。"""
    return [float(h) for h in range(6, int(max_hours) + 1, 6)]


cfgs = []
for _code, _start, _end in PROCESSES:
    _s = datetime.strptime(_start, "%Y%m%d")
    _e = datetime.strptime(_end, "%Y%m%d")
    _prev = (_s - timedelta(days=1)).strftime("%Y%m%d")
    _span_days = (_e - _s).days + 2  # 08–08 口径的全长天数（头尾各多 8 小时）
    _window_hours = _span_days * 24

    if SMOKE and _code != "202521":
        continue

    # ── 口径 A：逐日 24h（每个起报日单独一段）───────────────────────────
    # 默认采样规则在 6/12/18/24 中只筛出 24。
    _d = _s - timedelta(days=1)
    while _d <= _e:
        _init = _d.strftime("%Y%m%d")
        cfgs.append(
            EvalConfig(
                name=f"rainstorm_fuxi_{_code}_24h_{_init}",
                description=f"{_code} 过程 {_init} 起报 day-1 24h TS（{_init}08时→次日08时）",
                pipeline="weather_ts_det",
                forecast_reader={
                    "type": "fuxi",  # fuxi 目录布局：root/YYYYMMDD/NNN.nc
                    "root_dir": FUXI_ROOT,
                    "variable": "tp",
                    "step_hours": 6.0,
                    "lead_times": [6, 12, 18, 24],
                },
                observation_reader={
                    "type": "station",
                    "root_dir": STATION_ROOT,
                    "variable": "precipitation",
                    "station_list": STATION_LIST,
                },
                transform_options={"time_window_accumulator": {"window_hours": 24}},
                metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
                writers=["csv_long", "categorical_wide"],
                options={"local_utc_offset_hours": 8},  # 不分带，只出全球行
                start_date=_init,
                end_date=_init,  # 单起报：这个 08–08 窗只由这一根预报评
                limit=None,  # 限起报数，直接改这里；None = 不限
                output_dir=f"{OUTPUT_ROOT}/{_code}_24h/{_init}",
                log_level="INFO",
                execution={
                    "mode": "threads",  # 轻指标 → threads（同 weather_ts_single_fgvp）
                    "n_workers": 24,
                    "chunk_days": 1,
                    "lead_chunk_days": 1,  # 1 天时效 = 1 个 24h 采样点
                    "loads": {"observation": "resident"},
                    "resume": True,
                },
            )
        )
        _d += timedelta(days=1)

    # ── 口径 B：过程累积（S-1 日单起报，窗长 = 过程全长）───────────────
    # lead_times 密排到窗长；lead_chunk_days 一块装下整窗。
    cfgs.append(
        EvalConfig(
            name=f"rainstorm_fuxi_{_code}_total",
            description=f"{_code} 过程累积 TS（{_prev} 08时 起 {_span_days} 天）",
            pipeline="weather_ts_det",
            forecast_reader={
                "type": "fuxi",
                "root_dir": FUXI_ROOT,
                "variable": "tp",
                "step_hours": 6.0,
                "lead_times": _dense_leads(_window_hours),
            },
            observation_reader={
                "type": "station",
                "root_dir": STATION_ROOT,
                "variable": "precipitation",
                "station_list": STATION_LIST,
            },
            transform_options={
                "time_window_accumulator": {"window_hours": _window_hours}
            },
            metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
            writers=["csv_long", "categorical_wide"],
            options={"local_utc_offset_hours": 8},  # 不分带，只出全球行
            start_date=_prev,
            end_date=_prev,  # 单起报日
            limit=None,
            output_dir=f"{OUTPUT_ROOT}/{_code}_total",
            log_level="INFO",
            execution={
                "mode": "threads",
                "n_workers": 24,
                "chunk_days": 1,
                "lead_chunk_days": _span_days,  # 一块装下整个累积窗
                "loads": {"observation": "resident"},
                "resume": True,
            },
        )
    )
