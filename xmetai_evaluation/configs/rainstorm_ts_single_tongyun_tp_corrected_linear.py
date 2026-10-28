# -*- coding: utf-8 -*-
"""2025 年暴雨过程个例检验（通云 tp 线性订正版 × Diamond 站点）：43 个过程 × 两种口径。

数据源 ``/workspace/data/zhaoh/iwc_tongyun_tp_corrected_linear``：在通云单起报
输出上做的 tp 线性订正（zhaoh，2026-10 版），目录布局与 ``tongyun_single`` 相同
（``root/YYYYMMDD/NNN.nc``，每起报日 60 个 6h 步），只有 TP 一个变量，
721×1440 网格。单位与 ``tongyun_single`` 同为 mm（同日起报的场量级可比，
极值明显收敛），所以 **不加 scales**。

口径 A：每个 UTC 日一根 00UTC 起报，评 0–24h 的 24h 窗（= 08–08 北京时）。
口径 B：整段过程**只用 S 日 00UTC 一根起报**，逐日各评一次 24h（lead 24/48/…/N×24），
不把多日雨量累加成一个总量。日期一律按 UTC 整日理解。不分带，只出全球行。

产物根独立（``rainstorm_ts_single_tongyun_tp_corrected_linear``），
与未订正的通云档并列，便于报告里同口径对比。
"""
from datetime import datetime, timedelta

from xmetai_evaluation.configs.base import EvalConfig
from xmetai_evaluation.configs.rainstorm_ts_single_base import process_tuples

#: 降水 TS 阈值（mm）：模板 5 档 + ≥250 补档。
TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]

#: 冒烟开关：True 只跑 202521；False 跑全部 43 个过程。
SMOKE = False

#: (编号, 开始日, 结束日)，日期 YYYYMMDD，按 UTC 整日理解。
PROCESSES = process_tuples()

#: 数据源与输出根。
TONGYUN_ROOT = "/workspace/data/zhaoh/iwc_tongyun_tp_corrected_linear"
STATION_ROOT = "/workspace/data/worm/r0/2025"
STATION_LIST = "/workspace/data/worm/r0/zd_sta_10285.dat"
OUTPUT_ROOT = "/workspace/data/shenzw/evaluation_results/rainstorm_ts_single_tongyun_tp_corrected_linear"

#: 模型短名，进 cfg 名与产物段名。
MODEL = "tongyun_tp_corrected_linear"


def _dense_leads(max_hours):
    """按 6h 密排到 max_hours（24h 窗是 4 个连续 6h 步加出来的，必须密排）。"""
    return [float(h) for h in range(6, int(max_hours) + 1, 6)]


cfgs = []
for _code, _start, _end in PROCESSES:
    _s = datetime.strptime(_start, "%Y%m%d")
    _e = datetime.strptime(_end, "%Y%m%d")
    _start = _s.strftime("%Y%m%d")  # 过程起始日 = 口径 B 唯一一根起报
    _n_days = (_e - _s).days + 1  # UTC 整日口径：S–E 共 N 个 00–00 UTC 窗

    if SMOKE and _code != "202521":
        continue

    # ── 口径 A：逐日 24h（每个起报日单独一段）───────────────────────────
    # 默认采样规则在 6/12/18/24 中只筛出 24。
    _d = _s
    while _d <= _e:
        _init = _d.strftime("%Y%m%d")
        cfgs.append(
            EvalConfig(
                name=f"rainstorm_{MODEL}_{_code}_24h_{_init}",
                description=f"{_code} 过程 {_init} 起报 0–24h TS（{_init}08时→次日08时）",
                pipeline="weather_ts_det",
                forecast_reader={
                    "type": "fuxi",  # 通云落盘也用 fuxi 目录布局
                    "root_dir": TONGYUN_ROOT,
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
                    "mode": "threads",  # 轻指标 → threads（同 weather_ts_single_*）
                    "n_workers": 24,
                    "chunk_days": 1,
                    "lead_chunk_days": 1,  # 1 天时效 = 1 个 24h 采样点
                    "loads": {"observation": "resident"},
                    "resume": True,
                },
            )
        )
        _d += timedelta(days=1)

    # ── 口径 B：S 日单起报，逐日 24h 各评一次（不累加总量）─────────────
    # 窗长固定 24h；lead_times 密排到 N×24，在 24/48/…/N×24 各出一个采样点。
    # 目录仍叫 _total（沿用老产物路径），含义已变成「一根起报逐日各评」。
    cfgs.append(
        EvalConfig(
            name=f"rainstorm_{MODEL}_{_code}_total",
            description=f"{_code} 单起报逐日 TS（{_start} 08时起，{_n_days} 个 24h）",
            pipeline="weather_ts_det",
            forecast_reader={
                "type": "fuxi",
                "root_dir": TONGYUN_ROOT,
                "variable": "tp",
                "step_hours": 6.0,
                "lead_times": _dense_leads(_n_days * 24),
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
            start_date=_start,
            end_date=_start,  # 整段过程只这一根起报
            limit=None,
            output_dir=f"{OUTPUT_ROOT}/{_code}_total",
            log_level="INFO",
            execution={
                "mode": "threads",
                "n_workers": 24,
                "chunk_days": 1,
                "lead_chunk_days": _n_days,  # 一块装下 N 个逐日采样点
                "loads": {"observation": "resident"},
                "resume": True,
            },
        )
    )
