# -*- coding: utf-8 -*-
"""暴雨个例 rainstorm（SMOKE 过程 202521）真实数据档：跑正式 config，对标 run_categorical。

两个 config（``rainstorm_ts_single_fuxi`` / ``rainstorm_ts_single_fgvp``）都是
导出 ``cfgs`` 列表——43 个过程 × 两种口径（口径 A 逐日 24h 按起报日拆段；口径 B 只用
S 日一根起报、逐日各评 24h），本模块只挑出 SMOKE 过程 202521 的 8 段
（口径 A 7 个起报日 0723–0729 + 口径 B 1 段），覆盖数据路径指向最小 fixture 后逐段跑，
对拍 ``run_categorical.py``（确定性 TS）的整数列联表。

数据走 ``rainstorm_test_data`` fixture（原裁切脚本 ``scripts/cut_rainstorm_fixture.py``
已不在仓库里）：起报 20250722~20250729、7/22 有 32 时效、其余起报日 4 时效、
192 个小时站点实况 + 白名单。两套模型（fuxi / fgvp）各一份。
⚠ **口径 B 改成逐日窗后，fixture 需要 S=20250723 起报的 lead 24/48/…/168h 共 7 个
时效**；旧的 fixture 只有 4 时效，重新裁切前 ``test_lead_parity`` 会失败。

断言分三层：

- **coverage**：挑出的段数与起报日集合必须对齐 ``rainstorm_ts_single_base.expected_days``，
  每段产物齐全、无硬错误（failed）——拦"段被漏掉 / 拆段口径变了"。
- **golden 对表**：口径 A 用 7/23 起报、口径 B 用它 lead 24h 那一窗（= 同一根 0723
  起报的 0–24h，与口径 A 的 0723 段逐格相同），逐阈值对 ``run_categorical.py`` 的
  整数列联表（hits / false_alarms / misses）。其余段逻辑相同，靠 coverage + 自洽哨
  兜底，不逐段对表。
- **自洽哨**：全部段的 ``categorical_wide.csv`` 整数列联表必须能反推出
  ``scores.csv`` 里的 ts / pod / far / miss_rate / frequency_bias（1e-9）——拦两个
  writer 的取数口径分叉。

⚠ 标准答案内联在下面，来自 ``run_categorical.py`` 确定性 TS（口径 A 单起报 7/23 的
24h 窗；口径 B 的 lead 24h 窗与它是同一根起报、同一窗）——已逐格对过，框架与基准一致。
golden 对表在 ``test_contingency_parity``，其余段逻辑相同、靠 coverage + 自洽哨兜底。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xmetai_evaluation.configs.rainstorm_ts_single_base import expected_days, window_days

pytestmark = pytest.mark.realdata

#: 被测过程（特强 202521）。config 的 SMOKE 开关当前是 False（全量 43 过程），
#: 测试加载全部段后按 name 前缀自行筛选这一个过程。
CODE = "202521"

#: golden 只对这两档阈值（其余阈值靠自洽哨兜底，与 test_precip_ts_realdata 同口径）。
PARITY_THRESHOLDS = (0.1, 10.0)

#: 最小 fixture 根（scripts/cut_rainstorm_fixture.py 裁出）。换机器/换位置改这一行
#: 或设环境变量 XMETAI_FIXTURE_DIR。
FIXTURE_DIR = os.environ.get("XMETAI_FIXTURE_DIR", "/workspace/data/shenzw")
FIXTURE_ROOT = f"{FIXTURE_DIR}/rainstorm_test_data"

#: config 名 -> (模型短名, fixture 里的预报子目录)。
MODEL_TABLE = {
    "rainstorm_ts_single_fuxi": ("fuxi", "fuxi_single_output"),
    "rainstorm_ts_single_fgvp": ("fgvp", "fgvp_output"),
}

# 标准答案（内联）：run_categorical.py 确定性 TS 的整数列联表，逐格一致、未四舍五入。
# 可审计复现（zhaoh_xmetai_model_verification/run_categorical.py，var=TP、tz_shift=8、
# bilinear、6 档阈值 [0.1,10,25,50,100,250]、同一站表）：
#   对「只含 20250723 单日」的 root 跑 window_ts=[24]。
# 口径 A 的 20250723 段与口径 B 的 lead 24h 窗是同一根起报、同一个 0–24h 窗，
# 所以只需要这一份标准答案，两边都拿它对表。
# n_pairs 是总配对站数（含正确否定），钉住它拦「站表筛选/时间窗错位/插值 NaN 导致
# 样本集合漂移」这类列联表三格不变、但分母漂了的回归。
GOLDEN_COUNTS = {
    "fuxi": {
        "24h_20250723": {
            0.1: {"hits": 5245, "false_alarms": 3555, "misses": 41, "n_pairs": 10132},
            10.0: {"hits": 1153, "false_alarms": 1219, "misses": 732, "n_pairs": 10132},
        },
    },
    "fgvp": {
        "24h_20250723": {
            0.1: {"hits": 5228, "false_alarms": 3257, "misses": 58, "n_pairs": 10132},
            10.0: {"hits": 906, "false_alarms": 860, "misses": 979, "n_pairs": 10132},
        },
    },
}


def _run_model(config_name: str, model: str, forecast_subdir: str, tmp_root: Path) -> dict:
    """挑出 202521 的 8 段，覆盖数据路径与并发形态后逐段跑，返回 {段名: 输出目录}。"""
    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    forecast_root = f"{FIXTURE_ROOT}/{forecast_subdir}"
    station_root = f"{FIXTURE_ROOT}/station"
    station_list = f"{FIXTURE_ROOT}/station_list.dat"
    # fixture 不齐就 skip。fixture 根由 FIXTURE_DIR 统一覆盖，不再按数据源逐个覆盖
    # （FORECAST 这类单源变量对 fuxi/fgvp 是不同路径，共用同名会互相串）。
    for name, path in (
        ("forecast", forecast_root),
        ("station", station_root),
        ("station_list", station_list),
    ):
        if not Path(path).exists():
            pytest.skip(f"{name}={path} 不存在")

    cfgs = load_config(config_name)
    assert isinstance(cfgs, list), f"{config_name} 导出的是 cfgs 列表，不是单个 cfg"
    prefix = f"rainstorm_{model}_{CODE}_"
    segs = [c for c in cfgs if c.name.startswith(prefix)]
    expected_n = len(expected_days(CODE)) + 1  # 口径 A 每日一段 + 口径 B 一段
    assert len(segs) == expected_n, (
        f"202521 应挑出 {expected_n} 段，实有 {[c.name for c in segs]}"
    )

    results = {}
    for seg in segs:
        out = tmp_root / seg.name
        seg.forecast_reader = {**seg.forecast_reader, "root_dir": forecast_root}
        seg.observation_reader = {
            **seg.observation_reader,
            "root_dir": station_root,
            "station_list": station_list,
        }
        seg.output_dir = str(out)
        seg.execution = {**seg.execution, "mode": "serial", "n_workers": 1, "resume": False}
        run_evaluation(seg)
        results[seg.name] = out
    return results


@pytest.fixture(scope="session", params=list(MODEL_TABLE))
def model_results(tmp_path_factory, request):
    """每个模型跑一遍 8 段，本模块所有用例共用。"""
    config_name = request.param
    model, forecast_subdir = MODEL_TABLE[config_name]
    results = _run_model(config_name, model, forecast_subdir, tmp_path_factory.mktemp(model))
    return model, results


def _wide(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "diagnostics" / "categorical_wide.csv")


def _scores(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "scores.csv")


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------

def test_segment_coverage(model_results):
    """挑出的段数、起报日集合必须对齐 rainstorm_ts_single_base；每段产物齐全、无硬错误。"""
    model, results = model_results
    expected_a = {f"rainstorm_{model}_{CODE}_24h_{d}" for d in expected_days(CODE)}
    expected_b = f"rainstorm_{model}_{CODE}_total"
    assert set(results) == expected_a | {expected_b}, (
        f"{model} 段集合不对：\n实有 {sorted(results)}\n预期 {sorted(expected_a | {expected_b})}"
    )

    for seg_name, out in results.items():
        scores = _scores(out)
        # 单起报日 6 档阈值里 ≥100/≥250 常无事件 → TS=NaN → status 是 partial（预期，
        # 见 metrics/categorical.py 的整体状态判定）。success/partial 都算正常，只拒
        # failed（硬错误）与 no_valid_data（完全没样本）。
        assert scores["status"].isin(["success", "partial"]).all(), (
            f"{seg_name} 有异常状态: {sorted(scores['status'].unique())}"
        )
        assert scores["variable"].eq("precipitation").all()
        assert scores["region"].isna().all(), "暴雨个例不分纬度带，只出全球行"


def test_lead_parity(model_results):
    """口径 B 是 S 单起报、逐日各评一次 24h：采样点必须是 24/48/…/N×24h。"""
    model, results = model_results
    total = results[f"rainstorm_{model}_{CODE}_total"]
    # 天数不在产物里，从 lead_h 反推：每个 UTC 日一个 24h 窗、窗不重叠，
    # lead_h 集合应恰好是 {24h, 48h, …, N×24h}，多了少了都是采样口径漂了。
    scores = _scores(total)
    expected = {float(24 * k) for k in range(1, window_days(CODE) + 1)}
    assert set(scores["lead_h"]) == expected, (
        f"口径 B lead_h {sorted(scores['lead_h'].unique())} != 逐日窗 {sorted(expected)}"
    )


# --------------------------------------------------------------------------
# golden 对表
# --------------------------------------------------------------------------

def test_contingency_parity(model_results):
    """口径 A 代表段（7/23）与口径 B 第一窗（lead 24h）的列联表逐格对 run_categorical。

    口径 B 的 lead 24h 窗就是同一根 20250723 起报的 0–24h 预报，理应跟口径 A 的
    20250723 段逐格相同——这同时钉住「B 的第一天 = A 同一天」这一口径等价关系。
    """
    model, results = model_results
    for label, seg_name, lead in (
        ("24h_20250723", f"rainstorm_{model}_{CODE}_24h_20250723", None),
        ("24h_20250723", f"rainstorm_{model}_{CODE}_total", 24.0),
    ):
        wide = _wide(results[seg_name])
        for thr in PARITY_THRESHOLDS:
            rows = wide[wide["threshold_mm"] == thr]
            if lead is not None:
                rows = rows[np.isclose(rows["lead_h"], lead)]
            assert len(rows) == 1, f"{seg_name} ≥{thr}（lead {lead}）有 {len(rows)} 行"
            actual = {
                c: int(rows[c].iloc[0])
                for c in ("hits", "false_alarms", "misses", "n_pairs")
            }
            expected = GOLDEN_COUNTS[model][label][thr]
            for column, want in expected.items():
                assert actual[column] == want, (
                    f"{model} {label} ≥{thr} 的 {column}: 框架 {actual[column]} vs 基准 {want}"
                )


# --------------------------------------------------------------------------
# 自洽哨
# --------------------------------------------------------------------------

def test_consistency(model_results):
    """全部段的派生指标必须能由整数列联表算回来（无事件阈值指标是 NaN，跳过）。"""
    model, results = model_results
    for seg_name, out in results.items():
        scores = _scores(out)
        wide = _wide(out)
        assert set(scores["metric"]) == {"ts", "pod", "far", "miss_rate", "frequency_bias"}

        def value(metric, thr, lead):
            # 口径 B 的宽表一行一个 lead，长表也得按 (指标, 阈值, 时效) 定位
            cell = scores[
                (scores["metric"] == metric)
                & (scores["threshold"] == thr)
                & scores["region"].isna()
                & np.isclose(scores["lead_h"], lead)
            ]
            assert len(cell) == 1, (
                f"{seg_name} {metric} @ ≥{thr} lead {lead} 有 {len(cell)} 行"
            )
            return float(cell["value"].iloc[0])

        for _, row in wide.iterrows():
            thr = float(row["threshold_mm"])
            lead = float(row["lead_h"])
            hits, fls, miss = int(row["hits"]), int(row["false_alarms"]), int(row["misses"])
            assert hits + fls + miss <= int(row["n_pairs"])
            # 派生量只在分母非 0（有定义）时验证；无事件阈值分母是 0，指标本就是 NaN。
            if hits + fls + miss > 0:
                assert value("ts", thr, lead) == pytest.approx(
                    hits / (hits + fls + miss), abs=1e-9
                )
            if hits + miss > 0:
                assert value("pod", thr, lead) == pytest.approx(hits / (hits + miss), abs=1e-9)
                assert value("miss_rate", thr, lead) == pytest.approx(
                    miss / (hits + miss), abs=1e-9
                )
                assert value("frequency_bias", thr, lead) == pytest.approx(
                    (hits + fls) / (hits + miss), abs=1e-9
                )
            if hits + fls > 0:
                assert value("far", thr, lead) == pytest.approx(fls / (hits + fls), abs=1e-9)
