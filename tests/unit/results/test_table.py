"""统一长表的展开规则与落盘口径。"""

import pandas as pd
import pytest
import json

from xmetai_evaluation.core.contracts import MetricResult, ResultStatus
from xmetai_evaluation.core.errors import OutputError
from xmetai_evaluation.results import (
    COVERAGE_COLUMNS,
    DETAIL_COLUMNS,
    SCORE_COLUMNS,
    build_tables,
    categorical_wide,
)
from xmetai_evaluation.results.store import ResultStore, RunContext


def _ts_result() -> MetricResult:
    """构造一个分类检验结果（value 为 阈值 -> 字段 的嵌套字典）。"""
    value = {
        "≥0.1": {
            "threshold": 0.1,
            "threshold_name": "≥0.1",
            "hits": 10,
            "misses": 2,
            "false_alarms": 3,
            "correct_negatives": 100,
            "n_pairs": 115,
            "TS": 0.6667,
            "POD": 0.8333,
            "FAR": 0.2308,
            "miss_rate": 0.1667,
            "BIAS": 1.0833,
        },
        "≥10": {
            "threshold": 10.0,
            "threshold_name": "≥10",
            "hits": 4,
            "misses": 1,
            "false_alarms": 5,
            "correct_negatives": 105,
            "n_pairs": 115,
            "TS": 0.4,
            "POD": 0.8,
            "FAR": 0.5556,
            "miss_rate": 0.2,
            "BIAS": 1.8,
        },
    }
    return MetricResult(
        metric_name="ts",
        metric_version="1.0.0",
        value=value,
        status=ResultStatus.SUCCESS,
        n_requested=115,
        n_valid=115,
        aggregation="contingency_table",
        unit="1",
        product_kind="categorical",
        coordinates={
            "window_h": 24,
            "lead_h": 24,
            "variable": "tp",
            "sample_unit": "station",
        },
    )


def test_long_table_columns_are_canonical():
    tables = build_tables([_ts_result()], run_id="r1", model_id="fuxi", dataset_id="obs")
    assert list(tables.score_table.columns) == SCORE_COLUMNS
    assert set(SCORE_COLUMNS).issubset(set(tables.scores.columns))


def test_grouped_value_expands_to_one_row_per_score():
    tables = build_tables([_ts_result()], run_id="r1")
    scores = tables.scores

    assert {"ts", "pod", "far", "miss_rate", "frequency_bias"}.issubset(set(scores["metric"]))
    # 两个阈值 -> 每个评分两行
    for metric in ("ts", "pod", "far", "miss_rate", "frequency_bias"):
        assert len(scores[scores["metric"] == metric]) == 2


def test_counts_are_details_not_scores():
    tables = build_tables([_ts_result()], run_id="r1")
    assert {"hits", "misses", "false_alarms", "n_pairs", "threshold"}.issubset(
        set(tables.details["field"])
    )
    assert "hits" not in set(tables.scores["metric"])


def test_threshold_and_valid_count_come_from_the_group():
    tables = build_tables([_ts_result()], run_id="r1")
    row = tables.scores[(tables.scores["metric"] == "ts") & (tables.scores["threshold"] == 10.0)]
    assert len(row) == 1
    assert row.iloc[0]["n_valid"] == 115
    assert row.iloc[0]["value"] == pytest.approx(0.4)
    assert row.iloc[0]["window_h"] == 24
    assert row.iloc[0]["sample_unit"] == "station"


def test_undefined_value_keeps_status_instead_of_nan():
    result = MetricResult(
        metric_name="rmse",
        metric_version="1.0.0",
        value=float("nan"),
        status=ResultStatus.NO_VALID_DATA,
        n_requested=0,
        n_valid=0,
    )
    tables = build_tables([result], run_id="r1")
    row = tables.scores.iloc[0]
    assert row["status"] == "no_valid_data"
    assert row["value"] == ""


def test_categorical_wide_is_a_view_of_the_long_table():
    tables = build_tables([_ts_result()], run_id="r1")
    wide = categorical_wide(tables)

    assert {"grade", "threshold_mm", "hits", "TS", "POD", "FAR", "BIAS"}.issubset(
        set(wide.columns)
    )
    first = wide[wide["grade"] == "≥0.1"].iloc[0]
    assert first["threshold_mm"] == pytest.approx(0.1)
    assert first["hits"] == 10
    assert first["TS"] == pytest.approx(0.6667)
    assert len(wide) == 2


def test_store_writes_standard_artifacts(tmp_path):
    store = ResultStore(
        tmp_path / "run",
        RunContext(run_id="run", model_id="fuxi", dataset_id="diamond", protocol_id="p1"),
    )
    artifacts = store.write([_ts_result()], resolved_config={"schema_version": 1})

    scores = pd.read_csv(artifacts["scores"])
    assert list(scores.columns) == SCORE_COLUMNS
    # 核心产物只有两件：长表 + 运行记录（配置快照写在 manifest 里）
    assert set(artifacts) == {"scores", "manifest"}
    assert (artifacts["scores"].parent / "manifest.json").exists()
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["resolved_config"] == {"schema_version": 1}
    # 派生视图默认不写
    assert not (artifacts["scores"].parent / "coverage.csv").exists()
    assert not (artifacts["scores"].parent / "diagnostics").exists()


def test_optional_views_are_written_only_when_requested(tmp_path):
    store = ResultStore(tmp_path / "run", RunContext(run_id="run"))
    artifacts = store.write([_ts_result()], writers=("csv_long", "coverage", "details"))

    coverage = pd.read_csv(artifacts["coverage"])
    assert list(coverage.columns) == COVERAGE_COLUMNS
    assert len(coverage) == 1
    details = pd.read_csv(artifacts["details"])
    assert list(details.columns) == DETAIL_COLUMNS
    assert {"hits", "misses"}.issubset(set(details["field"]))


def test_store_refuses_empty_results(tmp_path):
    store = ResultStore(tmp_path / "empty", RunContext(run_id="empty"))
    with pytest.raises(OutputError):
        store.write([])


def test_categorical_wide_writer_is_wired_through_the_registry(tmp_path):
    store = ResultStore(tmp_path / "run", RunContext(run_id="r1"))
    artifacts = store.write([_ts_result()], writers=("csv_long", "categorical_wide"))

    wide = pd.read_csv(artifacts["categorical_wide"])
    assert {"grade", "threshold_mm", "hits", "TS", "POD", "FAR", "BIAS"}.issubset(
        set(wide.columns)
    )
    assert len(wide) == 2
