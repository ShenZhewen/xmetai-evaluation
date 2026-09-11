"""连续场评测（weather_field_scores）的出图、报告与能力路由。

不依赖任何真实产物：长表与 manifest 都是手搓的最小样本。
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib", reason="可视化测试需要可选依赖，pip install -e .[viz]")

import matplotlib.pyplot as plt

from xmetai_evaluation.visualization.field_plots import (
    FieldScorePlotter,
    load_spectrum_details,
    prepare_field_dataframe,
)
from xmetai_evaluation.visualization.field_report import (
    build_field_report,
    diagnose,
    manifest_info_from,
    requested_from,
    summarize,
)
from xmetai_evaluation.visualization.report import build_report, detect_capability

LEADS = (6.0, 12.0, 18.0)


def _row(variable, metric, lead, value, unit, status="success"):
    return {
        "run_id": "demo_run",
        "model_id": "fuxi",
        "dataset_id": "era5_zarr",
        "variable": variable,
        "metric": metric,
        "lead_h": lead,
        "value": value,
        "unit": unit,
        "status": status,
        "n_valid": 1038240,
    }


def _scores_rows():
    """两个变量、三个时效；指标按变量路由，不是满交叉（与真实产物同构）。"""
    rows = []
    # z500：RMSE 10 → 40（涨 4 倍），ACC 0.9 → 0.3，活跃度比 0.8 → 0.9（偏平滑）
    for lead, rmse, acc, ratio in zip(LEADS, (10.0, 20.0, 40.0), (0.9, 0.6, 0.3), (0.8, 0.85, 0.9)):
        rows.append(_row("z500", "rmse", lead, rmse, "m^2/s^2"))
        rows.append(_row("z500", "acc", lead, acc, "1"))
        rows.append(_row("z500", "activity_ratio", lead, ratio, "1"))
        rows.append(_row("z500", "activity_bias", lead, ratio - 1.0, "1"))
        rows.append(_row("z500", "activity_forecast", lead, 90.0 * ratio, "1"))
        rows.append(_row("z500", "activity_observation", lead, 90.0, "1"))
        rows.append(_row("z500", "spectrum_power_ratio", lead, 1.2, "1"))
    # t2m：只有 RMSE 与谱比；RMSE 1 → 3（涨 3 倍），谱比 1.5（比 z500 更偏）
    for lead, rmse in zip(LEADS, (1.0, 2.0, 3.0)):
        rows.append(_row("t2m", "rmse", lead, rmse, "K"))
        rows.append(_row("t2m", "spectrum_power_ratio", lead, 1.5, "1"))
    # 一行失败结果：不能进统计，但必须被数出来
    rows.append(_row("t2m", "rmse", 24.0, None, "K", status="failed"))
    return pd.DataFrame(rows)


@pytest.fixture()
def scores():
    return _scores_rows()


def _manifest(pipeline="weather_field_scores", **overrides):
    manifest = {
        "run_id": "demo_run",
        "protocol_id": "grid_valid_time",
        "metrics": ["acc", "activity", "rmse", "zonal_spectrum"],
        "forecast_source": "fuxi",
        "observation_source": "era5_zarr",
        "forecast_input_files": ["/data/fc/20250102/001.nc"],
        "observation_input_files": ["/data/obs/era5_pl.zarr"],
        "created_at": "2026-01-01T00:00:00",
        "resolved_config": {
            "name": "demo_run",
            "pipeline": pipeline,
            "start_date": "20250101",
            "end_date": "20250102",
            "forecast": {"reader": "fuxi_phys", "params": {}},
            "observation": {"reader": "era5_zarr", "params": {}},
            "reference": {"reader": "daily_climatology", "params": {}},
            "metric_options": {"rmse": {"variables": ["z500", "t2m", "q700"]}},
        },
    }
    manifest.update(overrides)
    return manifest


def _details_frame():
    """最小逐波数谱：k=0..3，外加两行 summary（不该被画进曲线）。"""
    rows = []
    for k in (0, 1, 2, 3):
        rows.append({"variable": "z500", "lead_h": 6.0, "group": f"k={k}", "field": "wavenumber", "value": float(k)})
        rows.append({"variable": "z500", "lead_h": 6.0, "group": f"k={k}", "field": "power_forecast", "value": 100.0 / (k + 1)})
        rows.append({"variable": "z500", "lead_h": 6.0, "group": f"k={k}", "field": "power_observation", "value": 80.0 / (k + 1)})
    rows.append({"variable": "z500", "lead_h": 6.0, "group": "summary", "field": "total_power_forecast", "value": 108.33})
    rows.append({"variable": "z500", "lead_h": 6.0, "group": "summary", "field": "total_power_observation", "value": 86.67})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# prepare_field_dataframe
# --------------------------------------------------------------------------- #
def test_prepare_requires_columns(scores):
    with pytest.raises(ValueError) as error:
        prepare_field_dataframe(scores.drop(columns=["value"]))
    assert "value" in str(error.value)
    assert "现有列" in str(error.value)


def test_prepare_rejects_empty():
    with pytest.raises(ValueError):
        prepare_field_dataframe(pd.DataFrame(columns=["variable", "metric", "lead_h", "value"]))


def test_prepare_fills_missing_optional_columns(scores):
    data = prepare_field_dataframe(scores.drop(columns=["unit", "status"]))
    assert set(data["unit"]) == {""}
    assert set(data["status"]) == {"success"}


def test_prepare_sorts_and_keeps_failed_rows(scores):
    data = prepare_field_dataframe(scores)
    # 排序键是 (metric, variable, lead_h)：lead_h 在每个「指标 × 变量」组内递增，
    # 不是全局递增——同一时效上有很多变量，全局排会把分组打散。
    keys = list(data[["metric", "variable"]].itertuples(index=False, name=None))
    assert keys == sorted(keys)
    for _, chunk in data.groupby(["metric", "variable"], observed=True):
        assert chunk["lead_h"].is_monotonic_increasing
    # 失败行还在（报告要数它），但 summary 不能把它算进曲线
    assert (data["status"] == "failed").sum() == 1
    assert summarize(data)["n_failed"] == 1


# --------------------------------------------------------------------------- #
# summarize / diagnose
# --------------------------------------------------------------------------- #
def test_summarize_curve_stats(scores):
    summary = summarize(scores)
    z500 = summary["curves"]["rmse"]["z500"]
    assert z500["first"] == 10.0
    assert z500["last"] == 40.0
    assert z500["ratio"] == 4.0
    assert z500["n_leads"] == 3
    assert z500["unit"] == "m^2/s^2"
    assert z500["n_rise"] == 2
    assert z500["n_fall"] == 0
    # 逐变量路由：acc 只有 z500 有，t2m 是缺席而不是"值为 0"
    assert list(summary["curves"]["acc"]) == ["z500"]
    assert "t2m" not in summary["curves"]["activity_ratio"]
    assert list(summary["curves"]["spectrum_power_ratio"]) == ["t2m", "z500"]


def test_summarize_reports_requested_gap(scores):
    summary = summarize(scores, requested={"rmse": ["z500", "t2m", "q700"]})
    assert summary["missing"] == [{"metric": "rmse", "variable": "q700"}]


def test_requested_from_manifest(scores):
    info = manifest_info_from(_manifest())
    assert requested_from(info) == {"rmse": ["z500", "t2m", "q700"]}
    assert manifest_info_from(None) == {}


def test_diagnose_every_finding_carries_a_number(scores):
    findings = diagnose(summarize(scores, requested={"rmse": ["z500", "t2m", "q700"]}))
    assert findings
    for finding in findings:
        assert re.search(r"\d", finding), f"结论里没有数值: {finding}"


def test_diagnose_points_at_the_gap_and_the_worst(scores):
    findings = " ".join(diagnose(summarize(scores, requested={"rmse": ["z500", "t2m", "q700"]})))
    assert "q700" in findings  # 该算但没出数
    assert "z500" in findings  # RMSE 涨得最快（4 倍 vs 3 倍）
    assert "t2m" in findings  # 谱比偏离 1 最多（1.5 vs 1.2）


def test_diagnose_without_any_number():
    empty = pd.DataFrame(columns=["variable", "metric", "lead_h", "value"])
    assert diagnose(summarize(empty)) == ["长表里没有任何可用的「变量 × 时效」数值，无法诊断。"]


# --------------------------------------------------------------------------- #
# detect_capability
# --------------------------------------------------------------------------- #
def _write_manifest(tmp_path, manifest):
    directory = Path(tmp_path) / "run"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def test_detect_uses_resolved_config_pipeline(tmp_path):
    # 目录名与流程名无关：认的是 manifest，不是目录名
    directory = _write_manifest(tmp_path, _manifest(pipeline="weather_field_scores"))
    assert detect_capability(directory) == "weather_field_scores"


def test_detect_falls_back_to_protocol_and_metrics(tmp_path):
    manifest = _manifest()
    manifest.pop("resolved_config")
    directory = _write_manifest(tmp_path, manifest)
    assert detect_capability(directory) == "weather_field_scores"


def test_detect_rejects_not_implemented_pipeline(tmp_path):
    directory = _write_manifest(tmp_path, _manifest(pipeline="weather_ens_field_scores"))
    with pytest.raises(ValueError) as error:
        detect_capability(directory)
    assert "还没做" in str(error.value)


def test_detect_rejects_unknown_manifest(tmp_path):
    # 既没有 resolved_config，协议 + 指标也凑不出已知判据。
    # （注：`station_valid_time` + `["ts"]` 是**认得出**的 TS 退路判据，别拿它当反例。）
    manifest = _manifest()
    manifest.pop("resolved_config")
    manifest["protocol_id"] = "station_valid_time"
    manifest["metrics"] = ["crps"]
    directory = _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError):
        detect_capability(directory)


def test_detect_rejects_missing_manifest(tmp_path):
    (Path(tmp_path) / "run").mkdir()
    with pytest.raises(ValueError) as error:
        detect_capability(Path(tmp_path) / "run")
    assert "manifest.json" in str(error.value)


# --------------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------------- #
def test_vs_lead_splits_panels_by_unit(scores):
    plotter = FieldScorePlotter()
    # RMSE 有 m^2/s^2 与 K 两种单位 → 两个子图；混在一根纵轴上比大小是错的
    figure = plotter.plot_metric_vs_lead(scores, "rmse")
    assert len(figure.axes) == 2
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert any("m^2/s^2" in label for label in labels)
    assert any("[K]" in label for label in labels)
    plt.close(figure)

    # 比值类只有一个单位（无量纲），画一根轴，并且要有 y=1 参考线
    figure = plotter.plot_metric_vs_lead(scores, "activity_ratio")
    assert len(figure.axes) == 1
    assert "[1]" not in figure.axes[0].get_ylabel()
    assert any(line.get_ydata()[0] == 1.0 for line in figure.axes[0].lines if len(line.get_ydata()))
    plt.close(figure)


def test_vs_lead_rejects_unknown_metric(scores):
    with pytest.raises(ValueError):
        FieldScorePlotter().plot_metric_vs_lead(scores, "no_such_metric")


def test_heatmap_is_unit_free(scores):
    figure = FieldScorePlotter().plot_metric_heatmap(scores, "rmse")
    # 填的是"相对首时效的倍数"，所以首列恒为 1，跨单位的行才可比
    mesh = figure.axes[0].collections[0]
    values = np.ma.filled(np.ma.asarray(mesh.get_array()), np.nan).reshape(2, -1)
    assert values[:, 0] == pytest.approx([1.0, 1.0])
    assert values[:, -1] == pytest.approx([4.0, 3.0])  # 按末/首倍数降序：z500 4 倍在前
    plt.close(figure)


def test_heatmap_needs_more_than_one_lead(scores):
    single = scores[scores["lead_h"] == 6.0]
    with pytest.raises(ValueError):
        FieldScorePlotter().plot_metric_heatmap(single, "rmse")


def test_spectrum_curve_drops_k0_and_summary(scores):
    details = _details_frame()
    figure = FieldScorePlotter().plot_spectrum_curve(details, "z500", 6.0)
    assert len(figure.axes) == 1
    lines = [line for line in figure.axes[0].lines if len(line.get_xdata())]
    assert len(lines) == 2  # 预报、实况
    assert list(lines[0].get_xdata()) == [1.0, 2.0, 3.0]  # k=0 功率为 0，log 轴画不了
    assert "总功率比" in figure.axes[0].get_title()
    plt.close(figure)


def test_load_spectrum_details_filters_and_raises(tmp_path):
    path = tmp_path / "scores_detail.csv"
    _details_frame().to_csv(path, index=False)
    frame = load_spectrum_details(path, variable="z500", lead_h=6.0)
    assert set(frame["group"]) == {"k=0", "k=1", "k=2", "k=3"}  # summary 行被滤掉
    with pytest.raises(ValueError):
        load_spectrum_details(path, variable="q700", lead_h=6.0)


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def test_report_sections_and_numbers(tmp_path, scores):
    info = manifest_info_from(_manifest())
    path = build_field_report(
        scores,
        tmp_path / "REPORT.md",
        model_name="FuXi",
        requested=requested_from(info),
        manifest_info=info,
    )
    text = path.read_text(encoding="utf-8")
    for heading in (
        "## 一、评测对象与数据来源",
        "## 二、执行摘要",
        "## 三、逐变量总览",
        "## 四、RMSE 随时效",
        "## 五、活跃度与纬向谱",
        "## 六、口径与注意事项",
    ):
        assert heading in text
    assert "weather_field_scores" in text  # 能力来自 manifest，不是猜的
    assert "m^2/s^2" in text  # 单位随数据源走，报告里要说清
    assert "q700" in text  # 该算但没出数的组合要被点出来
    assert "n_valid" in text  # 谱行的 n_valid 不是样本量，口径里要提醒


def test_report_without_manifest_is_honest(tmp_path, scores):
    path = build_field_report(scores, tmp_path / "REPORT.md", model_name="FuXi")
    text = path.read_text(encoding="utf-8")
    assert "未提供" in text
    assert "本次未出图" in text or "## 六、口径与注意事项" in text


def test_report_indexes_figures(tmp_path, scores):
    figure_dir = tmp_path / "figs"
    figure_dir.mkdir()
    png = figure_dir / "rmse_vs_lead.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    path = build_field_report(
        scores, tmp_path / "REPORT.md", artifacts={"rmse_vs_lead": png}
    )
    text = path.read_text(encoding="utf-8")
    assert "## 六、图表" in text
    assert "![rmse_vs_lead](rmse_vs_lead.png)" in text
    assert "## 七、口径与注意事项" in text


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def test_create_report_writes_artifacts(tmp_path, scores):
    details_path = tmp_path / "scores_detail.csv"
    _details_frame().to_csv(details_path, index=False)

    artifacts = FieldScorePlotter().create_report(
        scores,
        tmp_path / "report",
        details_path=details_path,
        spectrum_variable="z500",
        model_name="FuXi",
        manifest_info=manifest_info_from(_manifest()),
    )
    assert set(artifacts) == {
        "rmse_vs_lead",
        "acc_vs_lead",
        "activity_ratio_vs_lead",
        "spectrum_power_ratio_vs_lead",
        "rmse_heatmap",
        "spectrum_curve_z500_6h",
        "report",
    }
    for name, path in artifacts.items():
        assert Path(path).is_file(), name
        assert Path(path).stat().st_size > 0, name
    assert "谱曲线" in (tmp_path / "report" / "REPORT.md").read_text(encoding="utf-8")


def test_build_report_end_to_end(tmp_path, scores):
    source = tmp_path / "product"
    source.mkdir()
    scores.to_csv(source / "scores.csv", index=False)
    (source / "manifest.json").write_text(
        json.dumps(_manifest(), ensure_ascii=False), encoding="utf-8"
    )
    artifacts = build_report(source, out_dir=tmp_path / "out")
    assert artifacts["report"] == tmp_path / "out" / "REPORT.md"
    assert artifacts["report"].is_file()
    assert (tmp_path / "out" / "rmse_vs_lead.png").is_file()


def test_build_report_refuses_unimplemented(tmp_path, scores):
    source = tmp_path / "product"
    source.mkdir()
    scores.to_csv(source / "scores.csv", index=False)
    (source / "manifest.json").write_text(
        json.dumps(_manifest(pipeline="weather_ens_field_scores"), ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as error:
        build_report(source, out_dir=tmp_path / "out")
    assert "还没做" in str(error.value)
    assert not (tmp_path / "out").exists()  # 不静默回落到别的模板
