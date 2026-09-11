"""降水分类检验（TS 系列）的路由、派发，以及"报告格式固定"这条承诺。

不依赖任何真实产物：宽表与 manifest 都是手搓的最小样本。

最后几组用例是**格式契约**：实跑渲染器，拿真报告去对
``skills/xmetai-evaluation/assets/templates/ts.md`` 里的填空骨架——
章节序列逐项比对，骨架里**所有不含占位符的行**必须逐字出现且顺序一致。
渲染器改了结构而骨架没同步、或者骨架写错了，测试都会红。
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib", reason="可视化测试需要可选依赖，pip install -e .[viz]")

from matplotlib.cbook import boxplot_stats as mpl_boxplot_stats

from xmetai_evaluation.visualization.report import (
    build_report,
    detect_capability,
    parse_compare_specs,
)
from xmetai_evaluation.visualization.ts_report import box_stats, build_ts_report, summarize

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = REPO_ROOT / "skills" / "xmetai-evaluation" / "assets" / "templates" / "ts.md"

#: 5 个时效——两个点做箱线图是退化的（Q1/Q3 就是那两个点，栅栏里一个点都没有）
LEADS = (24.0, 48.0, 72.0, 96.0, 120.0)


def _wide_rows(model="demo_ts", grades=None):
    """4 个降水等级 × 5 个时效；末档命中为 0（复刻真实产物里 ≥250mm 的样子）。"""
    rows = []
    for grade, threshold, base in grades or (
        ("≥0.1", 0.1, 0.50),
        ("≥10", 10.0, 0.30),
        ("≥25", 25.0, 0.10),
        ("≥250", 250.0, 0.00),
    ):
        for index, lead in enumerate(LEADS):
            hits = int(base * 100 * (1.0 - 0.2 * index))
            rows.append(
                {
                    "run_id": model,
                    "window_h": 24,
                    "lead_h": lead,
                    "grade": grade,
                    "threshold_mm": threshold,
                    "hits": hits,
                    "misses": 20,
                    "false_alarms": 10,
                    "n_pairs": 5000,
                    "TS": base * (1.0 - 0.2 * index),
                    "POD": 0.5,
                    "FAR": 0.2 if hits else float("nan"),
                    "漏报率": 0.5,
                    "BIAS": 1.1 if hits else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _manifest(pipeline="weather_ts_det", **overrides):
    manifest = {
        "run_id": "demo_ts",
        "protocol_id": "station_valid_time",
        "metrics": ["ts"],
        # reader 类型，不是模型名——报告标题**不该**用它
        "forecast_source": "fuxi",
        "observation_source": "diamond_station",
        "created_at": "2026-01-01T00:00:00",
        "resolved_config": {
            "name": "demo_ts",
            "pipeline": pipeline,
            "start_date": "20250101",
            "end_date": "20250102",
            "forecast": {"reader": "fuxi", "params": {}},
            "observation": {"reader": "station", "params": {}},
            "metric_options": {"ts_score": {"thresholds": [0.1, 10.0, 25.0, 250.0]}},
            "writers": ["csv_long", "categorical_wide"],
        },
    }
    manifest.update(overrides)
    return manifest


def _write_product(tmp_path, *, with_wide=True, pipeline="weather_ts_det", manifest=None):
    product = Path(tmp_path) / "product"
    (product / "diagnostics").mkdir(parents=True, exist_ok=True)
    (product / "manifest.json").write_text(
        json.dumps(manifest or _manifest(pipeline), ensure_ascii=False), encoding="utf-8"
    )
    if with_wide:
        _wide_rows().to_csv(product / "diagnostics" / "categorical_wide.csv", index=False)
    return product


# --------------------------------------------------------------------------- #
# 认能力
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pipeline", ["weather_ts_det", "weather_ts_ens", "fdp_precip_ts"])
def test_detect_all_ts_pipelines(tmp_path, pipeline):
    # 三条 TS 流程共用一套模板，认出来是哪条都行，关键是都**认得出**
    assert detect_capability(_write_product(tmp_path, pipeline=pipeline)) == pipeline


def test_detect_ts_falls_back_without_resolved_config(tmp_path):
    manifest = _manifest()
    manifest.pop("resolved_config")
    product = _write_product(tmp_path, manifest=manifest)
    assert detect_capability(product) == "weather_ts_det"


# --------------------------------------------------------------------------- #
# 派发
# --------------------------------------------------------------------------- #
def test_build_report_ts_end_to_end(tmp_path):
    product = _write_product(tmp_path)
    artifacts = build_report(product, out_dir=tmp_path / "out")

    assert artifacts["report"] == tmp_path / "out" / "REPORT.md"
    assert artifacts["report"].is_file()
    assert (tmp_path / "out" / "24h_TS_vs_lead.png").is_file()
    assert (tmp_path / "out" / "24h_TS_heatmap.png").is_file()
    assert (tmp_path / "out" / "24h_TS_boxplot.png").is_file()
    # 走完整入口产出的报告，也必须合骨架
    _assert_matches_skeleton(artifacts["report"], "A")


def test_title_uses_run_id_not_forecast_source(tmp_path):
    product = _write_product(tmp_path)
    build_report(product, out_dir=tmp_path / "out")
    title = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8").splitlines()[0]
    # forecast_source 是 reader 类型（"fuxi"），拿它当标题就是张冠李戴
    assert title == "# demo_ts 降水分类检验报告"
    assert "fuxi" not in title


def test_explicit_model_name_wins(tmp_path):
    product = _write_product(tmp_path)
    build_report(product, out_dir=tmp_path / "out", model_name="FGVP-v2")
    title = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8").splitlines()[0]
    assert title == "# FGVP-v2 降水分类检验报告"


def test_missing_wide_table_is_refused(tmp_path):
    product = _write_product(tmp_path, with_wide=False)
    with pytest.raises(ValueError) as error:
        build_report(product, out_dir=tmp_path / "out")
    assert "categorical_wide.csv" in str(error.value)
    assert "csv_long" in str(error.value)  # 告诉用户怎么把宽表产出来
    assert not (tmp_path / "out").exists()  # 不静默产出半个空目录


def test_source_is_recorded_in_report_head(tmp_path):
    product = _write_product(tmp_path)
    build_report(product, out_dir=tmp_path / "out")
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "categorical_wide.csv" in text  # 来源可溯源


# --------------------------------------------------------------------------- #
# 格式契约：真报告必须填得进骨架
# --------------------------------------------------------------------------- #
def _skeleton_lines(branch):
    """抽出骨架文件里某个分支的完整骨架——测试的真值来源。"""
    text = TEMPLATE.read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- skeleton: {branch} -->\n````markdown\n(.*?)````", text, re.S
    )
    assert match, f"{TEMPLATE.name} 里没找到分支 {branch} 的骨架"
    return match.group(1).strip("\n").splitlines()


def _headings_of(lines):
    return [line for line in lines if re.match(r"^#{1,6} ", line)]


def _literal_lines(lines):
    """不含占位符的行——这些必须逐字出现在报告里，一个标点都不能差。"""
    return [line for line in lines if line.strip() and "{" not in line]


def _as_regex(declared):
    """骨架里的 `{…}` 当通配符，其余逐字匹配。"""
    parts = re.split(r"\{[^}]*\}", declared)
    return re.compile("^" + ".+?".join(re.escape(part) for part in parts) + "$")


def _assert_matches_skeleton(report_path, branch):
    skeleton = _skeleton_lines(branch)
    report = Path(report_path).read_text(encoding="utf-8").splitlines()

    # 1) 章节序列一致（标题里的 {…} 当通配符，比如 `三、TS 随时效变化（{首}–{末}h）`）
    declared = _headings_of(skeleton)
    actual = _headings_of(report)
    assert len(actual) == len(declared), (
        f"章节数对不上：报告 {len(actual)} 节，骨架分支 {branch} 有 {len(declared)} 节\n"
        f"报告：{actual}\n骨架：{declared}"
    )
    for got, want in zip(actual, declared):
        assert _as_regex(want).match(got), f"章节对不上：报告 {got!r}，骨架 {want!r}"

    # 2) 骨架里所有不含占位符的行（表头、分隔行、口径说明、固定句子…），
    #    必须在报告里逐字出现，且顺序一致
    cursor = 0
    for line in _literal_lines(skeleton):
        while cursor < len(report) and report[cursor] != line:
            cursor += 1
        assert cursor < len(report), (
            f"骨架分支 {branch} 的这行在报告里找不到（或顺序不符）：{line!r}"
        )
        cursor += 1


def test_skeletons_are_not_empty():
    # 骨架自己先得站得住：三个分支都要有，且得有实质内容
    for branch in ("A", "B", "C"):
        assert len(_skeleton_lines(branch)) >= 60, branch
        assert len(_literal_lines(_skeleton_lines(branch))) >= 24, branch


def test_headings_match_skeleton_branch_a(tmp_path):
    """单模型：不出现对比节，跨时效分布是五，建议/口径 是 六/七。"""
    artifact = tmp_path / "24h_TS_vs_lead.png"
    artifact.write_bytes(b"\x89PNG\r\n\x1a\n")
    path = build_ts_report(
        _wide_rows(),
        tmp_path / "REPORT.md",
        model_name="demo",
        artifacts={"TS_vs_lead": artifact},
    )
    _assert_matches_skeleton(path, "A")


def test_headings_match_skeleton_branch_b(tmp_path):
    """单基准：二～五节不变，对比表挪到六，后面整体右移一位。"""
    data = _wide_rows()
    path = build_ts_report(
        data,
        tmp_path / "REPORT.md",
        model_name="demo",
        baseline_df=data.copy(),
        baseline_name="FuXi",
        artifacts={"TS_vs_lead": tmp_path / "x.png"},
    )
    _assert_matches_skeleton(path, "B")
    assert "## 六、与基准 FuXi 的对比（24h）" in Path(path).read_text(encoding="utf-8")


def test_headings_match_skeleton_branch_c(tmp_path):
    """多模型对比：编号与单基准一致，只是第六节换成横表（并多一张差值图）。"""
    data = _wide_rows()
    path = build_ts_report(
        data,
        tmp_path / "REPORT.md",
        model_name="demo",
        baselines={"FuXi": data.copy(), "AIFS": data.copy()},
        artifacts={"TS_vs_lead": tmp_path / "x.png"},
    )
    _assert_matches_skeleton(path, "C")
    assert "## 六、与对比模型的比较（24h，TS）" in Path(path).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 论文式章节的第二条契约：图注/表题的数量与编号
#
# 图注和表题整行都含 `{…}`，上面的逐字扫描抓不到它们——一条都写不出来、
# 或者编号断成 1、2、5，逐字扫描照样全绿。所以这几条单独钉。
# --------------------------------------------------------------------------- #
def _captions(text, label):
    """报告里 `图 N：…` / `表 N：…` 的编号，按出现顺序。"""
    return [int(match.group(1)) for match in re.finditer(rf"^{label} (\d+)：", text, re.M)]


def _assert_sequential(text, label):
    numbers = _captions(text, label)
    assert numbers, f"报告里一条「{label} N：」都没有"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"{label} 的编号不是 1..N（跳号或重号）：{numbers}"
    )


def test_captions_are_numbered_sequentially(tmp_path):
    product = _write_product(tmp_path)
    fuxi = _write_compare(tmp_path, "fuxi")
    build_report(product, out_dir=tmp_path / "out", model_name="FGVP", compare=[f"FuXi={fuxi}"])
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    _assert_sequential(text, "图")
    _assert_sequential(text, "表")
    # 单模型分支也要连续（图的条数不同，规则相同）
    single = build_report(product, out_dir=tmp_path / "out_single").get("report")
    _assert_sequential(single.read_text(encoding="utf-8"), "图")


def test_missing_figures_do_not_leave_number_gaps(tmp_path):
    """只给一张图的产物，图注照样从 1 开始连号——缺的图连图注一起不出现。"""
    artifact = tmp_path / "24h_TS_vs_lead.png"
    artifact.write_bytes(b"\x89PNG\r\n\x1a\n")
    path = build_ts_report(
        _wide_rows(), tmp_path / "REPORT.md", model_name="demo",
        artifacts={"TS_vs_lead": artifact},
    )
    text = Path(path).read_text(encoding="utf-8")
    _assert_sequential(text, "图")
    assert "![TS_vs_lead]" in text
    assert "TS_heatmap" not in text  # 没给的图不许凭空出现
    assert "图 2" not in text  # 缺图也不许跳号


def _section_bodies(text):
    """{章节标题: 该节正文}——按 `## ` 切段。"""
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return {part.splitlines()[0]: part for part in parts}


#: 「论文实验」那五节——每节都得有自己的小结，小结里得有数字
ANALYSIS_HEADINGS = (
    "二、逐降水等级表现",
    "三、TS 随时效变化",
    "四、误差形态与偏差结构",
    "五、跨时效分布",
)


def test_every_analysis_section_has_a_summary_with_numbers(tmp_path):
    product = _write_product(tmp_path)
    fuxi = _write_compare(tmp_path, "fuxi")
    build_report(product, out_dir=tmp_path / "out", model_name="FGVP", compare=[f"FuXi={fuxi}"])
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    body = _section_bodies(text)

    headings = list(ANALYSIS_HEADINGS) + ["六、与对比模型的比较"]
    for wanted in headings:
        matched = next((key for key in body if key.startswith(wanted)), None)
        assert matched, f"报告里没有「{wanted}」这一节：{list(body)}"
        assert "**小结**：" in body[matched], f"「{matched}」这一节没有小结"
        summary = body[matched].split("**小结**：", 1)[1].strip()
        assert re.search(r"\d", summary), (
            f"「{matched}」的小结里一个数字都没有，退化成套话了：{summary[:60]}"
        )


def test_every_artifact_is_placed_in_the_report(tmp_path):
    """产物里出的每一张图都要在正文里带着图注出现——一张不多、一张不少。

    这条防的是「加了新图但没人安排章节」：图安安静静地躺在输出目录里，
    报告里连提都没提，谁也不会发现。
    """
    product = _write_product(tmp_path)
    fuxi = _write_compare(tmp_path, "fuxi")
    artifacts = build_report(
        product, out_dir=tmp_path / "out", model_name="FGVP", compare=[f"FuXi={fuxi}"]
    )
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")

    placed = {match.group(1) for match in re.finditer(r"^!\[(\w+)\]\((\S+)\)$", text, re.M)}
    images = {
        name for name, path in artifacts.items() if Path(path).suffix.lower() == ".png"
    }
    assert images, "这一轮一个图都没出，用例本身失效了"
    assert images <= placed, f"这些图没进报告：{sorted(images - placed)}"
    assert "图表" not in _section_bodies(text), "末尾不该再有裸图列表式的「图表」索引节"

    # 图行引用的文件名必须真是产物里那个，不能指到一个不存在的文件
    for match in re.finditer(r"^!\[(\w+)\]\((\S+)\)$", text, re.M):
        assert match.group(2) == Path(artifacts[match.group(1)]).name


# --------------------------------------------------------------------------- #
# 箱线图的统计量：神谕就是画图那套代码
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "values",
    [
        [0.50, 0.40, 0.30, 0.20, 0.10],
        [0.0] * 14 + [0.9],  # IQR 为 0，单个离群点
        [0.0] * 15,  # 真实 ≥250mm 的样子：全同值
        [0.2],  # 单点
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    ],
)
def test_box_stats_matches_matplotlib(values):
    """表里的数字必须和图上画出来的箱子逐项一致。

    神谕用 ``matplotlib.cbook.boxplot_stats``——**画图的就是它**，
    所以这条用例一劳永逸地钉死"表格与图不可能各说各话"。
    """
    frame = pd.DataFrame({"lead_h": list(range(len(values))), "TS": values})
    got = box_stats(frame)
    want = mpl_boxplot_stats(values)[0]
    assert got["n"] == len(values)
    for key, oracle in (
        ("q1", "q1"), ("median", "med"), ("q3", "q3"),
        ("whisker_lo", "whislo"), ("whisker_hi", "whishi"),
    ):
        assert got[key] == pytest.approx(want[oracle]), key
    assert sorted(got["outlier_values"]) == pytest.approx(sorted(want["fliers"]))


def test_box_stats_reports_which_lead_is_the_outlier():
    # 离群点要能定位到具体时效，否则表格里只能写"有离群点"而说不出是哪个
    values = [0.0] * 14 + [0.9]
    frame = pd.DataFrame({"lead_h": list(range(24, 24 * 16, 24)), "TS": values})
    stats = box_stats(frame)
    assert stats["n"] == 15
    assert stats["outlier_leads"] == [360.0]
    assert stats["outlier_values"] == pytest.approx([0.9])
    assert stats["iqr"] == pytest.approx(0.0)


def test_box_stats_empty_returns_nan():
    # 全 NaN 时不能去 np.percentile 空数组（会告警 + 出 nan 语义不清）
    frame = pd.DataFrame({"lead_h": [24.0, 48.0], "TS": [float("nan")] * 2})
    stats = box_stats(frame)
    assert stats["n"] == 0
    assert not np.isfinite(stats["iqr"])
    assert stats["outlier_leads"] == [] and stats["outlier_values"] == []


def test_box_stats_rejects_unknown_metric():
    with pytest.raises(KeyError):
        box_stats(pd.DataFrame({"lead_h": [24.0], "TS": [0.5]}), metric="CRPS")


def test_boxplot_survives_a_model_missing_a_grade(tmp_path):
    """某个模型没有某个量级时要照样出图（真实数据里就有：6 档 vs 5 档）。

    齐整的 fixture 走不到这条分支，只有这里真的会跳过一格。
    """
    from xmetai_evaluation.visualization.precipitation_plots import PrecipitationPlotter

    full = _wide_rows()
    fewer = _wide_rows(grades=(("≥0.1", 0.1, 0.50), ("≥10", 10.0, 0.30)))
    path = tmp_path / "box.png"
    PrecipitationPlotter().plot_metric_boxplot(
        {"full": full, "fewer": fewer}, metric="TS", save_path=path
    )
    assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 对比模型：--compare 的解析与端到端
# --------------------------------------------------------------------------- #
def _write_compare(tmp_path, name, grades=None):
    path = tmp_path / f"{name}.csv"
    _wide_rows(name, grades=grades).to_csv(path, index=False)
    return path


def test_parse_compare_specs_ok(tmp_path):
    path = _write_compare(tmp_path, "fuxi")
    assert parse_compare_specs([f"FuXi={path}"]) == {"FuXi": path}


@pytest.mark.parametrize(
    "spec, fragment",
    [
        ("FuXi", "名字=CSV路径"),  # 少了 =
        ("FuXi=", "名字=CSV路径"),  # 少了路径
        ("=a.csv", "名字=CSV路径"),  # 少了名字
    ],
)
def test_parse_compare_specs_rejects_malformed(tmp_path, spec, fragment):
    with pytest.raises(ValueError) as error:
        parse_compare_specs([spec])
    assert fragment in str(error.value)


def test_parse_compare_specs_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError) as error:
        parse_compare_specs([f"FuXi={tmp_path / 'nope.csv'}"])
    assert "nope.csv" in str(error.value)


def test_compare_rejects_name_clashing_with_main_model(tmp_path):
    product = _write_product(tmp_path)
    compare = _write_compare(tmp_path, "fuxi")
    with pytest.raises(ValueError) as error:
        build_report(product, out_dir=tmp_path / "out", compare=[f"demo_ts={compare}"])
    assert "重名" in str(error.value)
    assert not (tmp_path / "out").exists()  # 校验在 mkdir 之前，不留半个空目录


def test_compare_is_refused_for_other_families(tmp_path):
    # 连续场没有"同量级同阈值"的宽表概念，--compare 必须明确报错而不是被静默忽略
    product = tmp_path / "field_product"
    product.mkdir()
    (product / "manifest.json").write_text(
        json.dumps(
            {"run_id": "field_run", "resolved_config": {"pipeline": "weather_field_scores"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    compare = _write_compare(tmp_path, "fuxi")
    with pytest.raises(ValueError) as error:
        build_report(product, out_dir=tmp_path / "out", compare=[f"FuXi={compare}"])
    assert "只有 TS 系列支持" in str(error.value)
    assert not (tmp_path / "out").exists()


def test_build_report_ts_with_compare_end_to_end(tmp_path):
    product = _write_product(tmp_path)
    fuxi = _write_compare(tmp_path, "fuxi")
    aifs = _write_compare(tmp_path, "aifs")

    artifacts = build_report(
        product,
        out_dir=tmp_path / "out",
        model_name="FGVP",
        compare=[f"FuXi={fuxi}", f"AIFS={aifs}"],
    )

    assert (tmp_path / "out" / "24h_TS_boxplot.png").is_file()
    assert (tmp_path / "out" / "24h_multi_model_TS_delta_vs_FGVP.png").is_file()
    report = artifacts["report"]
    _assert_matches_skeleton(report, "C")

    text = report.read_text(encoding="utf-8")
    assert "## 六、与对比模型的比较（24h，TS）" in text
    # 三个模型都写进了来源，不能有"未提供"
    assert "未提供" not in text
    # 箱线表体：3 个模型 × 4 个量级
    body = _box_table_body(text)
    assert len(body) == 12
    assert sum(1 for row in body if row.startswith("| FGVP |")) == 4
    # 图就地展示，报告里只出现一次（不再有末尾的「图表」索引重复一遍）
    assert text.count("![TS_boxplot]") == 1


def test_box_section_present_in_branch_a(tmp_path):
    """单模型也要有这一节：一个量级一个箱体，看的是跨时效离散。"""
    product = _write_product(tmp_path)
    build_report(product, out_dir=tmp_path / "out")
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "## 五、跨时效分布（箱线图，TS）" in text
    body = _box_table_body(text)
    assert len(body) == 4  # 一个量级一行
    assert all(row.startswith("| demo_ts |") for row in body)
    # 离群时效那一列：末档 ≥250 全 0，没有离群点，应当写 —
    assert "| demo_ts | ≥250 |" in text


def _box_table_body(text):
    """报告「跨时效分布」一节箱线表的数据行（不含表头与分隔行）。"""
    start = text.index("## 五、跨时效分布（箱线图，TS）")
    end = text.index("## ", start + 10)
    return [
        line
        for line in text[start:end].splitlines()
        if line.startswith("| ")
        and not line.startswith("| 模型 ")
        and not set(line) <= set("|-")  # 表格分隔行
    ]


# --------------------------------------------------------------------------- #
# 小结函数：数据退化时给说明，不给假结论
# --------------------------------------------------------------------------- #
def test_analysis_functions_survive_degenerate_input():
    from xmetai_evaluation.visualization.ts_report import (
        analysis_by_grade,
        analysis_by_lead,
        analysis_error_structure,
    )

    # 一个等级都没有
    assert analysis_by_grade([])
    assert analysis_error_structure([])

    # 全是 NaN：不能当 0 报，也不能崩
    nan_entries = [
        {"grade": "≥0.1", "threshold": 0.1, "TS": float("nan"), "FAR": float("nan"),
         "漏报率": float("nan"), "BIAS": float("nan"), "events": 0, "hits": 0},
    ]
    assert "无法" in analysis_by_grade(nan_entries)

    # 只有一个时效：推不出衰减，要如实说而不是编一个数
    single = _wide_rows()
    single = single[single["lead_h"] == single["lead_h"].min()]
    assert "不足" in analysis_by_lead(summarize(single))


def test_analysis_stability_calls_identical_models_indistinguishable(tmp_path):
    """两个一模一样的模型，不能硬排出一个"更稳"——那是在报告噪声。"""
    from xmetai_evaluation.visualization.ts_report import analysis_stability

    data = _wide_rows()
    text = analysis_stability({"a": data, "b": data.copy()}, "a")
    assert "接近" in text and "不构成" in text


def test_analysis_comparison_reports_ties_instead_of_a_false_winner(tmp_path):
    """三个模型数值完全一样时必须说并列——否则报告会凭空宣布一个赢家。"""
    from xmetai_evaluation.visualization.ts_report import analysis_comparison

    data = _wide_rows()
    text = analysis_comparison(data, {"FuXi": data.copy(), "AIFS": data.copy()}, 24.0)
    assert "并列" in text
    # 不许出现「某一家领先」或硬排出来的名次
    assert "可分出高下" not in text
    assert "平均名次" not in text
