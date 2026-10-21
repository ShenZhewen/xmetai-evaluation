"""Unit tests for the shared visualization palette."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("matplotlib")

from matplotlib.colors import to_hex  # noqa: E402

from xmetai_evaluation.visualization.style import (  # noqa: E402
    BASE_WIDTH,
    CMAP_DIV,
    CMAP_SEQ,
    FIGURE_WIDTHS,
    GRADE_COLORS,
    GRADE_RAMP,
    MIN_PT,
    MODEL_COLOR_MAP,
    MODEL_FILL_COLORS,
    MODEL_LINE_COLORS,
    MODEL_PALETTE,
    NEUTRAL,
    TYPE_SCALE,
    font_sizes,
    grade_colors,
    model_colors,
    snap_width,
)

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_palette_is_unique_and_valid():
    assert len(MODEL_PALETTE) == 10
    assert len(set(MODEL_PALETTE)) == 10
    assert all(_HEX.match(color) for color in MODEL_PALETTE)
    assert MODEL_LINE_COLORS == MODEL_PALETTE[:8]
    assert MODEL_FILL_COLORS == MODEL_PALETTE[8:]
    # 填充色不出现在线条池里，避免同一个模型在线图和面积图里颜色打架
    assert not set(MODEL_LINE_COLORS) & set(MODEL_FILL_COLORS)


def test_known_models_have_fixed_colors():
    colors = model_colors(["FGVP", "FuXi", "AIFS", "FGVP-SPA", "FGVP-SPW", "S2B"])
    assert colors == {
        "FGVP": MODEL_COLOR_MAP["FGVP"],
        "FuXi": MODEL_COLOR_MAP["FuXi"],
        "AIFS": MODEL_COLOR_MAP["AIFS"],
        "FGVP-SPA": MODEL_COLOR_MAP["FGVP-SPA"],
        "FGVP-SPW": MODEL_COLOR_MAP["FGVP-SPW"],
        "S2B": MODEL_COLOR_MAP["S2B"],
    }
    # 只画一个模型时也是同一个颜色
    assert model_colors(["FuXi"])["FuXi"] == MODEL_COLOR_MAP["FuXi"]


def test_color_assignment_ignores_input_order():
    names = ["FGVP-SPW", "FGVP", "S2B", "AIFS", "FuXi", "FGVP-SPA"]
    assert model_colors(names) == model_colors(list(reversed(names)))


def test_unknown_models_are_deterministic_and_avoid_known_slots():
    forward = model_colors(["FuXi", "ModelB", "ModelA"])
    backward = model_colors(["ModelA", "FuXi", "ModelB"])
    assert forward == backward
    # 未知模型按名字排序取还没有被已知模型占用的槽位：
    # FuXi 占掉了雾霾蓝（第 2 槽），所以 ModelA/ModelB 拿第 1、第 3 槽。
    assert forward["ModelA"] == MODEL_LINE_COLORS[0]
    assert forward["ModelB"] == MODEL_LINE_COLORS[2]
    assert forward["FuXi"] == MODEL_COLOR_MAP["FuXi"]
    assert forward["FuXi"] not in (forward["ModelA"], forward["ModelB"])


def test_control_and_legacy_models_are_neutral_gray():
    colors = model_colors(["FGVP", "FGVP-ctrl", "FGVP_old"])
    assert colors["FGVP"] == MODEL_COLOR_MAP["FGVP"]
    assert colors["FGVP-ctrl"] == NEUTRAL["baseline"]
    assert colors["FGVP_old"] == NEUTRAL["baseline"]


def test_grade_colors_follow_the_ordered_ramp():
    grades = list(GRADE_COLORS)
    colors = grade_colors(grades)
    assert [colors[grade] for grade in grades] == list(GRADE_RAMP)
    assert len(set(GRADE_RAMP)) == len(GRADE_RAMP)
    assert all(_HEX.match(color) for color in GRADE_RAMP)


def test_unknown_grade_falls_back_to_the_ramp():
    colors = grade_colors(["≥0.1", "≥999"])
    assert colors["≥0.1"] == GRADE_COLORS["≥0.1"]
    assert colors["≥999"] in GRADE_RAMP


def test_neutral_colors_are_valid_hex():
    assert all(_HEX.match(color) for color in NEUTRAL.values())


def test_colormap_endpoints():
    assert to_hex(CMAP_SEQ(0.0)) == "#f4f1ea"
    assert to_hex(CMAP_SEQ(1.0)) == "#2a5052"
    # 发散色带的中点必须是米白，0 值才不会染上蓝/红倾向
    assert to_hex(CMAP_DIV(0.0)) == "#3d5a80"
    assert to_hex(CMAP_DIV(0.5)) == "#f4f1ea"
    assert to_hex(CMAP_DIV(1.0)) == "#b85c4e"


# --------------------------------------------------------------------- 排版


def test_font_sizes_at_double_column_are_the_nature_baseline():
    # 183 mm 双栏下的角色表：刻度 10 / 轴标签 12 / 标题 12 / 图例 10 /
    # 注释（含 inset 刻度）8 / 面板字母 12 bold / 总标题 12。
    assert font_sizes(BASE_WIDTH) == {
        "tick": 10.0,
        "label": 12.0,
        "title": 12.0,
        "legend": 10.0,
        "annot": 8.0,
        "panel": 12.0,
        "suptitle": 12.0,
    }
    assert set(font_sizes(BASE_WIDTH)) == set(TYPE_SCALE)


def test_font_sizes_scale_up_with_width():
    narrow = font_sizes(FIGURE_WIDTHS["single"])
    wide = font_sizes(FIGURE_WIDTHS["wide"])
    full = font_sizes(FIGURE_WIDTHS["full"])
    # 窄于双栏不再缩小，保持 Nature 下限量级。
    assert narrow == font_sizes(BASE_WIDTH)
    assert wide["tick"] == 17.0
    assert wide["label"] == 20.0
    assert full["tick"] == 22.0
    assert full["label"] == 27.0
    assert all(wide[role] <= full[role] for role in TYPE_SCALE)


def test_font_sizes_never_go_below_the_nature_minimum():
    for width in (0.1, 1.0, 2.0, FIGURE_WIDTHS["single"], BASE_WIDTH):
        assert all(value >= MIN_PT for value in font_sizes(width).values())
    # 非法宽度退回双栏基准，而不是抛异常。
    assert font_sizes(None) == font_sizes(BASE_WIDTH)
    assert font_sizes(0) == font_sizes(BASE_WIDTH)


def test_snap_width_uses_classes_and_never_shrinks():
    assert snap_width(3.4) == FIGURE_WIDTHS["single"]
    assert snap_width(6.0) == FIGURE_WIDTHS["double"]
    assert snap_width(9.0) == FIGURE_WIDTHS["wide"]
    assert snap_width(15.0) == FIGURE_WIDTHS["full"]
    # 类别多到 16 in 都放不下时保持动态宽度，字号继续按公式放大。
    assert snap_width(16.8) == 16.8
