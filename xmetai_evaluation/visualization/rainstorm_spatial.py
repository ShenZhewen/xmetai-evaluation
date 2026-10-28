# -*- coding: utf-8 -*-
"""暴雨空间降水图的共用件：分级色标、裁剪范围、南海诸岛小图。

读的是 ``skills/xmetai-evaluation/scripts/extract_rainstorm_spatial.py`` 聚合出的
目录（manifest.json + pred_*.nc + era_*.nc）。静态的多格大图已经撤掉——报告里的
空间分布一律是逐帧播放的动图，见 ``rainstorm_spatial_anim.py``；这里只留它复用的
常量与绘图片段（色标、范围、小图）。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape

# 中日韩标签：matplotlib 默认的 DejaVu Sans 没有汉字，不设就整张图都是「□」。
# 图的文字现在一律英文，但旧图/注释里的汉字仍要走这套回退；按 Windows / Linux
# 常见字体依次找，末尾留 DejaVu 兜底（缺字体时不至于报错）。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Micro Hei", "DejaVu Sans",
]
# 负号也要跟着换字体，否则和中文混排时字号不齐。
plt.rcParams["axes.unicode_minus"] = False

#: 中央气象台《全国降水量预报图》的雨量六档配色（2026-10-28 从用户给的样图
#: 图例逐格取样）：浅绿 0–10、绿 10–25、浅蓝 25–50、蓝 50–100、
#: 品红 100–250、暗红 ≥250 mm。样图里另有雨夹雪斜线与雪的灰阶，本产品不画。
PRECIP_COLORS = [
    "#A6F28F",  # 0–10
    "#3DBA3D",  # 10–25
    "#61B8FF",  # 25–50
    "#0000FF",  # 50–100
    "#FA00FA",  # 100–250
    "#950006",  # ≥250
]

#: 口径 A（24h）分级：中央气象台雨量档 0–10–25–50–100–250 mm。
#: 上界写成 1e6 让 ≥250mm 全部落在暗红档；<0.1mm（无降水）在作图时置成
#: NaN 留白，与样图一致。
PRECIP_LEVELS_MM = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0, 1e6]

#: 过程累积示意（S 单起报、整段，不参与评分）用同一套配色、边界放大 4 倍：
#: 0–40–100–200–400–1000 mm，1000 mm 以上暗红。多日累积比 24h 大一个量级，
#: 直接套 0–250 会把整片华南压进暗红档。
TOTAL_LEVELS_MM = [0.1, 40.0, 100.0, 200.0, 400.0, 1000.0, 1e6]

#: 南海诸岛小图的范围（lon0, lon1, lat0, lat1）。主图下沿在 15°N，
#: 九段线最南一段在 3.4°N，所以按国内出版地图的惯例在主图右下角嵌一张小图，
#: 而不是把主图整体南扩（那样中国大陆会缩掉两成）。
SCS_EXTENT = [105.0, 123.0, 2.0, 25.0]

_ASSET_DIR = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=1)
def _maritime_boundary() -> cfeature.ShapelyFeature:
    """九段线（含台湾海峡两段，共 10 段断续线）。

    数据取自 cnmaps-data 1.1.2 的 ``administrative/amap/maritime``（MIT），
    随包放在 ``visualization/assets/``，见同目录 README.md。段与段之间本来
    就是断开的，直接画实线就是断续线的样子，不用再设虚线。
    """
    with open(_ASSET_DIR / "china_maritime_boundary.geojson", encoding="utf-8") as f:
        geometry = shape(json.load(f)["geometry"])
    return cfeature.ShapelyFeature(
        [geometry], ccrs.PlateCarree(), facecolor="none", edgecolor="#a33", linewidth=0.9
    )

# 离散色表：A、B 共用同一套六档颜色，只是边界不同；低于首档（无降水）
# 走 under 色留白
_PRECIP_CMAP = mcolors.ListedColormap(PRECIP_COLORS)
_PRECIP_CMAP.set_under("white")
_PRECIP_TOTAL_CMAP = mcolors.ListedColormap(PRECIP_COLORS)
_PRECIP_TOTAL_CMAP.set_under("white")


def _region_pad(region):
    """给裁剪盒留一点边，图更透气。"""
    dlat = 3.0
    dlon = 2.0
    return [
        max(-90.0, region["lon0"] - dlon),
        min(360.0, region["lon1"] + dlon),
        max(-90.0, region["lat0"] - dlat),
        min(90.0, region["lat1"] + dlat),
    ]


def _add_scs_inset(fig, ax, lonm, latm, zz, norm, width_in: float = 0.50,
                   pad_in: float = 0.05, cmap=None):
    """在子图右下角嵌一张南海诸岛小图：同数据、同色标，另画九段线。

    小图位置**按子图实际绘图框**算（figure 坐标），不用 ``ax.inset_axes``：
    GeoAxes 是等比例的，draw 时会把自己的框从 gridspec 格子里收缩，而
    ``inset_axes`` 在收缩前就定好位置，结果会掉到地图框外。
    """
    pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    lon0, lon1, lat0, lat1 = SCS_EXTENT
    height_in = width_in * (lat1 - lat0) / (lon1 - lon0)  # 1:1 等比例，框要跟着数据长宽比
    # pos / box 都是 figure 比例，width_in / pad_in 是英寸，换算后再相减。
    box = [
        pos.x1 - (width_in + pad_in) / fig_w,
        pos.y0 + pad_in / fig_h,
        width_in / fig_w,
        height_in / fig_h,
    ]
    # 小图下面垫一圈白边：不垫的话小图的降水场会和主图连成一片，看不出是插图。
    margin = 0.02  # 英寸
    fig.patches.append(
        Rectangle(
            (box[0] - margin / fig_w, box[1] - margin / fig_h),
            box[2] + 2 * margin / fig_w,
            box[3] + 2 * margin / fig_h,
            transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=4,
        )
    )

    inset = fig.add_axes(box, projection=ccrs.PlateCarree(), zorder=5)
    inset.set_extent(SCS_EXTENT, crs=ccrs.PlateCarree())
    inset.pcolormesh(
        lonm, latm, zz, cmap=_PRECIP_CMAP if cmap is None else cmap, norm=norm,
        transform=ccrs.PlateCarree(), shading="auto", rasterized=True, zorder=1,
    )
    # 与主图同用 50m：既省一次 Natural Earth 下载，矢量也只是被裁到南海那一小块。
    inset.coastlines(resolution="50m", color="#4a4a4a", linewidth=0.3, zorder=2)
    inset.add_feature(_maritime_boundary(), zorder=3)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
    return inset
