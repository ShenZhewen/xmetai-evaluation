# -*- coding: utf-8 -*-
"""暴雨个例：24h 累计降水的动图（左预报 / 右实况，逐帧换日期）。

读 ``skills/xmetai-evaluation/scripts/extract_rainstorm_spatial.py`` 聚合出的目录
（manifest.json + pred_*.nc + era_*.nc）。两种口径：

    口径 A（默认）  一帧 = 一个 24h 窗，帧数 = 目录里的起报日数；
    过程累积示意（``plot_rainstorm_total``）  目录只有一个起报日（S），一张图 =
    整段累积（S 起报、全过程），单帧 GIF、色标 0–1000mm，**不参与评分**。

一帧的版面：

    左 = 预报（模型名见 manifest 与图注），右 = ERA5 实况，
    最右是两者共用的中央气象台雨量配色色条（A 用 0–10–25–50–100–250mm、
    B 用放大 4 倍的 0–40–100–200–400–1000mm；浅绿→绿→浅蓝→蓝→品红→暗红），
    每格右下角带南海诸岛小图（含九段线）。

图上文字一律英文：``Forecast`` / ``ERA5`` + 日期（B 用 ``process total Nh``）。
模型名不写进帧里——名字可能很长（``Tongyun_tp_corr``），报告图注已经写明
是哪家的预报。

色标、范围和小图跟 ``rainstorm_spatial.py`` 共用一份；不画经纬虚线——
动图帧小，虚线会和海岸线抢视线。

用法：
    python -m xmetai_evaluation.visualization.rainstorm_spatial_anim \
        --dir report/rainstorm_202510_spatial --out report/rainstorm_202510_spatial.gif
    python -m xmetai_evaluation.visualization.rainstorm_spatial_anim \
        --dir report/rainstorm_spatial_total/202510 --total-window-h 96 \
        --out report/rainstorm_202510_spatial_total.gif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 色标、范围、小图都放在 rainstorm_spatial 里共用，别再复制一遍
from xmetai_evaluation.visualization.rainstorm_spatial import (  # noqa: F401
    PRECIP_LEVELS_MM,
    SCS_EXTENT,
    TOTAL_LEVELS_MM,
    _PRECIP_CMAP,
    _PRECIP_TOTAL_CMAP,
    _add_scs_inset,
    _maritime_boundary,
    _region_pad,
)


def _load(root: Path, name: str):
    ds = xr.open_dataset(root / name)
    return ds["tp"].values, ds["lat"].values, ds["lon"].values


def _fmt_day(day: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]}"


def _draw_frame(day, pred, era, norm, extent, dpi=900, cmap=None, subtitle=None):
    """画一帧：左预报右实况 + 右侧公共色条。返回 figure（调用方负责关闭）。

    ``subtitle`` 缺省是帧日期（口径 A）；过程累积示意只有一张场，
    传 ``"process total 96 h"`` 之类的串代替日期。

    图上不放大标题、色条不写名字（``24h 累计降水 / mm``）——放 PPT 时这两个
    由页面上的标题和讲稿说明承担，画在帧里每翻一次都要重读一遍。

    ``dpi`` 是 GIF 唯一的清晰度杠杆（颜色被 256 色锁死）；11 in 宽 × dpi 即像素宽，
    220 dpi ≈ 2420 px，900 dpi ≈ 9900 px（文件会大一个量级）。
    """
    fig = plt.figure(figsize=(11.0, 4.6), dpi=dpi)
    spec = fig.add_gridspec(1, 2, wspace=0.06, left=0.02, right=0.88,
                            top=0.92, bottom=0.04)
    panels = []
    pcm = None
    colors = _PRECIP_CMAP if cmap is None else cmap
    shown = _fmt_day(day) if subtitle is None else subtitle
    for col, (data, label) in enumerate(((pred, "Forecast"), (era, "ERA5"))):
        z, lat, lon = data
        ax = fig.add_subplot(spec[0, col], projection=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", color="#4a4a4a", linewidth=0.6, zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5,
                       edgecolor="#777777", zorder=2)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        lonm, latm = np.meshgrid(lon, lat)
        # 低于首档（0.1mm）视为无降水：置 NaN 留白，与中央气象台样图一致
        zz = np.where(np.isfinite(z) & (z >= float(norm.boundaries[0])), z, np.nan)
        pcm = ax.pcolormesh(
            lonm, latm, zz, cmap=colors, norm=norm,
            transform=ccrs.PlateCarree(), shading="auto", zorder=1, rasterized=True,
        )
        # 全角空格是 CJK 字形，英文帧里换成普通空格
        ax.set_title(f"{label}  {shown}", fontsize=13)
        panels.append((ax, lonm, latm, zz))

    # GeoAxes 等比例收缩发生在 draw 时，先 draw 再拿真实框摆小图和色条
    fig.canvas.draw()
    for ax, lonm, latm, zz in panels:
        _add_scs_inset(fig, ax, lonm, latm, zz, norm, cmap=colors)

    pos = panels[1][0].get_position()
    cax = fig.add_axes([0.905, pos.y0, 0.016, pos.height])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    # 六档色标的档界全标（0.1 → 250 / 1000），哨兵上界 1e6 不标
    edges = list(norm.boundaries)[:-1]  # 丢掉 1e6 的哨兵上界
    step = max(1, len(edges) // 8)
    ticks = edges[::step]
    if ticks[-1] != edges[-1]:
        ticks.append(edges[-1])
    cbar.set_ticks(ticks)
    # 档界是 float，默认会写成 10.0 / 250.0；去掉多余的 .0，跟样图图例一致
    cbar.set_ticklabels([f"{tick:g}" for tick in ticks])
    return fig


def plot_rainstorm_anim(root: Path, out: Path, duration_ms: int = 1200,
                        dpi: int = 900) -> None:
    root = Path(root)
    manifest = json.load(open(root / "manifest.json", encoding="utf-8"))
    days = list(manifest["days"].keys())
    region = manifest.get("region") or dict(lat0=15.0, lat1=55.0, lon0=73.0, lon1=136.0)
    extent = _region_pad(region)
    norm = mcolors.BoundaryNorm(PRECIP_LEVELS_MM, _PRECIP_CMAP.N)

    # 模型名只进日志和图注，不进帧里的标题（见模块说明）
    print(f"过程 {manifest.get('process', '?')} / 预报模型 {manifest.get('model', '?')}")
    frames = []
    for index, day in enumerate(days, start=1):
        spec = manifest["days"][day]
        day_label = _fmt_day(day)
        print(f"帧 {index}/{len(days)} {day_label} 绘制中")
        fig = _draw_frame(
            day,
            _load(root, spec["pred"]),
            _load(root, spec["era"]),
            norm,
            extent,
            dpi=dpi,
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        # GIF 只有 256 色，逐帧自适应调色板；色标是 6 档高饱和离散色，量化后不会串档
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE, colors=256))
        plt.close(fig)
        print(f"帧 {index}/{len(days)} {day_label} 完成")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    print(f"已保存: {out}  ({len(frames)} 帧, {duration_ms} ms/帧)")


def plot_rainstorm_total(root: Path, out: Path, window_h: int,
                         duration_ms: int = 1200, dpi: int = 900) -> None:
    """过程累积示意：一根 S 起报、窗长 ``window_h`` 的**整段累积**场，单帧 GIF。

    **只作落区示意，不参与评分**——口径 B 已改成 S 单起报、逐日各评 24h。

    只有一张场（预报 vs ERA5），没有可播放的序列；仍按动图的尺寸、命名和
    色标机制出图，是为了和口径 A 的逐日动图并排时尺寸一致。色标另用
    :data:`TOTAL_LEVELS_MM`（0–1000mm，同一套雨量配色、边界放大 4 倍）——
    多日累积比 24h 大一个量级，共用 0–250 的档位会把整片华南压进暗红档。

    目录里必须**只有一个起报日**：示意场只取 S 那一根，多个起报日的目录
    是口径 A 的逐日场，拿错目录会把过程长度和累积量都画错。
    """
    root = Path(root)
    manifest = json.load(open(root / "manifest.json", encoding="utf-8"))
    days = list(manifest["days"])
    if len(days) != 1:
        raise ValueError(
            f"{root} 里应有 1 个起报日（口径 B），实际 {len(days)} 个；"
            "多起报日的目录是口径 A 的逐日场，别混用"
        )
    day = days[0]
    region = manifest.get("region") or dict(lat0=15.0, lat1=55.0, lon0=73.0, lon1=136.0)
    extent = _region_pad(region)
    norm = mcolors.BoundaryNorm(TOTAL_LEVELS_MM, _PRECIP_TOTAL_CMAP.N)
    print(
        f"过程累积 {manifest.get('process', '?')} / 预报模型 "
        f"{manifest.get('model', '?')} / 窗长 {window_h}h（{day} 起报）"
    )
    spec = manifest["days"][day]
    fig = _draw_frame(
        day,
        _load(root, spec["pred"]),
        _load(root, spec["era"]),
        norm,
        extent,
        dpi=dpi,
        cmap=_PRECIP_TOTAL_CMAP,
        subtitle=f"process total {window_h} h",
    )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE, colors=256)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 单帧 GIF：``save_all`` + 空 append 才是合法动图结构，直接 save 会存成 PNG 流
    frame.save(
        out, save_all=True, append_images=[], duration=duration_ms, loop=0, optimize=True,
    )
    plt.close(fig)
    print(f"已保存: {out}  (1 帧, {window_h}h 累积)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="暴雨空间分布动图（帧数 = manifest 里的天数）")
    p.add_argument("--dir", required=True, help="extract 产物目录（含 manifest.json）")
    p.add_argument("--out", default="rainstorm_spatial.gif", help="输出 gif")
    p.add_argument("--duration", type=int, default=1200, help="每帧停留毫秒")
    p.add_argument("--dpi", type=int, default=900,
                   help="出图分辨率（11 in 宽 × dpi 即像素宽；220≈2420px，900≈9900px，文件大一个量级）")
    p.add_argument("--total-window-h", type=int, default=None, metavar="H",
                   help="目录里只有一个起报日时给窗长小时数，出一张过程累积示意单帧图（不参与评分）")
    args = p.parse_args(argv)
    if args.total_window_h is not None:
        plot_rainstorm_total(Path(args.dir), Path(args.out), args.total_window_h,
                             args.duration, args.dpi)
    else:
        plot_rainstorm_anim(Path(args.dir), Path(args.out), args.duration, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
