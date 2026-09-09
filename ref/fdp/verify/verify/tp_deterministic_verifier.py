#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
确定性预报降水检验
参考《检验对象.docx》中 2.1.1 中国区域定量降水预报检验算法

检验指标:
1. TS (Threat Score) — 站点检验
2. Bias (偏差比) — 站点检验
3. FSS (Fractions Skill Score) — 网格邻域空间检验

检验内容:
- 6小时降水: TS/Bias 三量级(≥0.1/13.0/25.0mm) + FSS (单一阈值) + 综合评分
- 24小时降水: TS/Bias 五量级(≥0.1/10.0/25.0/50.0/100.0mm) + 综合评分
- FSS 不区分量级, 使用单一阈值(默认≥0.1mm 区分有无降水)的二值化覆盖率

综合评分:
  6h:  TS   = TS0.1×40% + TS13.0×30% + TS25×30%
       Bias = Bias0.1×40% + Bias13.0×30% + Bias25×30%
  24h: TS   = TS0.1×30% + TS10×20% + TS25×20% + TS50×20% + TS100×10%
       Bias = Bias0.1×30% + Bias10×20% + Bias25×20% + Bias50×20% + Bias100×10%

数据:
- 实况(站点): /gpu/zhaochy/fdp2/RDATA/rain/*.000 (diamond 3格式, 逐小时)
  * TS/Bias: 站点等权, 预报双线性插值到站点
  * FSS: CMPAS网格实况(优先), CRA网格实况(回退), 或站点插值到网格
- 预报: NetCDF, 参考 multi_model_verifier_fix.py
  * 路径: {fcstdata_root}/{Model}/DF/{date}/
  * 命名: {Model}_{Ver}_GLB_SURFACE_{ResTag}_6HOR_{df_tag}_{init}_{lead:03d}.nc
- CMPAS(1h降水网格实况, 优先用于FSS): GRIB2, 0.01°, 0-60°N/70-140°E
  * 路径: {cmpas_root}/{YYYYMMDD}/Z_SURF_C_BABJ_*_HOR-PRE-{YYYYMMDDHH}.GRB2
  * 6h FSS: 累加6个1h文件 (valid-5..valid 时次); 同一时次取最新入库版本
- CRA(网格实况, FSS回退): GRIB, 参考 multi_model_verifier_fix.py
  * 路径: {cra_root}/{date}/CRA40LAND_SURFACE_{date}_GLB_0P25_HOUR_V1_0_0.grib
"""

import numpy as np
import xarray as xr
import pandas as pd
import re
import argparse
import hashlib
import tempfile
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 复用 ensemble_verifier 的模型配置和工具函数
from ensemble_verifier import (
    MODEL_CONFIGS,
    CHINA_LAT_RANGE, CHINA_LON_RANGE,
    detect_variable_name,
    convert_tp,
)

# 可选依赖: scipy (FSS邻域滤波)
try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 可选依赖: matplotlib (绘图)
# 中文字体自动检测 + 双语标签: 缺失中文字体时回退英文, 避免图表出现方块乱码
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    plt.ioff()
    HAS_MPL = True

    # 中文字体候选列表 (macOS 优先, 兼顾 Linux 生产服务器)
    _CHINESE_FONT_CANDIDATES = [
        # macOS
        'PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS',
        'Songti SC', 'STSong', 'Hiragino Sans GB',
        # Linux (生产服务器)
        'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
        'Source Han Sans SC', 'SimHei', 'Microsoft YaHei',
    ]
    # 扫描系统字体, 提取字体名称
    _font_names = set()
    for _fp_path in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        try:
            _font_names.add(fm.FontProperties(fname=_fp_path).get_name())
        except Exception:
            continue
    _cn_font = None
    for _candidate in _CHINESE_FONT_CANDIDATES:
        if _candidate in _font_names:
            _cn_font = _candidate
            break
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        _HAS_CN_FONT = True
        print(f"[绘图] 使用中文字体: {_cn_font}")
    else:
        plt.rcParams['axes.unicode_minus'] = False
        _HAS_CN_FONT = False
        print("[绘图] 未找到中文字体, 图表标签将使用英文")

    # 双语标签: 有中文字体时用中文, 否则用英文
    def _T(zh, en):
        """按字体可用性返回中文或英文标签。"""
        return zh if _HAS_CN_FONT else en

    _L = {
        'lead_hour':   _T('预报时效 (h)',         'Forecast Hour'),
        'ts_ylabel':   _T('TS 评分',             'Threat Score'),
        'bias_ylabel':  _T('Bias 偏差',           'Bias'),
        'fss_ylabel':   _T('FSS',                 'FSS'),
        'ts_title_6h':  _T('6小时降水 TS 评分',   '6h Precipitation TS'),
        'bias_title_6h': _T('6小时降水 Bias 偏差', '6h Precipitation Bias'),
        'ts_title_24h': _T('24小时降水 TS 评分',  '24h Precipitation TS'),
        'bias_title_24h': _T('24小时降水 Bias 偏差', '24h Precipitation Bias'),
        'ts_composite':  _T('TS 综合评分',        'TS Composite Score'),
        'bias_composite': _T('Bias 综合评分',     'Bias Composite Score'),
        'composite_title': _T('降水综合评分',     'Precipitation Composite Score'),
        'fss_title':     _T('6小时降水 FSS',      '6h Precipitation FSS'),
    }
except ImportError:
    HAS_MPL = False
    _HAS_CN_FONT = False
    def _T(zh, en):
        return en
    _L = {}


# ======================================================================
# 检验阈值与权重
# ======================================================================
# 6h 降水检验量级阈值 (mm)
PRECIP_THRESHOLDS_6H = [0.1, 13.0, 25.0]
# 24h 降水检验量级阈值 (mm)
PRECIP_THRESHOLDS_24H = [0.1, 10.0, 25.0, 50.0, 100.0]

# 6h 综合评分权重 (对应三量级)
TS_WEIGHTS_6H = [0.4, 0.3, 0.3]
BIAS_WEIGHTS_6H = [0.4, 0.3, 0.3]
# 24h 综合评分权重 (对应五量级)
TS_WEIGHTS_24H = [0.3, 0.2, 0.2, 0.2, 0.1]
BIAS_WEIGHTS_24H = [0.3, 0.2, 0.2, 0.2, 0.1]

# FSS 邻域窗口尺寸 (n×n 格点)
FSS_WINDOW_SIZES = [1, 3, 5, 15, 31, 63]
# FSS 单一阈值 (不区分量级; 13mm 用于二值化区分有无降水)
FSS_THRESHOLD = 13


class TPDeterministicVerifier:
    """确定性预报降水检验器

    支持:
    - 6h: TS/Bias (3量级) + FSS + 综合评分
    - 24h: TS/Bias (5量级) + 综合评分

    实况数据: 站点 (*.000 diamond 3格式)
    预报数据: NetCDF (多模型)
    FSS实况: CRA网格实况 (GRIB), 回退站点插值到网格
    """

    def __init__(self, config, output_dir='./tp_deterministic_results'):
        self.obs_root = Path(config.get('obs_root',
              '/gpu/zhaochy/fdp2/RDATA/rain'))
        self.fcstdata_root = Path(config.get('fcstdata_root',
              '/gpu/zhaochy/fdp2/FCSTDATA'))
        self.forecast_root = Path(config.get('forecast_root', ''))
        self.cra_root = Path(config.get('cra_root',
              '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026'))
        # CMPAS 1h 降水实况 (0.01°, GRB2, 用于 FSS)
        # 路径: {cmpas_root}/{YYYYMMDD}/Z_SURF_C_BABJ_*_HOR-PRE-{YYYYMMDDHH}.GRB2
        self.cmpas_root = Path(config.get('cmpas_root',
              '/gpu/zhaochy/fdp2/RDATA/cmpas'))
        self.models = config.get('models', list(MODEL_CONFIGS.keys()))
        self.fss_windows = config.get('fss_windows', FSS_WINDOW_SIZES)
        self.fss_threshold = config.get('fss_threshold', FSS_THRESHOLD)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 缓存: 避免跨模型重复读取同一时次数据
        self._obs_cache = {}     # (valid_date_str, accum_hours) -> DataFrame
        self._cra_cache = {}     # (filepath_str,) -> DataArray

        self.model_styles = {
            'Fengqing':  {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-',  'linewidth': 2},
            'NJU-Earth': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'linewidth': 2},
            'PuYun':     {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'linewidth': 2},
            'YJ-TianJi': {'color': '#d62728', 'marker': 'D', 'linestyle': ':',  'linewidth': 2},
            'W2S':       {'color': '#9467bd', 'marker': 'v', 'linestyle': '-',  'linewidth': 2},
        }

    # ==================================================================
    # 站点观测读取 (diamond 3 格式 *.000)
    # ==================================================================
    def read_station_file(self, filepath):
        """读取单个 diamond 3 格式站点降水文件。

        文件格式:
          第1行: diamond  3 <描述>
          第2行: <年> <月> <日> <时> ... <站点总数>
          第3行+: <站点ID> <经度> <纬度> <海拔> <降水量>

        返回: pandas.DataFrame (station_id, lon, lat, elev, precip)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        try:
            df = pd.read_csv(
                filepath, sep=r'\s+', skiprows=2, header=None,
                names=['station_id', 'lon', 'lat', 'elev', 'precip'],
                usecols=list(range(5)),
                dtype={'station_id': str, 'lon': float, 'lat': float,
                       'elev': float, 'precip': float},
                encoding='gbk', on_bad_lines='skip')
        except Exception:
            return None
        if len(df) == 0:
            return None
        return df

    def read_station_obs(self, valid_date_str, accum_hours):
        """读取站点降水实况, 支持多时次累积。

        valid_date_str: YYYYMMDDHH (累积结束时间)
        accum_hours: 累积小时数 (6 或 24)

        从 valid_date_str 往前取 accum_hours 个时次文件累加。
        例如 valid=2026082506, accum=6: 读取 01,02,03,04,05,06 时次文件。
        """
        cache_key = (valid_date_str, accum_hours)
        if cache_key in self._obs_cache:
            return self._obs_cache[cache_key]

        valid_dt = datetime.strptime(valid_date_str, "%Y%m%d%H")
        dfs = []
        for h in range(accum_hours):
            t = valid_dt - timedelta(hours=h)
            t_str = t.strftime("%Y%m%d%H")
            filepath = self.obs_root / f"{t_str}.000"
            if filepath.exists():
                df = self.read_station_file(filepath)
                if df is not None:
                    dfs.append(df)

        if not dfs:
            print(f"    ✗ 未找到站点观测文件 ({valid_date_str}, 累积{accum_hours}h)")
            return None

        # 向量化合并: concat + groupby
        all_df = pd.concat(dfs, ignore_index=True)
        precip_sum = all_df.groupby('station_id')['precip'].sum()
        base = (dfs[0][['station_id', 'lon', 'lat', 'elev']]
                .drop_duplicates('station_id').copy())
        base = base.set_index('station_id')
        base['precip'] = precip_sum.reindex(base.index).fillna(0.0)
        base = base.reset_index()
        base.loc[base['precip'] < 0, 'precip'] = 0.0
        print(f"    站点实况: 累积{len(dfs)}/{accum_hours}时次 → {len(base)}站 "
              f"(降水范围 {base['precip'].min():.1f}-{base['precip'].max():.1f}mm)")

        self._obs_cache[cache_key] = base
        return base

    def filter_china_stations(self, df):
        """筛选中国区域站点 (经纬度范围)。"""
        mask = ((df['lat'] >= CHINA_LAT_RANGE[0]) & (df['lat'] <= CHINA_LAT_RANGE[1]) &
                (df['lon'] >= CHINA_LON_RANGE[0]) & (df['lon'] <= CHINA_LON_RANGE[1]))
        return df[mask].reset_index(drop=True)

    # ==================================================================
    # 预报数据读取 (NetCDF, 参考 multi_model_verifier_fix.py)
    # ==================================================================
    def find_model_file(self, model_name, init_date_str, lead_hour):
        """查找模型确定性预报 SURFACE 文件 (含降水)。

        路径: {fcstdata_root}/{Model}/DF/{date}/{filename}
        标准命名: {Model}_{Ver}_GLB_SURFACE_{ResTag}_6HOR_{df_tag}_{init}_{lead:03d}.nc
        W2S 命名: W2S_V1_GLB_0P25_HOUR_{init}_{lead:03d}.nc (无 SURFACE/6HOR, 所有要素同文件)
        回退: {forecast_root}/{date}/  (本地路径)
        """
        if model_name not in MODEL_CONFIGS:
            return None
        cfg = MODEL_CONFIGS[model_name]
        date_only = init_date_str[:8]
        lead_str = f"{lead_hour:03d}"

        search_dirs = []
        if self.fcstdata_root:
            search_dirs.append(self.fcstdata_root / model_name / 'DF' / date_only)
        if self.forecast_root:
            search_dirs.append(self.forecast_root / date_only)

        if cfg.get('file_pattern') == 'w2s':
            # W2S: W2S_V1_GLB_0P25_HOUR_{init}_{lead:03d}.nc
            # 不区分 SURFACE, 所有要素在同一文件
            # 确定性预报: 文件名含 HOUR 但不含 ENS_HOUR
            fname = (f"W2S_{cfg['version']}_GLB_{cfg['resolution_tag']}_"
                     f"HOUR_{init_date_str}_{lead_str}.nc")
            glob_pats = [
                f"*W2S*HOUR*{init_date_str}_{lead_str}*.nc",
                f"*W2S*{init_date_str}_{lead_str}*.nc",
            ]
            # glob 回退时排除集合预报文件 (HOUR 是 ENS_HOUR 的子串, 否则会误匹配)
            glob_exclude = 'ENS_HOUR'
        else:
            # 标准命名
            type_tag = 'SURFACE'
            ftag = cfg['df_tag']
            fname = (f"{model_name}_{cfg['version']}_GLB_{type_tag}_"
                     f"{cfg['resolution_tag']}_6HOR_{ftag}_"
                     f"{init_date_str}_{lead_str}.nc")
            glob_pats = [
                f"*{model_name}*{type_tag}*6HOR*{ftag}*{init_date_str}_{lead_str}*.nc",
                f"*{model_name}*{type_tag}*{init_date_str}_{lead_str}*.nc",
            ]
            glob_exclude = None

        for d in search_dirs:
            if not d.exists():
                continue
            cand = d / fname
            if cand.exists():
                return cand
            for pat in glob_pats:
                files = list(d.glob(pat))
                # 确定性预报: 排除集合预报文件 (W2S 的 ENS_HOUR)
                if glob_exclude:
                    files = [f for f in files if glob_exclude not in f.name]
                if files:
                    return files[0]

        # 诊断: 输出搜索过的目录, 方便排查文件缺失原因
        searched = [str(d) for d in search_dirs if d.exists()]
        if not searched:
            print(f"    ⚠ [{model_name}] 搜索目录均不存在: {list(map(str, search_dirs))}")
        else:
            print(f"    ⚠ [{model_name}] 在以下目录未找到 {fname}:")
            for s in searched:
                print(f"        - {s}")
        return None

    def read_model_tp(self, filepath, model_name, lead_hours=6):
        """读取模型降水预报 NetCDF, 返回 DataArray (lat, lon), 单位 mm。

        确定性预报: 压缩 member 维 (若有 singleton)。
        支持 W2S 模式: 从单一变量 'data' 的 channel 维度中选取 tp6 通道。
        纬度统一为降序(90→-90)以匹配实况。
        """
        if filepath is None or not Path(filepath).exists():
            return None
        cfg = MODEL_CONFIGS.get(model_name)
        if cfg is None:
            return None
        try:
            ds = xr.open_dataset(str(filepath))

            # --- W2S 模式: 单一变量 'data' + channel 维度 ---
            single_var = cfg.get('single_variable')
            channel_dim = cfg.get('channel_dim', 'channel')
            if single_var and single_var in ds.data_vars:
                da_all = ds[single_var]
                if channel_dim in da_all.dims:
                    ch_vals = da_all[channel_dim].values
                    ch_strs = np.array([str(c) for c in ch_vals])
                    var_candidates = cfg['tp']['var_candidates']
                    found = False
                    for cand in var_candidates:
                        matches = (ch_strs == cand)
                        if not matches.any():
                            matches = (np.char.lower(ch_strs) == cand.lower())
                        if matches.any():
                            idx = int(np.where(matches)[0][0])
                            da = da_all.isel({channel_dim: idx})
                            var_name = cand
                            found = True
                            break
                    if not found:
                        print(f"      ✗ [{model_name}] 通道 {var_candidates} 未在 "
                              f"{channel_dim}={list(ch_strs[:20])} 中找到")
                        ds.close()
                        return None
                else:
                    da = da_all
                    var_name = single_var
            else:
                # --- 标准模式: 变量名直接匹配 ---
                var_name = detect_variable_name(ds, cfg['tp']['var_candidates'])
                if var_name is None:
                    print(f"      ✗ [{model_name}] 未找到 TP 变量 (候选: "
                          f"{cfg['tp']['var_candidates']}, 实际: {list(ds.data_vars)})")
                    ds.close()
                    return None
                da = ds[var_name]

            # 压缩 singleton 维 (level/time/dtime/channel/member)
            for d in ('level', 'time', 'dtime', 'isobaricInhPa', 'channel',
                       'member', 'ensemble', 'ens', 'number', 'realization'):
                if d in da.dims and da.sizes[d] == 1:
                    da = da.squeeze(dim=d, drop=True)

            # 坐标统一
            if 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat', 'longitude': 'lon'})

            # 缩放因子还原 (W2S 输出对降水做了 ×1000 缩放, 需先 ÷scale_factor 还原)
            scale_factor = cfg.get('tp', {}).get('scale_factor')
            raw_vals = da.values
            if scale_factor and scale_factor != 1.0:
                raw_vals = raw_vals / scale_factor

            # 单位转换 (优先用数据自带 units, 缺失时用配置 source_unit_hint)
            source_unit = da.attrs.get('units', None)
            if not source_unit:
                source_unit = cfg.get('tp', {}).get('source_unit_hint')
            da_vals = convert_tp(raw_vals, source_unit, lead_hours=lead_hours)
            da = xr.DataArray(
                da_vals, dims=da.dims, coords=da.coords,
                attrs={**da.attrs, 'units': 'mm'})

            # 负值清零
            da = da.where(da >= 0, 0.0)

            # 纬度降序 (W2S 纬度为升序, 需反转为降序以匹配实况)
            lat_vals = da.lat.values if 'lat' in da.coords else None
            if lat_vals is not None and len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                da = da.isel(lat=slice(None, None, -1))

            ds.close()
            scale_info = f", ÷{scale_factor:g}" if (scale_factor and scale_factor != 1.0) else ""
            print(f"      [{model_name}] TP: 变量='{var_name}', "
                  f"源单位={source_unit!r} → mm{scale_info}, shape={da.shape}")
            return da
        except Exception as e:
            print(f"      [{model_name}] 读取 TP 失败: {e}")
            return None

    def read_model_tp_accumulated(self, model_name, init_date_str,
                                   end_lead, accum_hours=24):
        """读取并累加多个6h预报文件 (用于24h累计降水)。

        对于24h累计降水, 累加 leads: end_lead-18, -12, -6, end_lead
        (每个文件为该6h窗口的降水量)
        """
        n_files = accum_hours // 6
        leads = [end_lead - (n_files - 1 - i) * 6 for i in range(n_files)]

        accum_da = None
        for lead in leads:
            f = self.find_model_file(model_name, init_date_str, lead)
            if f is None:
                print(f"      ✗ [{model_name}] 预报文件缺失 (lead={lead:03d}h)")
                return None
            da = self.read_model_tp(f, model_name, lead_hours=6)
            if da is None:
                return None
            if accum_da is None:
                accum_da = da.copy()
            else:
                accum_da = accum_da + da

        print(f"      [{model_name}] {accum_hours}h累计: 累加{n_files}个6h文件 "
              f"(leads: {'+'.join(f'{l:03d}' for l in leads)})")
        return accum_da

    # ==================================================================
    # CRA 网格实况读取 (用于FSS, 参考 multi_model_verifier_fix.py)
    # ==================================================================
    def _cfgrib_indexpath(self, filepath):
        """生成可写的 cfgrib 索引文件路径 (避免只读目录权限问题)。"""
        cache_dir = Path(tempfile.gettempdir()) / 'cfgrib_idx'
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp_hash = hashlib.md5(str(filepath).encode()).hexdigest()[:8]
        return str(cache_dir / f'{fp_hash}.idx')

    def find_cra_file(self, valid_date_str):
        """查找 CRA40LAND 地面实况文件 (含降水)。

        路径: {cra_root}/{date}/CRA40LAND_SURFACE_{date}_GLB_0P25_HOUR_V1_0_0.grib
        """
        date_only = valid_date_str[:8]
        date_dir = self.cra_root / date_only
        if not date_dir.exists():
            return None

        fname = f"CRA40LAND_SURFACE_{valid_date_str}_GLB_0P25_HOUR_V1_0_0.grib"
        cand = date_dir / fname
        if cand.exists():
            return cand
        files = list(date_dir.glob(f"*SURFACE_{valid_date_str}*.grib*"))
        return files[0] if files else None

    def read_cra_precip(self, filepath):
        """读取 CRA 降水 (tp/apcp, 单位 mm)。

        尝试多种 shortName, 返回 DataArray (lat, lon)。
        """
        if filepath is None or not Path(filepath).exists():
            return None
        cache_key = (str(filepath),)
        if cache_key in self._cra_cache:
            return self._cra_cache[cache_key]

        candidates = [
            {'shortName': 'tp', 'stepType': 'accum'},
            {'shortName': 'tp'},
            {'shortName': 'apcp'},
        ]
        indexpath = self._cfgrib_indexpath(filepath)
        for keys in candidates:
            try:
                ds = xr.open_dataset(
                    str(filepath), engine='cfgrib',
                    backend_kwargs={
                        'filter_by_keys': keys,
                        'indexpath': indexpath,
                    })
                var = [v for v in ds.data_vars if v not in
                       ('latitude', 'longitude', 'valid_time', 'time', 'step')]
                if var:
                    da = ds[var[0]]
                    if 'latitude' in da.coords:
                        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
                    # 纬度降序
                    lat_vals = da.lat.values
                    if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                        da = da.isel(lat=slice(None, None, -1))
                    ds.close()
                    self._cra_cache[cache_key] = da
                    print(f"      CRA降水: 变量='{var[0]}', shape={da.shape}")
                    return da
                ds.close()
            except Exception:
                continue
        return None

    # ==================================================================
    # CMPAS 1h 降水实况读取 (0.01°, GRB2, 用于 FSS)
    #   文件名: Z_SURF_C_BABJ_{归档时间YYYYMMDDHHMMSS}_P_CMPA_{RT|NRT}_CHN_0P01_HOR-PRE-{数据时次yyyymmddhh}.GRB2
    #   1h 累积降水 (hh-1 → hh), 北京时, 0-60°N/70-140°E, 6001*7001 格点
    #   同一数据时次可能有多个入库版本, 取最新归档时间者
    # ==================================================================
    _CMPAS_ARCHIVE_TIME_RE = re.compile(r'_(\d{14})_')

    def find_cmpas_file(self, valid_date_str):
        """查找 CMPAS 1h 降水 GRB2 文件 (取最新入库版本)。

        valid_date_str: YYYYMMDDHH (10位, 整点数据时次, 北京时)
        路径: {cmpas_root}/{YYYYMMDD}/Z_SURF_C_BABJ_*_HOR-PRE-{valid_date_str}.GRB2
        返回: Path 或 None
        """
        date_only = valid_date_str[:8]
        date_dir = self.cmpas_root / date_only
        if not date_dir.exists():
            return None

        # 精确匹配数据时次 (文件名末尾 yyyymmddhh = valid_date_str)
        files = list(date_dir.glob(f"*HOR-PRE-{valid_date_str}.GRB2"))
        if not files:
            # 宽松匹配
            files = list(date_dir.glob(f"*PRE-{valid_date_str}*"))
        if not files:
            return None

        # 同一时次多个入库版本: 按归档时间(14位)取最大者=最新入库
        def _archive_time(f):
            m = self._CMPAS_ARCHIVE_TIME_RE.search(f.name)
            return m.group(1) if m else ''
        files.sort(key=_archive_time, reverse=True)
        return files[0]

    def read_cmpas_precip(self, filepath):
        """读取 CMPAS 1h 降水 (GRB2, 单位 mm)。

        CMPAS GRIB2 降水变量 paramId=0, shortName/cfVarName='unknown',
        故直接取首个数据变量。返回 DataArray (lat, lon), 纬度降序。
        """
        if filepath is None or not Path(filepath).exists():
            return None
        cache_key = (str(filepath),)
        if cache_key in self._cra_cache:
            return self._cra_cache[cache_key]

        indexpath = self._cfgrib_indexpath(filepath)
        try:
            ds = xr.open_dataset(
                str(filepath), engine='cfgrib',
                backend_kwargs={'indexpath': indexpath})
            var = [v for v in ds.data_vars if v not in
                   ('latitude', 'longitude', 'valid_time', 'time', 'step')]
            if not var:
                ds.close()
                return None
            da = ds[var[0]]
            if 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
            # 纬度降序 (CMPAS 纬度 0→60 升序, 反转为 60→0 以匹配预报)
            lat_vals = da.lat.values
            if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                da = da.isel(lat=slice(None, None, -1))
            ds.close()
            # 负值清零, 标注单位 mm
            da = da.where(da >= 0, 0.0)
            da.attrs['units'] = 'mm'
            self._cra_cache[cache_key] = da
            print(f"      CMPAS 1h降水: 变量='{var[0]}', shape={da.shape}")
            return da
        except Exception as e:
            print(f"      CMPAS读取失败: {e}")
            return None

    def read_cmpas_accumulated(self, valid_date_str, accum_hours=6):
        """读取并累加多个 CMPAS 1h 降水文件 (用于 6h/24h 累计)。

        valid_date_str: 累积结束时刻 YYYYMMDDHH (北京时, 整点)
        accum_hours: 累积小时数 (6 或 24)

        例如 valid=2026082812, accum=6: 累加 07,08,09,10,11,12 时次
        (每个文件为该1h窗口降水量, 累加得 06-12h 共6h降水)
        返回 DataArray (lat, lon), 单位 mm; 失败返回 None。
        """
        valid_dt = datetime.strptime(valid_date_str, "%Y%m%d%H")
        accum_da = None
        found = 0
        missing_times = []
        for i in range(accum_hours):
            t = valid_dt - timedelta(hours=accum_hours - 1 - i)
            t_str = t.strftime("%Y%m%d%H")
            f = self.find_cmpas_file(t_str)
            if f is None:
                missing_times.append(t_str)
                continue
            da = self.read_cmpas_precip(f)
            if da is None:
                missing_times.append(t_str)
                continue
            found += 1
            if accum_da is None:
                accum_da = da.copy()
            else:
                # CMPAS 各时次网格一致 (0.01° 6001*7001), 直接相加
                if accum_da.shape == da.shape:
                    accum_da = accum_da + da
                else:
                    # 网格不一致时插值对齐再相加
                    da = self.interpolate_to_grid(
                        da, accum_da.lat.values, accum_da.lon.values)
                    accum_da = accum_da + da

        if accum_da is None:
            print(f"    ✗ CMPAS {accum_hours}h累计失败: 无可用文件 "
                  f"(缺失: {missing_times})")
            return None
        print(f"    CMPAS {accum_hours}h累计: 累加{found}/{accum_hours}个1h文件 "
              f"(缺失{len(missing_times)}时次)")
        if missing_times:
            print(f"      缺失时次: {missing_times}")
        return accum_da

    # ==================================================================
    # 插值
    # ==================================================================
    def interpolate_to_stations(self, fcst_da, station_lats, station_lons):
        """将网格预报双线性插值到站点位置。"""
        station_lat_da = xr.DataArray(station_lats, dims='station')
        station_lon_da = xr.DataArray(station_lons, dims='station')
        try:
            return fcst_da.interp(
                lat=station_lat_da, lon=station_lon_da, method='linear')
        except Exception:
            return fcst_da.interp(
                lat=station_lat_da, lon=station_lon_da, method='nearest')

    def interpolate_to_grid(self, src_data, target_lat, target_lon):
        """将数据插值到目标网格。"""
        try:
            if 'latitude' in src_data.coords:
                src_data = src_data.rename(
                    {'latitude': 'lat', 'longitude': 'lon'})
            # 纬度方向对齐
            src_lat = src_data.lat.values
            if len(src_lat) > 1 and len(target_lat) > 1:
                src_desc = src_lat[0] > src_lat[-1]
                tgt_desc = target_lat[0] > target_lat[-1]
                if src_desc != tgt_desc:
                    src_data = src_data.isel(lat=slice(None, None, -1))
            return src_data.interp(
                lat=target_lat, lon=target_lon,
                kwargs={"fill_value": "extrapolate"})
        except Exception as e:
            print(f"      网格插值失败: {e}")
            return None

    def grid_station_to_field(self, station_df, target_lat, target_lon):
        """将站点数据插值到网格 (FSS备用, 当CRA不可用时)。

        使用 scipy.interpolate.griddata 双线性插值, 最近邻填充。
        """
        try:
            from scipy.interpolate import griddata
        except ImportError:
            print("      ✗ scipy 不可用, 站点→网格插值失败")
            return None
        points = station_df[['lon', 'lat']].values
        values = station_df['precip'].values
        lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
        grid_vals = griddata(points, values, (lon_grid, lat_grid),
                             method='linear')
        nn = griddata(points, values, (lon_grid, lat_grid), method='nearest')
        mask = np.isnan(grid_vals)
        grid_vals[mask] = nn[mask]
        grid_vals = np.where(grid_vals < 0, 0, grid_vals)
        da = xr.DataArray(
            grid_vals,
            dims=('lat', 'lon'),
            coords={'lat': target_lat, 'lon': target_lon})
        print(f"      站点→网格: {len(station_df)}站 → {da.shape}")
        return da

    # ==================================================================
    # 检验指标: TS & Bias (站点等权)
    # ==================================================================
    def compute_ts_bias(self, fcst_values, obs_values, threshold):
        """计算 TS 和 Bias。

        NA = hits   (预报≥阈值 & 实况≥阈值)
        NB = 空报   (预报≥阈值 & 实况<阈值)
        NC = 漏报   (预报<阈值 & 实况≥阈值)

        TS   = NA / (NA + NB + NC)
        Bias = (NA + NB) / (NA + NC)
        """
        mask = ~(np.isnan(fcst_values) | np.isnan(obs_values))
        fc = fcst_values[mask]
        ob = obs_values[mask]

        fc_ge = fc >= threshold
        ob_ge = ob >= threshold

        hits = np.sum(fc_ge & ob_ge)        # NA
        fa = np.sum(fc_ge & ~ob_ge)         # NB
        misses = np.sum(~fc_ge & ob_ge)     # NC

        denom_ts = hits + fa + misses
        ts = float(hits / denom_ts) if denom_ts > 0 else float('nan')

        denom_bias = hits + misses
        bias = float((hits + fa) / denom_bias) if denom_bias > 0 else float('nan')

        return ts, bias, int(hits), int(fa), int(misses)

    # ==================================================================
    # 检验指标: FSS (Fractions Skill Score, 网格邻域法)
    # ==================================================================
    def compute_fss(self, fcst_field, obs_field, threshold, window_size):
        """计算 FSS (Fractions Skill Score)。

        FSS = 1 - Σ(PF - PO)² / Σ(PF² + PO²)

        PF: 预报邻域概率 (n×n窗口内预报≥阈值的格点比例)
        PO: 实况邻域概率 (n×n窗口内实况≥阈值的格点比例)

        参数:
            fcst_field: ndarray (lat, lon) 预报降水
            obs_field:  ndarray (lat, lon) 实况降水
            threshold:  降水阈值 (mm)
            window_size: 邻域窗口边长 (n×n)
        """
        if not HAS_SCIPY:
            return float('nan')

        # NaN→0 (域外无降水)
        fc = np.nan_to_num(fcst_field, nan=0.0)
        ob = np.nan_to_num(obs_field, nan=0.0)

        # 二值化
        fc_binary = (fc >= threshold).astype(np.float64)
        ob_binary = (ob >= threshold).astype(np.float64)

        # 邻域概率 (uniform_filter 计算窗口内均值 = 覆盖率)
        pf = uniform_filter(fc_binary, size=window_size,
                            mode='constant', cval=0.0)
        po = uniform_filter(ob_binary, size=window_size,
                            mode='constant', cval=0.0)

        # 仅在有值的格点上计算 (排除域外NaN格点)
        valid_mask = ~(np.isnan(fcst_field) | np.isnan(obs_field))
        pf = pf[valid_mask]
        po = po[valid_mask]

        numerator = np.sum((pf - po) ** 2)
        denominator = np.sum(pf ** 2 + po ** 2)

        if denominator == 0:
            return float('nan')
        return float(1.0 - numerator / denominator)

    # ==================================================================
    # 综合评分
    # ==================================================================
    def compute_composite(self, scores, weights):
        """加权综合评分 (自动跳过NaN, 权重归一化)。"""
        valid = [(s, w) for s, w in zip(scores, weights) if s == s]
        if not valid:
            return float('nan')
        total_w = sum(w for _, w in valid)
        if total_w <= 0:
            return float('nan')
        return float(sum(s * w for s, w in valid) / total_w)

    # ==================================================================
    # 6h 降水检验
    # ==================================================================
    def verify_6h(self, model_name, init_date_str, lead_hour, obs_df=None):
        """6小时降水检验: TS, Bias, FSS, 综合评分。"""
        start_dt = datetime.strptime(init_date_str, "%Y%m%d%H")
        valid_dt = start_dt + timedelta(hours=lead_hour)
        valid_str = valid_dt.strftime("%Y%m%d%H")

        result = {
            'model': model_name, 'lead_hour': lead_hour,
            'valid_time': valid_str, 'accum_hours': 6,
        }

        # 读取预报 (单文件, 6h)
        fc_file = self.find_model_file(model_name, init_date_str, lead_hour)
        if not fc_file:
            print(f"    ✗ 预报文件未找到 ({model_name} lead={lead_hour}h)")
            return result
        print(f"    预报: {fc_file.name}")

        fc_tp = self.read_model_tp(fc_file, model_name, lead_hours=6)
        if fc_tp is None:
            return result

        # 读取站点观测
        if obs_df is None:
            obs_df = self.read_station_obs(valid_str, 6)
        if obs_df is None or len(obs_df) == 0:
            return result

        # 筛选中国区域站点
        china_df = self.filter_china_stations(obs_df)
        n_stations = len(china_df)
        if n_stations == 0:
            print("    ✗ 无中国区域站点")
            return result
        print(f"    中国区域站点: {n_stations}")

        # 插值预报到站点
        station_lats = china_df['lat'].values
        station_lons = china_df['lon'].values
        fc_at_stations = self.interpolate_to_stations(
            fc_tp, station_lats, station_lons)
        fc_values = np.nan_to_num(fc_at_stations.values, nan=0.0)
        obs_precip = china_df['precip'].values

        # TS, Bias (3量级)
        ts_list, bias_list = [], []
        for thr in PRECIP_THRESHOLDS_6H:
            ts, bias, hits, fa, misses = self.compute_ts_bias(
                fc_values, obs_precip, thr)
            ts_list.append(ts)
            bias_list.append(bias)
            print(f"    ≥{thr}mm: TS={ts:.4f} Bias={bias:.4f} "
                  f"(hits={hits} fa={fa} misses={misses})")

        result['ts_6h'] = ts_list
        result['bias_6h'] = bias_list

        # 综合评分
        result['ts_composite_6h'] = self.compute_composite(
            ts_list, TS_WEIGHTS_6H)
        result['bias_composite_6h'] = self.compute_composite(
            bias_list, BIAS_WEIGHTS_6H)
        print(f"    综合: TS={result['ts_composite_6h']:.4f} "
              f"Bias={result['bias_composite_6h']:.4f}")

        # FSS (网格邻域法)
        # 实况优先级: CMPAS(0.01°, 1h累加6h) > CRA(0.25°) > 站点插值到网格
        obs_grid = None

        # 1. 优先 CMPAS (1h 降水累加 6h, 0.01° 高分辨率)
        cmpas_acc = self.read_cmpas_accumulated(valid_str, accum_hours=6)
        if cmpas_acc is not None:
            obs_grid = cmpas_acc
            print(f"    FSS实况: CMPAS 6h累计 (0.01°, {obs_grid.shape})")

        # 2. 回退 CRA (0.25° 网格)
        if obs_grid is None:
            cra_file = self.find_cra_file(valid_str)
            if cra_file:
                print(f"    FSS实况: CRA {cra_file.name}")
                cra_tp = self.read_cra_precip(cra_file)
                if cra_tp is not None:
                    obs_grid = cra_tp

        if obs_grid is not None:
            # 将实况插值到预报网格 (CMPAS 0.01°/CRA 0.25° → 预报网格)
            if obs_grid.shape != fc_tp.shape:
                obs_grid = self.interpolate_to_grid(
                    obs_grid, fc_tp.lat.values, fc_tp.lon.values)
        else:
            # 回退: 站点插值到网格
            print("    CMPAS/CRA实况均不可用, 回退站点→网格插值")
            obs_grid = self.grid_station_to_field(
                china_df, fc_tp.lat.values, fc_tp.lon.values)

        if obs_grid is not None:
            fc_field = np.nan_to_num(fc_tp.values, nan=0.0)
            obs_field = np.nan_to_num(obs_grid.values, nan=0.0)

            # 中国区域mask
            lat_vals = fc_tp.lat.values
            lon_vals = fc_tp.lon.values
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
            china_mask = ((lat2d >= CHINA_LAT_RANGE[0]) &
                          (lat2d <= CHINA_LAT_RANGE[1]) &
                          (lon2d >= CHINA_LON_RANGE[0]) &
                          (lon2d <= CHINA_LON_RANGE[1]))
            fc_field_china = np.where(china_mask, fc_field, np.nan)
            obs_field_china = np.where(china_mask, obs_field, np.nan)

            # FSS 使用单一阈值 (≥13mm 二值化区分有无该量级降水)
            # 结果扁平存储: {'fss_w1':..., 'fss_w3':..., 'fss_w5':..., ...}
            fss_results = {}
            for ws in self.fss_windows:
                fss = self.compute_fss(
                    fc_field_china, obs_field_china,
                    self.fss_threshold, ws)
                fss_results[f'fss_w{ws}'] = fss
            w_str = ', '.join(f'w{ws}={fss_results[f"fss_w{ws}"]:.4f}'
                               for ws in self.fss_windows)
            print(f"    FSS (≥{self.fss_threshold}mm): {w_str}")
            result['fss_6h'] = fss_results
        else:
            print("    ✗ FSS实况数据不可用, 跳过FSS")

        result['n_stations'] = n_stations
        return result

    # ==================================================================
    # 24h 降水检验
    # ==================================================================
    def verify_24h(self, model_name, init_date_str, lead_hour, obs_df=None):
        """24小时降水检验: TS, Bias, 综合评分。"""
        start_dt = datetime.strptime(init_date_str, "%Y%m%d%H")
        valid_dt = start_dt + timedelta(hours=lead_hour)
        valid_str = valid_dt.strftime("%Y%m%d%H")

        result = {
            'model': model_name, 'lead_hour': lead_hour,
            'valid_time': valid_str, 'accum_hours': 24,
        }

        # 读取预报 (累加4个6h文件)
        fc_tp = self.read_model_tp_accumulated(
            model_name, init_date_str, lead_hour, accum_hours=24)
        if fc_tp is None:
            return result

        # 读取站点观测 (24h累积)
        if obs_df is None:
            obs_df = self.read_station_obs(valid_str, 24)
        if obs_df is None or len(obs_df) == 0:
            return result

        china_df = self.filter_china_stations(obs_df)
        n_stations = len(china_df)
        if n_stations == 0:
            print("    ✗ 无中国区域站点")
            return result
        print(f"    中国区域站点: {n_stations}")

        # 插值预报到站点
        station_lats = china_df['lat'].values
        station_lons = china_df['lon'].values
        fc_at_stations = self.interpolate_to_stations(
            fc_tp, station_lats, station_lons)
        fc_values = np.nan_to_num(fc_at_stations.values, nan=0.0)
        obs_precip = china_df['precip'].values

        # TS, Bias (5量级)
        ts_list, bias_list = [], []
        for thr in PRECIP_THRESHOLDS_24H:
            ts, bias, hits, fa, misses = self.compute_ts_bias(
                fc_values, obs_precip, thr)
            ts_list.append(ts)
            bias_list.append(bias)
            print(f"    ≥{thr}mm: TS={ts:.4f} Bias={bias:.4f} "
                  f"(hits={hits} fa={fa} misses={misses})")

        result['ts_24h'] = ts_list
        result['bias_24h'] = bias_list

        # 综合评分
        result['ts_composite_24h'] = self.compute_composite(
            ts_list, TS_WEIGHTS_24H)
        result['bias_composite_24h'] = self.compute_composite(
            bias_list, BIAS_WEIGHTS_24H)
        print(f"    综合: TS={result['ts_composite_24h']:.4f} "
              f"Bias={result['bias_composite_24h']:.4f}")

        result['n_stations'] = n_stations
        return result

    # ==================================================================
    # 多模型批量检验
    # ==================================================================
    def run_verification(self, init_date_str, hours_6h, hours_24h,
                         model_names=None):
        """多模型批量检验。"""
        if model_names is None:
            model_names = self.models

        print("\n" + "=" * 72)
        print("确定性预报降水检验 (TS / Bias / FSS)")
        print("=" * 72)
        print(f"起报时间: {init_date_str}Z")
        print(f"检验模型: {model_names}")
        print(f"6h时效: {hours_6h}")
        print(f"24h时效: {hours_24h}")
        print(f"6h阈值: {PRECIP_THRESHOLDS_6H} mm")
        print(f"24h阈值: {PRECIP_THRESHOLDS_24H} mm")
        print(f"站点实况: {self.obs_root}")
        print(f"预报路径: {self.fcstdata_root}")
        print(f"CRA路径: {self.cra_root}")
        print(f"中国区域: lat[{CHINA_LAT_RANGE[0]}, {CHINA_LAT_RANGE[1]}] "
              f"lon[{CHINA_LON_RANGE[0]}, {CHINA_LON_RANGE[1]}]")
        if not HAS_SCIPY:
            print("⚠ scipy 不可用, FSS将跳过")
        print("=" * 72)

        all_results = {}
        for model in model_names:
            print(f"\n{'─' * 48}")
            print(f"模型: {model}")

            model_results = {'6h': [], '24h': []}

            for lead in hours_6h:
                print(f"\n  [{model}] [6h] [{lead:03d}h]")
                valid_dt = (datetime.strptime(init_date_str, "%Y%m%d%H")
                            + timedelta(hours=lead))
                valid_str = valid_dt.strftime("%Y%m%d%H")
                obs_df = self.read_station_obs(valid_str, 6)
                r = self.verify_6h(model, init_date_str, lead, obs_df)
                model_results['6h'].append(r)

            for lead in hours_24h:
                print(f"\n  [{model}] [24h] [{lead:03d}h]")
                valid_dt = (datetime.strptime(init_date_str, "%Y%m%d%H")
                            + timedelta(hours=lead))
                valid_str = valid_dt.strftime("%Y%m%d%H")
                obs_df = self.read_station_obs(valid_str, 24)
                r = self.verify_24h(model, init_date_str, lead, obs_df)
                model_results['24h'].append(r)

            all_results[model] = model_results

        # 保存
        self._save_csv(all_results, init_date_str)
        if HAS_MPL:
            self._plot(all_results, init_date_str)
        else:
            print("⚠ matplotlib 不可用, 跳过绘图")

        return all_results

    # ==================================================================
    # 结果输出
    # ==================================================================
    def _save_csv(self, all_results, init_date_str):
        """保存结果到CSV。"""
        rows_6h, rows_24h = [], []

        for model, mr in all_results.items():
            # 6h
            for r in mr['6h']:
                if 'ts_6h' not in r:
                    continue
                row = {
                    'model': model,
                    'lead_hour': r.get('lead_hour'),
                    'valid_time': r.get('valid_time'),
                    'n_stations': r.get('n_stations', ''),
                }
                for i, thr in enumerate(PRECIP_THRESHOLDS_6H):
                    row[f'TS_{thr}mm'] = r['ts_6h'][i]
                    row[f'Bias_{thr}mm'] = r['bias_6h'][i]
                row['TS_composite'] = r.get('ts_composite_6h', '')
                row['Bias_composite'] = r.get('bias_composite_6h', '')
                # FSS (扁平, 不按量级)
                if 'fss_6h' in r:
                    for ws in self.fss_windows:
                        fk = f'fss_w{ws}'
                        if fk in r['fss_6h']:
                            row[f'FSS_w{ws}'] = r['fss_6h'][fk]
                rows_6h.append(row)

            # 24h
            for r in mr['24h']:
                if 'ts_24h' not in r:
                    continue
                row = {
                    'model': model,
                    'lead_hour': r.get('lead_hour'),
                    'valid_time': r.get('valid_time'),
                    'n_stations': r.get('n_stations', ''),
                }
                for i, thr in enumerate(PRECIP_THRESHOLDS_24H):
                    row[f'TS_{thr}mm'] = r['ts_24h'][i]
                    row[f'Bias_{thr}mm'] = r['bias_24h'][i]
                row['TS_composite'] = r.get('ts_composite_24h', '')
                row['Bias_composite'] = r.get('bias_composite_24h', '')
                rows_24h.append(row)

        if rows_6h:
            df = pd.DataFrame(rows_6h)
            csv = self.output_dir / f"tp_deterministic_6h_{init_date_str}.csv"
            df.to_csv(csv, index=False, float_format='%.4f')
            print(f"\n✓ 6h结果: {csv}")

        if rows_24h:
            df = pd.DataFrame(rows_24h)
            csv = self.output_dir / f"tp_deterministic_24h_{init_date_str}.csv"
            df.to_csv(csv, index=False, float_format='%.4f')
            print(f"✓ 24h结果: {csv}")

    # ==================================================================
    # 绘图
    # ==================================================================
    def _plot(self, all_results, init_date_str):
        """绘制检验结果图。"""
        # 6h TS / Bias / 综合评分 / FSS
        self._plot_metric(all_results, init_date_str, '6h', 'ts',
                          PRECIP_THRESHOLDS_6H,
                          _L['ts_ylabel'], _L['ts_title_6h'])
        self._plot_metric(all_results, init_date_str, '6h', 'bias',
                          PRECIP_THRESHOLDS_6H,
                          _L['bias_ylabel'], _L['bias_title_6h'])
        self._plot_composite(all_results, init_date_str, '6h')
        if self.fss_windows:
            self._plot_fss(all_results, init_date_str)
            # 每个窗口尺寸绘制一张五模型横向比较图
            for ws in self.fss_windows:
                self._plot_fss_comparison(all_results, init_date_str, ws)

        # 24h TS / Bias / 综合评分
        self._plot_metric(all_results, init_date_str, '24h', 'ts',
                          PRECIP_THRESHOLDS_24H,
                          _L['ts_ylabel'], _L['ts_title_24h'])
        self._plot_metric(all_results, init_date_str, '24h', 'bias',
                          PRECIP_THRESHOLDS_24H,
                          _L['bias_ylabel'], _L['bias_title_24h'])
        self._plot_composite(all_results, init_date_str, '24h')

    def _plot_metric(self, all_results, init_date_str, period, metric,
                     thresholds, ylabel, title_prefix):
        """绘制各量级TS/Bias随预报时效变化。"""
        metric_key = f'{metric}_{period}'
        n_thr = len(thresholds)
        fig, axes = plt.subplots(1, n_thr, figsize=(5 * n_thr, 5))
        if n_thr == 1:
            axes = [axes]

        for idx, thr in enumerate(thresholds):
            ax = axes[idx]
            has_data = False
            for model, mr in all_results.items():
                results = mr[period]
                pairs = []
                for r in results:
                    if (metric_key in r and r[metric_key]
                            and idx < len(r[metric_key])):
                        v = r[metric_key][idx]
                        if v == v:
                            pairs.append((r['lead_hour'], v))
                if pairs:
                    has_data = True
                    pairs.sort()
                    ls = [p[0] for p in pairs]
                    vs = [p[1] for p in pairs]
                    style = self.model_styles.get(model, {})
                    ax.plot(ls, vs, label=model,
                            color=style.get('color'),
                            marker=style.get('marker', 'o'),
                            linestyle=style.get('linestyle', '-'),
                            linewidth=style.get('linewidth', 2))
            ax.set_xlabel(_L['lead_hour'])
            ax.set_ylabel(ylabel)
            ax.set_title(f'≥{thr}mm')
            ax.grid(True, alpha=0.3)
            if has_data:
                ax.legend(fontsize=8)

        plt.suptitle(f'{title_prefix} ({init_date_str})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = self.output_dir / f'tp_{period}_{metric}_{init_date_str}.png'
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"✓ 图表: {path}")

    def _plot_composite(self, all_results, init_date_str, period):
        """绘制综合评分 (TS + Bias)。"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_idx, metric in enumerate(['ts', 'bias']):
            ax = axes[ax_idx]
            key = f'{metric}_composite_{period}'
            ylabel = _L['ts_composite'] if metric == 'ts' else _L['bias_composite']
            has_data = False
            for model, mr in all_results.items():
                results = mr[period]
                pairs = [(r['lead_hour'], r[key]) for r in results
                         if key in r and r[key] == r[key]]
                if pairs:
                    has_data = True
                    pairs.sort()
                    ls = [p[0] for p in pairs]
                    vs = [p[1] for p in pairs]
                    style = self.model_styles.get(model, {})
                    ax.plot(ls, vs, label=model,
                            color=style.get('color'),
                            marker=style.get('marker', 'o'),
                            linestyle=style.get('linestyle', '-'),
                            linewidth=style.get('linewidth', 2))
            ax.set_xlabel(_L['lead_hour'])
            ax.set_ylabel(ylabel)
            ax.set_title(f'{period.upper()} {ylabel}')
            ax.grid(True, alpha=0.3)
            if has_data:
                ax.legend(fontsize=9)

        plt.suptitle(f'{_L["composite_title"]} ({init_date_str})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = self.output_dir / f'tp_{period}_composite_{init_date_str}.png'
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"✓ 图表: {path}")

    def _plot_fss(self, all_results, init_date_str):
        """绘制FSS (单图, 每个窗口尺寸一条曲线, 跨模型独立子图)。"""
        n_models = len(all_results)
        if n_models == 0:
            return
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                                 squeeze=False)
        axes = axes.flatten()

        # 每个子图对应一个模型, 内含多个窗口尺寸的曲线
        for ax_idx, (model, mr) in enumerate(all_results.items()):
            ax = axes[ax_idx]
            results = mr['6h']
            has_data = False
            for ws in self.fss_windows:
                fk = f'fss_w{ws}'
                pairs = []
                for r in results:
                    if 'fss_6h' in r and fk in r['fss_6h']:
                        v = r['fss_6h'][fk]
                        if v == v:
                            pairs.append((r['lead_hour'], v))
                if pairs:
                    has_data = True
                    pairs.sort()
                    ls = [p[0] for p in pairs]
                    vs = [p[1] for p in pairs]
                    ax.plot(ls, vs, label=f'window {ws}×{ws}',
                            linewidth=2, marker='o', markersize=4)
            ax.set_xlabel(_L['lead_hour'])
            ax.set_ylabel(_L['fss_ylabel'])
            ax.set_title(f'{model} (≥{self.fss_threshold}mm)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            if has_data:
                ax.legend(fontsize=8)

        plt.suptitle(f'{_L["fss_title"]} ({init_date_str})',
                     fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = self.output_dir / f'tp_6h_fss_{init_date_str}.png'
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"✓ 图表: {path}")

    def _plot_fss_comparison(self, all_results, init_date_str, window):
        """绘制指定窗口尺寸下五模型 FSS 横向比较 (单图, 每模型一条曲线)。

        window: 邻域窗口边长 (n×n 格点)
        横轴=预报时效 lead_hour, 纵轴=FSS_w{window}, 各模型用独立配色。
        用于直接对比不同模型在该邻域尺度上的降水落区预报能力。
        """
        ws = window
        fk = f'fss_w{ws}'
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        for model, mr in all_results.items():
            results = mr.get('6h', [])
            pairs = []
            for r in results:
                if 'fss_6h' in r and fk in r['fss_6h']:
                    v = r['fss_6h'][fk]
                    if v == v:  # 过滤 NaN
                        pairs.append((r['lead_hour'], v))
            if not pairs:
                continue
            has_data = True
            pairs.sort()
            ls = [p[0] for p in pairs]
            vs = [p[1] for p in pairs]
            style = self.model_styles.get(model, {})
            ax.plot(ls, vs,
                    color=style.get('color'),
                    marker=style.get('marker', 'o'),
                    linestyle=style.get('linestyle', '-'),
                    linewidth=style.get('linewidth', 2),
                    label=model, markersize=5)
        if not has_data:
            plt.close(fig)
            return
        ax.set_xlabel(_L['lead_hour'])
        ax.set_ylabel(_L['fss_ylabel'])
        title = _T('6h降水 FSS 横向比较 (w=%d, ≥%dmm)' % (ws, self.fss_threshold),
                   '6h Precip FSS Comparison (w=%d, ≥%dmm)' % (ws, self.fss_threshold))
        ax.set_title(f'{title} ({init_date_str})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc='best')
        plt.tight_layout()
        path = self.output_dir / f'tp_6h_fss_w{ws}_comparison_{init_date_str}.png'
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"✓ 图表: {path}")


# ======================================================================
# 配置
# ======================================================================
def get_config(obs_root=None, fcstdata_root=None, cra_root=None,
               forecast_root=None, models=None, fss_windows=None,
               fss_threshold=None):
    """生产环境配置。

    站点实况: /gpu/zhaochy/fdp2/RDATA/rain (逐小时 *.000 文件)
    预报路径: /gpu/zhaochy/fdp2/FCSTDATA/{Model}/DF/{YYYYMMDD}/
    CRA路径:  /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026 (用于FSS)
    """
    return {
        'obs_root': obs_root or '/gpu/zhaochy/fdp2/RDATA/rain',
        'fcstdata_root': fcstdata_root or '/gpu/zhaochy/fdp2/FCSTDATA',
        'cra_root': cra_root or
            '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026',
        'forecast_root': forecast_root or '',
        'models': models or list(MODEL_CONFIGS.keys()),
        'fss_windows': fss_windows or FSS_WINDOW_SIZES,
        'fss_threshold': fss_threshold or FSS_THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser(
        description='确定性预报降水检验 (TS/Bias/FSS) '
                    '- 站点实况 + 多模型确定性预报')
    parser.add_argument('--date', required=True,
                        help='起报时间 YYYYMMDDHH (如 2026082000)')
    parser.add_argument('--hours-6h', type=int, nargs='+',
                        default=[6, 12, 18, 24, 30, 36, 42, 48],
                        help='6h检验时效列表 (默认: 6-48 step 6)')
    parser.add_argument('--hours-24h', type=int, nargs='+',
                        default=[24, 48, 72, 96, 120, 144, 168, 192, 216, 240],
                        help='24h检验时效列表 (默认: 24-240 step 24)')
    parser.add_argument('--models', nargs='+',
                        default=list(MODEL_CONFIGS.keys()),
                        help=f'模型列表 (可选: {list(MODEL_CONFIGS.keys())})')
    parser.add_argument('--obs-root',
                        default='/gpu/zhaochy/fdp2/RDATA/rain',
                        help='站点观测根目录 (*.000文件)')
    parser.add_argument('--fcstdata-root',
                        default='/gpu/zhaochy/fdp2/FCSTDATA',
                        help='预报数据根目录 (预期 {Model}/DF/{date}/ 子目录)')
    parser.add_argument('--forecast-root', default='',
                        help='本地预报数据根目录 (回退路径)')
    parser.add_argument('--cra-root',
                        default='/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026',
                        help='CRA实况根目录 (用于FSS)')
    parser.add_argument('--fss-windows', type=int, nargs='+',
                        default=FSS_WINDOW_SIZES,
                        help='FSS邻域窗口尺寸列表 (默认: 1 3 5 15 31 63)')
    parser.add_argument('--fss-threshold', type=float,
                        default=FSS_THRESHOLD,
                        help='FSS单一阈值 (mm, 默认13 区分有无降水)')
    parser.add_argument('--output-dir',
                        default='./tp_deterministic_results',
                        help='输出目录')
    parser.add_argument('--no-24h', action='store_true',
                        help='跳过24h检验')
    parser.add_argument('--no-6h', action='store_true',
                        help='跳过6h检验')
    args = parser.parse_args()

    if len(args.date) != 10:
        print("Error: --date must be YYYYMMDDHH")
        return

    # 校验模型名
    invalid = [m for m in args.models if m not in MODEL_CONFIGS]
    if invalid:
        print(f"Error: 未知模型 {invalid}")
        print(f"可用: {list(MODEL_CONFIGS.keys())}")
        return

    config = get_config(
        obs_root=args.obs_root,
        fcstdata_root=args.fcstdata_root,
        cra_root=args.cra_root,
        forecast_root=args.forecast_root,
        models=args.models,
        fss_windows=args.fss_windows,
        fss_threshold=args.fss_threshold,
    )

    verifier = TPDeterministicVerifier(config, args.output_dir)

    hours_6h = [] if args.no_6h else args.hours_6h
    hours_24h = [] if args.no_24h else args.hours_24h

    print("确定性预报降水检验...")
    print(f"站点实况: {config['obs_root']}")
    print(f"预报路径: {config['fcstdata_root']}")
    print(f"CRA路径: {config['cra_root']}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    verifier.run_verification(
        args.date, hours_6h, hours_24h, args.models)


if __name__ == '__main__':
    main()
