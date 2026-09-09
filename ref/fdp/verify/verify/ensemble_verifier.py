#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集合预报检验指标算法实现
参考《4.集合预报检验指标及计算方法》

四个检验指标：
1. CRPS (Continuous Ranked Probability Score, 连续分级概率评分) —— 全球500hPa位势高度
2. Spread-Error Ratio (离散度-误差比 / Spread-Skill Ratio) —— 全球500hPa位势高度
3. BSS (Brier Skill Score, Brier技巧评分) —— 中国区域6小时降水 (≥0.1/4.0/13.0/25.0mm, 算术平均)
4. AROC (Area under ROC curve, ROC曲线下面积) —— 中国区域6小时降水 (≥0.1/4.0/13.0/25.0mm, 算术平均)

数据说明：
- 实况(观测)：CMA-RA (CRA40) 再分析数据，GRIB2/GRIB1 格式
    * 500hPa 位势高度 gh (m)：ART_ATM_GLB_0P25_6HOR_ANAL_*.grib2
    * 降水 tp (mm)：CRA40LAND_SURFACE_*.grib (本地数据未包含降水变量，算法已实现，数据具备即可运行)
- 预报：Fengqing 集合预报，NetCDF 格式，21 个集合成员
    * Z500 (m²/s² 位势) → 除以 9.80665 转换为位势高度 (m)，对应 gh@500hPa
    * TP (mm 6小时降水量)：每个时效文件为该6小时窗口的降水量

CRPS 公式 (预报与观测累积分布之间的综合方差)：
    CRPS = ∫[P(x) - P_o(x)]² dx，  P_o(x) = H(x - A)，A 为观测真值
    集合预报闭式解 (M 个等权成员)：
        CRPS_i = (1/M) Σ_m |f_{i,m} - o_i| - (1/(2M²)) Σ_m Σ_n |f_{i,m} - f_{i,n}|
    全球纬度加权：CRPS = Σ_i w_i CRPS_i / Σ_i w_i， w_i = cos(lat_i)

Spread-Error Ratio 公式：
    Spread = sqrt( (1/(N·M)) Σ_i Σ_m (f_{i,m} - f̄_i)² )          (集合离散度)
    RMSE  = sqrt( (1/N) Σ_i (f̄_i - o_i)² )                       (集合平均的均方根误差)
    Spread-Error Ratio = Spread / RMSE                            (越接近1越可靠)
    全球纬度加权版本：以 w_i=cos(lat_i) 加权各格点

BSS 公式：
    BS    = (1/N) Σ_i (P_i - O_i)²                                (Brier评分)
    BS_ref= (1/N) Σ_i (p_clim - O_i)² = p_clim(1-p_clim)          (气候概率Brier评分)
    BSS   = 1 - BS/BS_ref                                          (越大越好, 1最理想)
    其中 P_i 为集合预报概率(成员超过阈值的比例)，O_i 为二值化观测(≥阈值取1否则0)，
         p_clim 为气候概率(取观测事件频率)

AROC 公式：
    在 K 个概率区间，对每个概率阈值计算命中率 POD 与空报探测率 POFD，
    绘制 ROC 曲线 (横轴 POFD, 纵轴 POD)，曲线下面积即 AROC (越大越好, 1最理想)。
    集合预报概率取值离散 (0/M, 1/M, ..., M/M)，以 M+1 个分位为阈值。
"""

import numpy as np
import xarray as xr
import pandas as pd
import argparse
import warnings
import hashlib
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# matplotlib 为可选依赖: 缺失时跳过绘图, 但不影响指标计算与 CSV 输出
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    HAS_MPL = True

    # 中文字体配置: 自动检测系统中可用的中文字体, 避免图表出现方块乱码
    import matplotlib.font_manager as fm
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
    for fp_path in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        try:
            _font_names.add(fm.FontProperties(fname=fp_path).get_name())
        except Exception:
            continue
    _cn_font = None
    for candidate in _CHINESE_FONT_CANDIDATES:
        if candidate in _font_names:
            _cn_font = candidate
            break
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        # 未找到中文字体, 用英文标签替代
        plt.rcParams['axes.unicode_minus'] = False

    # 双语标签: 有中文字体时用中文, 否则用英文
    _HAS_CN_FONT = _cn_font is not None
    def _T(zh, en):
        """按字体可用性返回中文或英文标签。"""
        return zh if _HAS_CN_FONT else en

    _LABELS = {
        'lead_hour':   _T('预报时效 (h)',           'Forecast Hour'),
        'crps':        _T('500hPa 位势高度 CRPS',    '500hPa Z500 CRPS'),
        'ssr':         _T('500hPa 离散度-误差比',
                           '500hPa Spread-Error Ratio / Spread-Skill Ratio'),
        'spread_rmse': _T('500hPa Spread vs RMSE',  '500hPa Spread vs RMSE'),
        'rmse':        _T('500hPa RMSE',            '500hPa RMSE'),
        'bss':         _T('6h降水 BSS (中国区域, 4量级平均)',
                           '6h Precip BSS (China, 4-threshold avg)'),
        'aroc':        _T('6h降水 AROC (中国区域, 4量级平均)',
                           '6h Precip AROC (China, 4-threshold avg)'),
        'bss_title':   _T('6小时降水 BSS (中国区域)',
                           '6h Precip BSS (China)'),
        'aroc_title':  _T('6小时降水 AROC (中国区域)',
                           '6h Precip AROC (China)'),
        'bss_avg':     _T('BSS (4量级平均)',         'BSS (4-threshold avg)'),
        'aroc_avg':    _T('AROC (4量级平均)',        'AROC (4-threshold avg)'),
        'spread':      _T('Spread (集合离散度)',     'Spread (Ensemble)'),
        'rmse_label':  _T('RMSE (集合平均误差)',    'RMSE (Ensemble Mean)'),
        'ratio1':      _T('Ratio=1 (理想可靠)',     'Ratio=1 (Ideal)'),
        'init_label':  _T('起报',                   'Init'),
    }
except ImportError:
    HAS_MPL = False

GRAVITY = 9.80665  # 重力加速度，用于位势(m²/s²) → 位势高度(m) 转换

# 降水检验量级阈值 (mm, 6小时)
PRECIP_THRESHOLDS = [0.1, 4.0, 13.0, 25.0]

# 中国区域近似范围 (用于降水检验的经纬度裁剪)
CHINA_LAT_RANGE = (15.0, 55.0)
CHINA_LON_RANGE = (70.0, 140.0)


# ======================================================================
# 多模型配置注册表
#   每个模型记录文件名构成要素 + 变量候选名 + 目标单位，
#   配合自动检测函数实现跨模型数据读取与单位校准。
#   文件命名通用模式:
#     {ModelName}_{Version}_GLB_{PLEVELS|SURFACE}_{ResolutionTag}_6HOR_{ForecastTag}_{init}_{lead:03d}.nc
# ======================================================================
MODEL_CONFIGS = {
    'Fengqing': {
        'version': '1.0',
        'resolution_tag': 'OP25',   # 字母 O (与生产环境的 0P25 数字 0 不同)
        'ens_tag': 'ENS_FCST',       # 集合预报标识
        'df_tag': 'FCST',            # 确定性预报标识
        'data_format': 'netcdf',
        'z500': {
            'var_candidates': ['Z500', 'z500'],
            'target_unit': 'm',      # 目标: 位势高度(m)
        },
        'tp': {
            'var_candidates': ['TP', 'tp'],
            'target_unit': 'mm',     # 目标: 6小时累计降水(mm)
        },
    },
    'NJU-Earth': {
        'version': 'v1',
        'resolution_tag': '0P25',   # 数字 0
        'ens_tag': 'FCST',           # 注意: NJU-Earth 集合预报用 FCST 标识
        'df_tag': 'FCST',
        'data_format': 'netcdf',
        'z500': {
            'var_candidates': ['Z500', 'z500', 'Z', 'gh', 'GH'],
            'target_unit': 'm',
        },
        'tp': {
            'var_candidates': ['TP', 'tp', 'tp6', 'precipitation', 'APCP'],
            'target_unit': 'mm',
        },
    },
    'PuYun': {
        'version': 'v1',
        'resolution_tag': 'OP25',   # 字母 O
        'ens_tag': 'ENS_FCST',
        'df_tag': 'FCST',
        'data_format': 'netcdf',
        'z500': {
            'var_candidates': ['Z500', 'z500', 'Z', 'gh', 'GH'],
            'target_unit': 'm',
        },
        'tp': {
            'var_candidates': ['TP', 'tp', 'tp6', 'precipitation'],
            'target_unit': 'mm',
        },
    },
    'YJ-TianJi': {
        'version': 'v0.1',
        'resolution_tag': '0P25',   # 数字 0
        'ens_tag': 'ENS_FCST',
        'df_tag': 'FCST',
        'data_format': 'netcdf',
        'z500': {
            'var_candidates': ['Z500', 'z500', 'Z', 'gh', 'GH'],
            'target_unit': 'm',
        },
        'tp': {
            'var_candidates': ['TP', 'tp', 'tp6', 'precipitation'],
            'target_unit': 'mm',
        },
    },
    # W2S (onnx_infer_dfens.py 输出):
    #   文件名: W2S_V1_GLB_0P25_{ENS_}HOUR_{init}_{lead:03d}.nc
    #   所有通道存储在单一变量 'data' 中, 用 'channel' 维度区分 (z500/tp6/...)
    #   不区分 PLEVELS/SURFACE, 所有要素在同一文件
    #   纬度升序(-90→90), 需反转为降序以匹配实况
    #   z500 通道: 位势(m²/s²) → 除以 9.80665 得位势高度(m)
    #   tp6 通道: 6小时累计降水(mm), 无需转换
    'W2S': {
        'version': 'V1',
        'resolution_tag': '0P25',   # 数字 0
        'ens_tag': 'ENS_HOUR',      # 集合预报: ..._0P25_ENS_HOUR_...
        'df_tag': 'HOUR',           # 确定性预报: ..._0P25_HOUR_...
        'data_format': 'netcdf',
        'file_pattern': 'w2s',      # W2S 特有文件名模式 (无 PLEVELS/SURFACE/6HOR)
        'single_variable': 'data',  # 所有通道在单一变量 'data' 中
        'channel_dim': 'channel',   # 通道维度名称
        'z500': {
            'var_candidates': ['z500', 'Z500', 'gh500'],
            'target_unit': 'm',     # 位势(m²/s²) → m, 由 convert_z500 自动处理
        },
        'tp': {
            'var_candidates': ['tp6', 'tp', 'TP'],
            'target_unit': 'mm',    # 目标: 6小时累计降水(mm)
            'source_unit_hint': 'mm/s',  # W2S 存储: 6h降水率(mm/s), 需 ×21600→mm
            # W2S 输出时对降水率做了 ×1000 缩放(便于存储精度),
            # 读取后需先 ÷1000 还原为真实 mm/s, 再做单位转换。
            'scale_factor': 1000.0,
        },
    },
}


# ======================================================================
# 自动检测与单位转换工具函数
# ======================================================================
def detect_variable_name(ds, candidates):
    """从 xarray Dataset 中按候选名列表查找变量，返回变量名或 None。"""
    for name in candidates:
        if name in ds.data_vars:
            return name
    # 大小写不敏感的二次查找
    lower_map = {v.lower(): v for v in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def detect_member_dim(da):
    """自动检测集合成员维度名称。"""
    candidate_names = ['member', 'ensemble', 'ens', 'number',
                       'realization', 'members', 'perturbation']
    for dim in da.dims:
        if dim.lower() in candidate_names:
            return dim
    # 回退: 找第一个非 lat/lon/time/level 的维度
    skip = ('lat', 'lon', 'latitude', 'longitude', 'time',
            'level', 'dtime', 'valid_time', 'step')
    for dim in da.dims:
        if dim.lower() not in skip:
            return dim
    return None


def normalize_unit_string(unit_str):
    """将单位字符串归一化以便比较。"""
    if unit_str is None:
        return ''
    u = unit_str.strip().lower()
    u = u.replace(' ', '').replace('**', '^')
    # 常见等价映射
    equivalences = {
        'm^2/s^2': 'm2/s2', 'm^2s^-2': 'm2/s2', 'm2s-2': 'm2/s2',
        'm^2s^(-2)': 'm2/s2',
        'kg/m^2': 'kg/m2', 'kgm-2': 'kg/m2', 'kgm^-2': 'kg/m2',
        'kgm^(-2)': 'kg/m2', 'kg/m^2': 'kg/m2',
        'mm/s': 'mm/s', 'mms-1': 'mm/s', 'mm*s^-1': 'mm/s',
        'mm*s^(-1)': 'mm/s', 'mm/s^1': 'mm/s',
        'm/s': 'm/s', 'ms-1': 'm/s', 'm*s^-1': 'm/s',
        'millimeter': 'mm', 'millimeters': 'mm',
        'meter': 'm', 'meters': 'm', 'gpm': 'm',
    }
    return equivalences.get(u, u)


def convert_z500(data, source_unit):
    """将 Z500 数据转换为位势高度(m)。

    源单位可能是:
      - m²/s² (位势) → 除以 9.80665 得位势高度(m)
      - m / gpm (位势高度) → 无需转换
      - 未知 → 按数值范围推断
    """
    su = normalize_unit_string(source_unit)
    if su in ('m', 'gpm', 'meter', 'meters'):
        return data
    if su in ('m2/s2',):
        return data / GRAVITY
    # 未知单位: 按数值范围推断 (Z500 位势 ~48000 m²/s², 位势高度 ~5800 m)
    mean_val = float(np.nanmean(data))
    if mean_val > 10000:
        print(f"      ⚠ 单位未知({source_unit!r}), 均值{mean_val:.1f} > 10000, 推断为位势(m²/s²), 除以 g")
        return data / GRAVITY
    print(f"      ⚠ 单位未知({source_unit!r}), 均值{mean_val:.1f}, 推断为位势高度(m)")
    return data


def convert_tp(data, source_unit, lead_hours=6):
    """将降水数据转换为 6 小时累计降水量(mm)。

    源单位可能是:
      - mm (累计降水) → 无需转换
      - kg/m² (等价于 mm) → 无需转换
      - mm/s (降水率, 毫米/秒) → 乘以 lead_hours×3600
      - m/s (降水率, 米/秒) → 乘以 lead_hours×3600 (注: 结果为 m, 部分模型将 mm/s 标注为 m/s)
      - m (米) → 乘以 1000
    """
    su = normalize_unit_string(source_unit)
    if su in ('mm', 'kg/m2', ''):
        return data
    if su in ('mm/s',):
        # mm/s → mm (累计): rate(mm/s) × lead_hours × 3600(s)
        return data * lead_hours * 3600.0
    if su in ('m/s',):
        return data * lead_hours * 3600.0
    if su in ('m', 'meter', 'meters') and su != 'm2/s2':
        # 需要区分 m (米) 和 m/s — 若单位字符串明确是 m 则可能是累计降水(米)
        # 但通常降水不会用纯 m, 这里保守按降水率处理
        mean_val = float(np.nanmean(data))
        if mean_val < 0.01:
            print(f"      ⚠ 降水单位疑似 m/s, 均值{mean_val:.6f}, 转换为 {lead_hours}h 累计")
            return data * lead_hours * 3600.0
        return data * 1000.0
    # 未知单位: 按数值范围推断
    mean_val = float(np.nanmean(data))
    if mean_val < 0.01:
        print(f"      ⚠ 降水单位未知({source_unit!r}), 均值{mean_val:.6f} < 0.01, 推断为降水率(m/s)")
        return data * lead_hours * 3600.0
    print(f"      ⚠ 降水单位未知({source_unit!r}), 均值{mean_val:.4f}, 推断为 mm")
    return data


class EnsembleVerifier:
    """集合预报检验指标计算器"""

    def __init__(self, config, output_dir='./ensemble_verification_results'):
        self.cra_root = Path(config['cra_root'])
        self.forecast_root = Path(config.get('forecast_root', ''))
        # 多模型支持: 生产环境路径 /gpu/zhaochy/fdp2/FCSTDATA/{MODEL}/{ENS|DF}/{date}/
        self.fcstdata_root = Path(config.get('fcstdata_root', ''))
        self.forecast_type = config.get('forecast_type', 'ens')  # 'ens' 或 'df'
        self.models = config.get('models', list(MODEL_CONFIGS.keys()))
        # 站点降水实况: /gpu/zhaochy/fdp2/RDATA/rain/ 下的逐小时 *.000 文件
        self.obs_rain_root = Path(config.get('obs_rain_root', ''))
        self.accum_hours = config.get('accum_hours', 6)  # 降水累积小时数
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 观测数据缓存: 避免跨模型重复读取同一时次实况
        # _gh_obs_cache: key=(filepath_str, level) -> DataArray
        # _rain_obs_cache: key=(valid_date_str, accum_hours) -> DataFrame
        self._gh_obs_cache = {}
        self._rain_obs_cache = {}

        # 模型绘图样式
        self.model_styles = {
            'Fengqing':   {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-',  'linewidth': 2},
            'NJU-Earth':  {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'linewidth': 2},
            'PuYun':      {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'linewidth': 2},
            'YJ-TianJi':  {'color': '#d62728', 'marker': 'D', 'linestyle': ':',  'linewidth': 2},
            'W2S':        {'color': '#9467bd', 'marker': 'v', 'linestyle': '-',  'linewidth': 2},
        }

    # ------------------------------------------------------------------
    # 数据定位与读取
    # ------------------------------------------------------------------
    def find_cra_file(self, valid_date_str, data_type='pressure'):
        """在 CMA-RA 目录中查找实况文件。

        valid_date_str: YYYYMMDDHH (实况有效时间)
        data_type: 'pressure' -> ART_ATM (含 gh); 'surface' -> CRA40LAND (含降水/地面要素)
        实况目录按日期分文件夹，因此跨日时效(如 init+24h)需在下一日期目录查找。
        """
        date_only = valid_date_str[:8]
        date_dir = self.cra_root / date_only
        if not date_dir.exists():
            return None

        if data_type == 'pressure':
            fname = f"ART_ATM_GLB_0P25_6HOR_ANAL_{valid_date_str}.grib2"
            cand = date_dir / fname
            if cand.exists():
                return cand
            # 宽松匹配
            files = list(date_dir.glob(f"*ANAL_{valid_date_str}*.grib2"))
            return files[0] if files else None
        else:  # surface (降水等)
            fname = f"CRA40LAND_SURFACE_{valid_date_str}_GLB_0P25_HOUR_V1_0_0.grib"
            cand = date_dir / fname
            if cand.exists():
                return cand
            files = list(date_dir.glob(f"*SURFACE_{valid_date_str}*.grib*"))
            return files[0] if files else None

    def find_fengqing_file(self, init_date_str, lead_hour, data_type='pressure'):
        """查找 Fengqing 集合预报 NetCDF 文件。"""
        date_only = init_date_str[:8]
        date_dir = self.forecast_root / date_only
        if not date_dir.exists():
            return None
        type_tag = 'SURFACE' if data_type == 'surface' else 'PLEVELS'
        # 文件名: Fengqing_1.0_GLB_{PLEVELS|SURFACE}_OP25_6HOR_ENS_FCST_{init}_{lead:03d}.nc
        fname = (f"Fengqing_1.0_GLB_{type_tag}_OP25_6HOR_ENS_FCST_"
                 f"{init_date_str}_{lead_hour:03d}.nc")
        cand = date_dir / fname
        if cand.exists():
            return cand
        files = list(date_dir.glob(f"*{type_tag}*{init_date_str}_{lead_hour:03d}*.nc"))
        return files[0] if files else None

    def _cfgrib_indexpath(self, filepath):
        """生成可写的 cfgrib 索引文件路径。

        cfgrib 默认在 GRIB 文件旁创建 .idx 索引, 但生产数据目录通常只读,
        导致 PermissionError。此处将索引重定向到临时目录, 用文件路径哈希
        避免不同 GRIB 文件间的索引冲突。
        """
        cache_dir = Path(tempfile.gettempdir()) / 'cfgrib_idx'
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp_hash = hashlib.md5(str(filepath).encode()).hexdigest()[:8]
        return str(cache_dir / f'{fp_hash}.idx')

    def read_gh_obs(self, filepath, level=500):
        """读取 CMA-RA 500hPa 位势高度观测 (gh, 单位 m)。

        带缓存: 同一 GRIB 文件只读取一次, 跨模型重复调用直接返回缓存。
        """
        if filepath is None or not Path(filepath).exists():
            return None
        # 缓存检查: 同一 GRIB 文件不重复读取
        cache_key = (str(filepath), level)
        if cache_key in self._gh_obs_cache:
            return self._gh_obs_cache[cache_key]
        try:
            ds = xr.open_dataset(
                str(filepath), engine='cfgrib',
                backend_kwargs={
                    'filter_by_keys': {
                        'shortName': 'gh',
                        'typeOfLevel': 'isobaricInhPa',
                        'level': level},
                    'indexpath': self._cfgrib_indexpath(filepath),
                }
            )
            da = ds['gh']
            if 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
            ds.close()
            self._gh_obs_cache[cache_key] = da
            return da
        except Exception as e:
            print(f"      读取 gh 观测失败: {e}")
            return None

    def read_precip_obs(self, filepath):
        """读取 CMA-RA 降水观测 (tp / apcp, 单位 mm)。

        本地 CMA-RA 数据(CRA40LAND)未包含降水变量，该函数在数据具备时可用。
        """
        if filepath is None or not Path(filepath).exists():
            return None
        # 尝试多种降水 shortName
        candidates = [
            {'shortName': 'tp', 'typeOfLevel': 'surface', 'stepType': 'accum'},
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
                    ds.close()
                    return da
                ds.close()
            except Exception:
                continue
        return None

    # ==================================================================
    # 站点降水实况读取 (diamond 3 格式 *.000 文件)
    # ==================================================================
    def read_station_file(self, filepath):
        """读取单个 diamond 3 格式站点降水文件。

        文件格式:
          第1行: diamond  3 <描述>
          第2行: <年> <月> <日> <时> <其他元数据> <站点总数>
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
        print(f"      读取 {len(df)} 站点 (文件: {filepath.name})")
        return df

    def read_rain_obs(self, valid_date_str, accum_hours=None):
        """读取站点降水实况, 支持多时次累积。

        valid_date_str: YYYYMMDDHH (有效时间/累积结束时间)
        accum_hours: 累积小时数 (默认6)

        从 valid_date_str 往前取 accum_hours 个时次文件累加
        例如 valid=2026082506, accum=6: 读取 01,02,03,04,05,06 时次

        带缓存: 同一 (valid_date_str, accum_hours) 只读取一次,
        跨模型重复调用直接返回缓存结果。
        """
        if accum_hours is None:
            accum_hours = self.accum_hours

        # 缓存检查: 同一 valid_time + accum_hours 不重复读取
        cache_key = (valid_date_str, accum_hours)
        if cache_key in self._rain_obs_cache:
            return self._rain_obs_cache[cache_key]

        if not self.obs_rain_root:
            print("    ✗ 未配置站点降水实况路径 (obs_rain_root)")
            return None

        valid_dt = datetime.strptime(valid_date_str, "%Y%m%d%H")

        # 尝试累积多个时次文件
        if accum_hours > 1:
            dfs = []
            for h in range(accum_hours):
                t = valid_dt - timedelta(hours=h)
                t_str = t.strftime("%Y%m%d%H")
                filepath = self.obs_rain_root / f"{t_str}.000"
                if filepath.exists():
                    df = self.read_station_file(filepath)
                    if df is not None:
                        dfs.append(df)

            if len(dfs) >= 2:
                # 向量化合并: concat + groupby, O(N) 替代 O(N²) 循环
                all_df = pd.concat(dfs, ignore_index=True)
                precip_sum = all_df.groupby('station_id')['precip'].sum()
                # 以第一个df的站点为基准 (保留 lon/lat/elev 信息)
                base = (dfs[0][['station_id', 'lon', 'lat', 'elev']]
                        .drop_duplicates('station_id').copy())
                base = base.set_index('station_id')
                base['precip'] = precip_sum.reindex(base.index).fillna(0.0)
                base = base.reset_index()
                base.loc[base['precip'] < 0, 'precip'] = 0.0
                print(f"    实况: 累积{len(dfs)}时次 → {len(base)}站 "
                      f"(降水范围 {base['precip'].min():.1f}-"
                      f"{base['precip'].max():.1f}mm)")
                self._rain_obs_cache[cache_key] = base
                return base
            elif len(dfs) == 1:
                print(f"    实况: 仅1个时次文件, 使用单时次数据 "
                      f"({len(dfs[0])}站)")
                self._rain_obs_cache[cache_key] = dfs[0]
                return dfs[0]

        # 回退: 直接读取 valid_date_str 的文件
        filepath = self.obs_rain_root / f"{valid_date_str}.000"
        if filepath.exists():
            df = self.read_station_file(filepath)
            if df is not None:
                print(f"    实况: 单时次文件 {filepath.name} ({len(df)}站, "
                      f"未做{accum_hours}h累积)")
                self._rain_obs_cache[cache_key] = df
                return df

        print(f"    ✗ 未找到站点降水实况文件 ({valid_date_str}, "
              f"累积{accum_hours}h)")
        return None

    def filter_china_stations(self, df):
        """筛选中国区域站点 (经纬度范围)。"""
        mask = ((df['lat'] >= CHINA_LAT_RANGE[0]) & (df['lat'] <= CHINA_LAT_RANGE[1]) &
                (df['lon'] >= CHINA_LON_RANGE[0]) & (df['lon'] <= CHINA_LON_RANGE[1]))
        return df[mask].reset_index(drop=True)

    def interpolate_to_stations(self, fcst_da, station_lats, station_lons):
        """将网格预报双线性插值到站点位置。

        fcst_da: DataArray, dims (member, lat, lon) 或 (lat, lon)
        station_lats/lons: 1D array
        返回: DataArray, dims (member, station) 或 (station,)
        """
        station_lat_da = xr.DataArray(station_lats, dims='station')
        station_lon_da = xr.DataArray(station_lons, dims='station')
        try:
            result = fcst_da.interp(
                lat=station_lat_da, lon=station_lon_da, method='linear')
        except Exception:
            result = fcst_da.interp(
                lat=station_lat_da, lon=station_lon_da, method='nearest')
        return result

    def read_fengqing_z500(self, filepath):
        """读取 Fengqing Z500 集合预报，返回 (members, lat, lon) 的位势高度(m) numpy 数组与坐标。

        Z500 原始单位 m²/s²(位势) → 除以 9.80665 得位势高度(m)，与实况 gh 一致。
        纬度统一为降序(90→-90)以匹配实况。
        """
        if filepath is None or not Path(filepath).exists():
            return None
        try:
            ds = xr.open_dataset(str(filepath))
            da = ds['Z500']
            # 压缩 singleton 维 (level/time/dtime)
            for d in ('level', 'time', 'dtime'):
                if d in da.dims and da.sizes[d] == 1:
                    da = da.squeeze(dim=d, drop=True)
            if 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
            # 位势 → 位势高度
            da = da / GRAVITY
            # 纬度降序
            lat_vals = da.lat.values
            if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                da = da.isel(lat=slice(None, None, -1))
            ds.close()
            return da
        except Exception as e:
            print(f"      读取 Fengqing Z500 失败: {e}")
            return None

    def read_fengqing_tp(self, filepath):
        """读取 Fengqing TP 6小时降水集合预报 (mm)，返回 DataArray (member, lat, lon)。"""
        if filepath is None or not Path(filepath).exists():
            return None
        try:
            ds = xr.open_dataset(str(filepath))
            da = ds['TP']
            for d in ('level', 'time', 'dtime'):
                if d in da.dims and da.sizes[d] == 1:
                    da = da.squeeze(dim=d, drop=True)
            if 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
            # 负值清零(物理非负)
            da = da.where(da >= 0, 0.0)
            lat_vals = da.lat.values
            if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                da = da.isel(lat=slice(None, None, -1))
            ds.close()
            return da
        except Exception as e:
            print(f"      读取 Fengqing TP 失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 通用多模型数据定位与读取 (使用 MODEL_CONFIGS 自动检测)
    # ------------------------------------------------------------------
    def find_model_file(self, model_name, init_date_str, lead_hour,
                        data_type='pressure'):
        """通用模型文件查找器。

        优先尝试生产路径: {fcstdata_root}/{MODEL}/{ENS|DF}/{date}/
        回退到本地路径: {forecast_root}/{date}/  (仅 Fengqing 可用)

        文件命名通用模式:
          {ModelName}_{Version}_GLB_{PLEVELS|SURFACE}_{ResTag}_6HOR_{FcstTag}_{init}_{lead:03d}.nc
        W2S 特殊模式 (file_pattern='w2s'):
          W2S_V1_GLB_0P25_{ENS_}HOUR_{init}_{lead:03d}.nc
          (不区分 PLEVELS/SURFACE, 所有通道在同一文件)
        """
        if model_name not in MODEL_CONFIGS:
            print(f"    ✗ 未知模型: {model_name}")
            return None
        cfg = MODEL_CONFIGS[model_name]
        date_only = init_date_str[:8]
        ftag = cfg['ens_tag'] if self.forecast_type == 'ens' else cfg['df_tag']
        lead_str = f"{lead_hour:03d}"

        if cfg.get('file_pattern') == 'w2s':
            # W2S 文件名: W2S_V1_GLB_0P25_{ENS_}HOUR_{init}_{lead:03d}.nc
            # 不区分 PLEVELS/SURFACE, 所有通道在同一文件
            ens_prefix = "ENS_" if self.forecast_type == 'ens' else ""
            fname = (f"W2S_{cfg['version']}_GLB_{cfg['resolution_tag']}_"
                     f"{ens_prefix}HOUR_{init_date_str}_{lead_str}.nc")
            # glob 用 resolution_tag + ENS_/HOUR 区分集合/确定性
            glob_pat = (f"*{cfg['resolution_tag']}_{ens_prefix}HOUR_"
                        f"{init_date_str}_{lead_str}*.nc")
            glob_pat_loose = (f"*W2S*HOUR*{init_date_str}_{lead_str}*.nc")
        else:
            # 标准文件名模式
            type_tag = 'SURFACE' if data_type == 'surface' else 'PLEVELS'
            fname = (f"{model_name}_{cfg['version']}_GLB_{type_tag}_"
                     f"{cfg['resolution_tag']}_6HOR_{ftag}_"
                     f"{init_date_str}_{lead_str}.nc")
            glob_pat = (f"*{type_tag}*{cfg['resolution_tag']}*6HOR*"
                        f"{ftag}*{init_date_str}_{lead_str}*.nc")
            glob_pat_loose = (f"*{model_name}*{type_tag}*6HOR*"
                              f"{init_date_str}_{lead_str}*.nc")

        # 候选搜索目录列表
        search_dirs = []
        # 1) 生产路径: fcstdata_root/MODEL/ENS/date
        if self.fcstdata_root:
            sub = 'ENS' if self.forecast_type == 'ens' else 'DF'
            search_dirs.append(self.fcstdata_root / model_name / sub / date_only)
        # 2) 本地路径: forecast_root/date (仅 Fengqing)
        if self.forecast_root:
            search_dirs.append(self.forecast_root / date_only)

        for d in search_dirs:
            if not d.exists():
                continue
            # 精确匹配
            cand = d / fname
            if cand.exists():
                return cand
            # 宽松 glob
            files = list(d.glob(glob_pat))
            if files:
                return files[0]
            files = list(d.glob(glob_pat_loose))
            if files:
                return files[0]

        return None

    def read_model_z500(self, filepath, model_name):
        """通用 Z500 读取器 (自动检测变量名、单位、集合成员维度)。

        返回 DataArray (member, lat, lon), 单位 m (位势高度)。
        支持 W2S 模式: 从单一变量 'data' 的 channel 维度中选取 z500 通道。
        纬度统一为降序(90→-90)以匹配实况。
        """
        if filepath is None or not Path(filepath).exists():
            return None
        cfg = MODEL_CONFIGS.get(model_name)
        if cfg is None:
            return None
        try:
            ds = xr.open_dataset(str(filepath))
            da, var_name = self._get_model_dataarray(ds, cfg, 'z500', model_name)
            if da is None:
                ds.close()
                return None

            # 压缩 singleton 维 (level/time/dtime/channel 等)
            for d in ('level', 'time', 'dtime', 'isobaricInhPa', 'channel'):
                if d in da.dims and da.sizes[d] == 1:
                    da = da.squeeze(dim=d, drop=True)

            # 坐标名统一
            if 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
            elif 'lat' not in da.coords and 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat'})

            # 单位检测与转换
            source_unit = da.attrs.get('units', None)
            da_vals = convert_z500(da.values, source_unit)
            da = xr.DataArray(
                da_vals, dims=da.dims, coords=da.coords,
                attrs={**da.attrs, 'units': 'm'})

            # 纬度降序 (W2S 纬度为升序, 需反转为降序以匹配实况)
            lat_vals = da.lat.values if 'lat' in da.coords else \
                       da.latitude.values if 'latitude' in da.coords else None
            if lat_vals is not None and len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                da = da.isel(lat=slice(None, None, -1))

            ds.close()
            print(f"      [{model_name}] Z500: 变量='{var_name}', "
                  f"源单位={source_unit!r} → m, shape={da.shape}")
            return da
        except Exception as e:
            print(f"      [{model_name}] 读取 Z500 失败: {e}")
            return None

    def read_model_tp(self, filepath, model_name, lead_hours=6):
        """通用 TP 读取器 (自动检测变量名、单位、集合成员维度)。

        返回 DataArray (member, lat, lon), 单位 mm (6小时累计降水)。
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
            da, var_name = self._get_model_dataarray(ds, cfg, 'tp', model_name)
            if da is None:
                ds.close()
                return None

            # 压缩 singleton 维 (level/time/dtime/channel 等)
            for d in ('level', 'time', 'dtime', 'isobaricInhPa', 'channel'):
                if d in da.dims and da.sizes[d] == 1:
                    da = da.squeeze(dim=d, drop=True)

            # 坐标名统一
            if 'latitude' in da.coords:
                da = da.rename({'latitude': 'lat', 'longitude': 'lon'})

            # 缩放因子还原 (W2S 输出对降水做了 ×1000 缩放, 需先 ÷scale_factor 还原)
            scale_factor = cfg.get('tp', {}).get('scale_factor')
            raw_vals = da.values
            if scale_factor and scale_factor != 1.0:
                raw_vals = raw_vals / scale_factor

            # 单位检测与转换 (优先用数据自带 units, 缺失时用配置 source_unit_hint)
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
            lat_vals = da.lat.values if 'lat' in da.coords else \
                       da.latitude.values if 'latitude' in da.coords else None
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

    def align_to_grid(self, src_data, target_lat, target_lon):
        """将 src_data 插值到目标网格 (如预报与实况网格不一致时)。"""
        try:
            if 'latitude' in src_data.coords:
                src_data = src_data.rename({'latitude': 'lat', 'longitude': 'lon'})
            return src_data.interp(lat=target_lat, lon=target_lon,
                                   kwargs={"fill_value": "extrapolate"})
        except Exception as e:
            print(f"      网格插值失败: {e}")
            return None

    def _get_model_dataarray(self, ds, cfg, var_key, model_name):
        """从 Dataset 中选取目标变量的 DataArray。

        支持两种模式:
        1. 标准模式: 变量名直接匹配 (如 ds['Z500'])
        2. W2S 模式 (single_variable='data'): 所有通道在单一变量 'data' 中,
           通过 'channel' 维度选取 (如 ds['data'].isel(channel=z500_idx))

        返回 (da, var_name) 或 (None, None)。
        """
        var_candidates = cfg[var_key]['var_candidates']
        single_var = cfg.get('single_variable')
        channel_dim = cfg.get('channel_dim', 'channel')

        if single_var and single_var in ds.data_vars:
            # W2S 模式: 单一变量 + channel 维度
            da_all = ds[single_var]
            if channel_dim in da_all.dims:
                ch_vals = da_all[channel_dim].values
                ch_strs = np.array([str(c) for c in ch_vals])
                for cand in var_candidates:
                    matches = (ch_strs == cand)
                    if not matches.any():
                        matches = (np.char.lower(ch_strs) == cand.lower())
                    if matches.any():
                        idx = int(np.where(matches)[0][0])
                        return da_all.isel({channel_dim: idx}), cand
                print(f"      ✗ [{model_name}] 通道 {var_candidates} 未在 "
                      f"{channel_dim}={list(ch_strs[:20])} 中找到")
                return None, None
            return da_all, single_var

        # 标准模式: 变量名直接匹配
        var_name = detect_variable_name(ds, var_candidates)
        if var_name is None:
            print(f"      ✗ [{model_name}] 未找到 {var_key} 变量 (候选: "
                  f"{var_candidates}, 实际: {list(ds.data_vars)})")
            return None, None
        return ds[var_name], var_name

    # ------------------------------------------------------------------
    # 检验指标 1: CRPS (连续分级概率评分)
    # ------------------------------------------------------------------
    def compute_crps(self, ensemble, obs, lats):
        """计算纬度加权 CRPS。

        参数:
            ensemble: ndarray (M, lat, lon) 集合预报 (已与观测同网格、同单位)
            obs:      ndarray (lat, lon) 实况
            lats:     ndarray (lat,) 纬度(降序或升序均可)
        返回:
            crps: float 纬度加权平均 CRPS
        采用闭式解:
            CRPS_i = (1/M)Σ_m|f_m - o| - (1/(2M²))Σ_mΣ_n|f_m - f_n|
        其中第二项利用排序后的线性系数高效计算:
            Σ_mΣ_n|f_m - f_n| = 2·Σ_k x_(k)·(2k+1-M),  x_(k) 为升序排列的第k个成员(0-indexed)
        """
        M = ensemble.shape[0]
        # 第一项: (1/M)Σ_m |f_m - o|
        term1 = np.mean(np.abs(ensemble - obs[np.newaxis, ...]), axis=0)  # (lat, lon)

        # 第二项: (1/(2M²))Σ_mΣ_n|f_m - f_n|，用排序系数法
        ens_sorted = np.sort(ensemble, axis=0)  # (M, lat, lon) 升序
        k = np.arange(M, dtype=np.float64)
        coeff = (2.0 * k + 1.0 - M) / (M ** 2)  # (M,)  已含 1/M² 因子
        # term2 = Σ_k x_(k)·coeff_k   (等价于 (1/(2M²))·2·Σ_k x_(k)(2k+1-M) / M² ... 见模块docstring推导)
        term2 = np.sum(ens_sorted * coeff[:, np.newaxis, np.newaxis], axis=0)

        crps_field = term1 - term2  # (lat, lon)
        crps_field = np.clip(crps_field, 0.0, None)  # 数值误差保护(非负)

        # 纬度加权平均
        w = np.cos(np.deg2rad(np.abs(lats)))  # cos(lat), 与纬度方向无关
        w2d = np.broadcast_to(w[:, np.newaxis], crps_field.shape)
        wsum = np.nansum(w2d)
        crps = float(np.nansum(w2d * crps_field) / wsum) if wsum > 0 else float('nan')
        return crps

    # ------------------------------------------------------------------
    # 检验指标 2: Spread-Error Ratio (离散度-误差比)
    # ------------------------------------------------------------------
    def compute_spread_error_ratio(self, ensemble, obs, lats):
        """计算纬度加权 Spread-Error Ratio。

        依据《4.集合预报检验指标及计算方法》公式:
        Spread = sqrt( Σ_i w_i Σ_m (f_{i,m}-f̄_i)² / (Σ_i w_i·(M-1)) )
        RMSE  = sqrt( Σ_i w_i (f̄_i - o_i)² / Σ_i w_i )
        Ratio = Spread / RMSE

        Spread 分母使用 (M-1) (样本标准差, ddof=1, 无偏估计)。
        M=1 时 Spread/SSR 无定义, 返回 NaN。
        """
        M = ensemble.shape[0]
        ens_mean = np.mean(ensemble, axis=0)  # (lat, lon)

        w = np.cos(np.deg2rad(np.abs(lats)))
        w2d = np.broadcast_to(w[:, np.newaxis], ens_mean.shape)
        wsum = np.nansum(w2d)

        # Spread: 各成员与集合平均的偏差平方, 加权, 分母用 (M-1)
        if M > 1:
            spread_var = np.nansum(w2d * np.sum((ensemble - ens_mean[np.newaxis, ...]) ** 2, axis=0))
            spread = float(np.sqrt(spread_var / (wsum * (M - 1)))) if wsum > 0 else float('nan')
        else:
            spread = float('nan')

        # RMSE: 集合平均与实况的误差
        rmse_var = np.nansum(w2d * (ens_mean - obs) ** 2)
        rmse = float(np.sqrt(rmse_var / wsum)) if wsum > 0 else float('nan')

        ratio = float(spread / rmse) if (spread == spread and rmse and rmse > 0 and not np.isnan(rmse)) else float('nan')
        return ratio, spread, rmse

    # ------------------------------------------------------------------
    # 检验指标 3: BSS (Brier 技巧评分)
    # ------------------------------------------------------------------
    def compute_bss(self, prob, binary_obs, weights=None):
        """计算 BSS。

        参数:
            prob:       ndarray 集合预报概率 (0~1)
            binary_obs: ndarray 二值化观测 (0/1)
            weights:    可选权重(纬度加权), 默认等权
        BS     = Σ w (p - o)² / Σ w
        BS_ref = Σ w (p_clim - o)² / Σ w = p_clim(1 - p_clim)  (p_clim=Σw·o/Σw)
        BSS    = 1 - BS/BS_ref
        """
        if weights is None:
            weights = np.ones_like(binary_obs, dtype=np.float64)
        wsum = np.nansum(weights)
        if wsum <= 0:
            return float('nan')

        bs = float(np.nansum(weights * (prob - binary_obs) ** 2) / wsum)
        p_clim = float(np.nansum(weights * binary_obs) / wsum)  # 气候概率=观测事件频率
        bs_ref = p_clim * (1.0 - p_clim)
        if bs_ref <= 0:
            # 事件全发生或全不发生，无技巧可言
            return float('nan')
        bss = 1.0 - bs / bs_ref
        return float(bss)

    # ------------------------------------------------------------------
    # 检验指标 4: AROC (ROC 曲线下面积)
    # ------------------------------------------------------------------
    def compute_aroc(self, prob, binary_obs, weights=None):
        """计算 AROC (ROC 曲线下面积)。

        对 M 成员集合，概率取值离散 0/M..M/M，以 M+1 个概率阈值构建 ROC 曲线。
        每个阈值 p*:
            预报发生 = prob >= p*
            hits   = Σ w·(prob>=p* & obs==1)
            misses = Σ w·(prob< p* & obs==1)
            fa     = Σ w·(prob>=p* & obs==0)
            cn     = Σ w·(prob< p* & obs==0)
            POD  = hits/(hits+misses)
            POFD = fa/(fa+cn)
        ROC 曲线 (横 POFD, 纵 POD)，加入 (0,0) 与 (1,1) 端点，梯形积分得面积。
        """
        if weights is None:
            weights = np.ones_like(binary_obs, dtype=np.float64)

        total_obs = np.nansum(weights * binary_obs)  # 总发生
        total_non = np.nansum(weights * (1 - binary_obs))  # 总未发生
        if total_obs <= 0 or total_non <= 0:
            return float('nan')

        # 概率阈值: 用观测到的唯一概率值作为阈值, 从小到大
        unique_p = np.unique(prob)
        thresholds = unique_p[unique_p > 0]  # 去掉0, p*=0 即 (1,1) 端点

        points = [(0.0, 0.0)]  # (POFD, POD)
        for p_star in thresholds:
            forecast_yes = prob >= p_star
            hits = np.nansum(weights * (forecast_yes & (binary_obs == 1)))
            fa = np.nansum(weights * (forecast_yes & (binary_obs == 0)))
            pod = hits / total_obs if total_obs > 0 else 0.0
            pofd = fa / total_non if total_non > 0 else 0.0
            points.append((float(pofd), float(pod)))
        points.append((1.0, 1.0))

        # 按 POFD 升序排序后梯形积分
        points = sorted(points, key=lambda x: x[0])
        aroc = 0.0
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            aroc += (x1 - x0) * (y0 + y1) / 2.0
        return float(max(0.0, min(1.0, aroc)))

    # ------------------------------------------------------------------
    # 中国区域掩膜
    # ------------------------------------------------------------------
    def china_mask(self, lats, lons):
        """返回中国区域(近似经纬度范围)布尔掩膜。"""
        lat2d, lon2d = np.meshgrid(lats, lons, indexing='ij')
        mask = ((lat2d >= CHINA_LAT_RANGE[0]) & (lat2d <= CHINA_LAT_RANGE[1]) &
                (lon2d >= CHINA_LON_RANGE[0]) & (lon2d <= CHINA_LON_RANGE[1]))
        return mask

    # ------------------------------------------------------------------
    # 单时效检验
    # ------------------------------------------------------------------
    def verify_lead(self, init_date_str, lead_hour):
        """对单个预报时效计算四个指标。返回结果字典或 None。"""
        start_dt = datetime.strptime(init_date_str, "%Y%m%d%H")
        valid_dt = start_dt + timedelta(hours=lead_hour)
        valid_str = valid_dt.strftime("%Y%m%d%H")
        result = {'lead_hour': lead_hour, 'valid_time': valid_str}

        # ============ 500hPa 位势高度: CRPS & Spread-Error Ratio ============
        print(f"  [{lead_hour:03d}h] 有效时间 {valid_str} — 500hPa位势高度")
        fc_file = self.find_fengqing_file(init_date_str, lead_hour, 'pressure')
        obs_file = self.find_cra_file(valid_str, 'pressure')
        if fc_file and obs_file:
            print(f"    预报: {fc_file.name}")
            print(f"    实况: {obs_file.name}")
            fc = self.read_fengqing_z500(fc_file)
            obs = self.read_gh_obs(obs_file, level=500)
            if fc is not None and obs is not None:
                # 统一坐标
                if fc.shape[-2:] != obs.shape:
                    print(f"    网格不一致 预报{fc.shape[-2:]} 实况{obs.shape}，插值实况")
                    obs = self.align_to_grid(obs, fc.lat.values, fc.lon.values)
                if obs is not None:
                    ens = fc.values  # (M, lat, lon)
                    obs_arr = obs.values
                    lats = fc.lat.values
                    crps = self.compute_crps(ens, obs_arr, lats)
                    ratio, spread, rmse = self.compute_spread_error_ratio(ens, obs_arr, lats)
                    result['crps'] = crps
                    result['spread'] = spread
                    result['rmse'] = rmse
                    result['spread_error_ratio'] = ratio
                    print(f"    ✓ CRPS={crps:.4f} m | Spread={spread:.4f} | RMSE={rmse:.4f} | Ratio={ratio:.4f}")
                else:
                    print("    ✗ 实况插值失败")
            else:
                print("    ✗ 读取 Z500/gh 失败")
        else:
            missing = []
            if not fc_file:
                missing.append("预报文件")
            if not obs_file:
                missing.append("实况文件")
            print(f"    ✗ {','.join(missing)}未找到 (lead={lead_hour}h)")

        # ============ 6小时降水: BSS & AROC (站点实况) ============
        print(f"  [{lead_hour:03d}h] 有效时间 {valid_str} — 6小时降水(中国区域, 站点实况)")
        fc_tp_file = self.find_fengqing_file(init_date_str, lead_hour, 'surface')
        # 读取站点降水实况 (diamond 3 格式 *.000, 支持多时次累积)
        obs_df = self.read_rain_obs(valid_str)
        if fc_tp_file and obs_df is not None:
            print(f"    预报: {fc_tp_file.name}")
            fc_tp = self.read_fengqing_tp(fc_tp_file)
            if fc_tp is not None:
                # 筛选中国区域站点
                china_df = self.filter_china_stations(obs_df)
                n_stations = len(china_df)
                if n_stations == 0:
                    print("    ✗ 无中国区域站点, 跳过降水检验")
                else:
                    print(f"    中国区域站点: {n_stations}")
                    station_lats = china_df['lat'].values
                    station_lons = china_df['lon'].values
                    obs_arr = china_df['precip'].values.astype(np.float64)
                    # 插值预报到站点 → (member, station)
                    fc_at_stations = self.interpolate_to_stations(
                        fc_tp, station_lats, station_lons)
                    ens_tp = fc_at_stations.values  # (M, station)
                    # 站点纬度权重
                    w_station = np.cos(np.deg2rad(np.abs(station_lats)))
                    bss_list, aroc_list = [], []
                    for thr in PRECIP_THRESHOLDS:
                        binary_obs = (obs_arr >= thr).astype(np.float64)
                        prob = np.mean(ens_tp >= thr, axis=0).astype(np.float64)
                        bss = self.compute_bss(prob, binary_obs,
                                               weights=w_station)
                        aroc = self.compute_aroc(prob, binary_obs,
                                                 weights=w_station)
                        bss_list.append(bss)
                        aroc_list.append(aroc)
                        print(f"    ≥{thr}mm: BSS={bss:.4f} | AROC={aroc:.4f}")
                    result['bss'] = float(np.nanmean(bss_list))
                    result['aroc'] = float(np.nanmean(aroc_list))
                    result['bss_by_threshold'] = bss_list
                    result['aroc_by_threshold'] = aroc_list
                    result['n_stations'] = n_stations
                    print(f"    ✓ BSS(平均)={result['bss']:.4f} | "
                          f"AROC(平均)={result['aroc']:.4f}")
        else:
            missing = []
            if not fc_tp_file:
                missing.append("降水预报文件")
            if obs_df is None:
                missing.append("站点降水实况")
            print(f"    ✗ {','.join(missing)}未找到 (lead={lead_hour}h)")

        return result

    # ------------------------------------------------------------------
    # 多模型: 单模型单时效检验
    # ------------------------------------------------------------------
    def verify_lead_for_model(self, model_name, init_date_str, lead_hour):
        """对单个模型、单个预报时效计算四个指标。

        使用通用 find_model_file / read_model_z500 / read_model_tp，
        自动检测变量名、单位、集合成员维度。
        返回结果字典或 None。
        """
        start_dt = datetime.strptime(init_date_str, "%Y%m%d%H")
        valid_dt = start_dt + timedelta(hours=lead_hour)
        valid_str = valid_dt.strftime("%Y%m%d%H")
        result = {'model': model_name, 'lead_hour': lead_hour,
                  'valid_time': valid_str}

        # ============ 500hPa 位势高度: CRPS & Spread-Error Ratio ============
        print(f"  [{model_name}] [{lead_hour:03d}h] {valid_str} — 500hPa Z500")
        fc_file = self.find_model_file(model_name, init_date_str, lead_hour, 'pressure')
        obs_file = self.find_cra_file(valid_str, 'pressure')
        if fc_file and obs_file:
            print(f"    预报: {fc_file}")
            print(f"    实况: {obs_file.name}")
            fc = self.read_model_z500(fc_file, model_name)
            obs = self.read_gh_obs(obs_file, level=500)
            if fc is not None and obs is not None:
                # 检测集合成员维度并转置为 (member, lat, lon)
                mem_dim = detect_member_dim(fc)
                if mem_dim is not None and mem_dim != fc.dims[0]:
                    fc = fc.transpose(mem_dim, 'lat', 'lon')
                elif mem_dim is None:
                    print(f"    ⚠ [{model_name}] 未检测到集合成员维度, dims={fc.dims}")
                # 网格对齐
                if fc.shape[-2:] != obs.shape:
                    print(f"    网格不一致 预报{fc.shape[-2:]} 实况{obs.shape}, 插值实况")
                    obs = self.align_to_grid(obs, fc.lat.values, fc.lon.values)
                if obs is not None:
                    ens = fc.values  # (M, lat, lon)
                    obs_arr = obs.values
                    lats = fc.lat.values
                    print(f"    集合成员数 M={ens.shape[0]}")
                    crps = self.compute_crps(ens, obs_arr, lats)
                    ratio, spread, rmse = self.compute_spread_error_ratio(ens, obs_arr, lats)
                    result['crps'] = crps
                    result['spread'] = spread
                    result['rmse'] = rmse
                    result['spread_error_ratio'] = ratio
                    result['n_members'] = ens.shape[0]
                    print(f"    ✓ CRPS={crps:.4f} m | Spread={spread:.4f} | "
                          f"RMSE={rmse:.4f} | Ratio={ratio:.4f}")
                else:
                    print("    ✗ 实况插值失败")
            else:
                print("    ✗ 读取 Z500/gh 失败")
        else:
            missing = []
            if not fc_file:
                missing.append("预报文件")
            if not obs_file:
                missing.append("实况文件")
            print(f"    ✗ {','.join(missing)}未找到 ({model_name} lead={lead_hour}h)")

        # ============ 6小时降水: BSS & AROC (站点实况) ============
        print(f"  [{model_name}] [{lead_hour:03d}h] {valid_str} — "
              f"6h降水(中国区域, 站点实况)")
        fc_tp_file = self.find_model_file(model_name, init_date_str,
                                          lead_hour, 'surface')
        # 读取站点降水实况 (diamond 3 格式 *.000, 支持多时次累积)
        obs_df = self.read_rain_obs(valid_str)
        if fc_tp_file and obs_df is not None:
            print(f"    预报: {fc_tp_file}")
            fc_tp = self.read_model_tp(fc_tp_file, model_name, lead_hours=6)
            if fc_tp is not None:
                # 检测集合成员维度并转置为 (member, lat, lon)
                mem_dim = detect_member_dim(fc_tp)
                if mem_dim is not None and mem_dim != fc_tp.dims[0]:
                    fc_tp = fc_tp.transpose(mem_dim, 'lat', 'lon')
                # 筛选中国区域站点
                china_df = self.filter_china_stations(obs_df)
                n_stations = len(china_df)
                if n_stations == 0:
                    print(f"    ✗ [{model_name}] 无中国区域站点, 跳过降水检验")
                else:
                    print(f"    [{model_name}] 中国区域站点: {n_stations}")
                    station_lats = china_df['lat'].values
                    station_lons = china_df['lon'].values
                    obs_arr = china_df['precip'].values.astype(np.float64)
                    # 插值预报到站点 → (member, station)
                    fc_at_stations = self.interpolate_to_stations(
                        fc_tp, station_lats, station_lons)
                    ens_tp = fc_at_stations.values  # (M, station)
                    # 站点纬度权重
                    w_station = np.cos(np.deg2rad(np.abs(station_lats)))
                    bss_list, aroc_list = [], []
                    for thr in PRECIP_THRESHOLDS:
                        binary_obs = (obs_arr >= thr).astype(np.float64)
                        prob = np.mean(ens_tp >= thr, axis=0).astype(np.float64)
                        bss = self.compute_bss(prob, binary_obs,
                                               weights=w_station)
                        aroc = self.compute_aroc(prob, binary_obs,
                                                 weights=w_station)
                        bss_list.append(bss)
                        aroc_list.append(aroc)
                        print(f"    ≥{thr}mm: BSS={bss:.4f} | AROC={aroc:.4f}")
                    result['bss'] = float(np.nanmean(bss_list))
                    result['aroc'] = float(np.nanmean(aroc_list))
                    result['bss_by_threshold'] = bss_list
                    result['aroc_by_threshold'] = aroc_list
                    result['n_stations'] = n_stations
                    print(f"    ✓ [{model_name}] BSS(平均)={result['bss']:.4f} | "
                          f"AROC(平均)={result['aroc']:.4f}")
        else:
            missing = []
            if not fc_tp_file:
                missing.append("降水预报文件")
            if obs_df is None:
                missing.append("站点降水实况")
            print(f"    ✗ [{model_name}] {','.join(missing)}未找到 "
                  f"(lead={lead_hour}h)")

        return result

    # ------------------------------------------------------------------
    # 多模型: 批量检验 + 保存 + 对比绘图
    # ------------------------------------------------------------------
    def run_multi_model(self, init_date_str, lead_hours, model_names=None):
        """多模型批量检验。对每个模型每个时效计算四个指标, 输出对比CSV和图表。"""
        if model_names is None:
            model_names = self.models
        print("\n" + "=" * 72)
        print("多模型集合预报检验 (CRPS / Spread-Error Ratio / BSS / AROC)")
        print("=" * 72)
        print(f"起报时间: {init_date_str}Z")
        print(f"预报类型: {'集合预报(ENS)' if self.forecast_type == 'ens' else '确定性预报(DF)'}")
        print(f"检验模型: {model_names}")
        print(f"预报时效: {lead_hours}")
        if self.fcstdata_root:
            print(f"生产路径: {self.fcstdata_root}")
        if self.forecast_root:
            print(f"本地路径: {self.forecast_root}")
        print(f"实况路径: {self.cra_root}")
        print("=" * 72)

        all_results = {}  # {model_name: [result, ...]}
        for model in model_names:
            print(f"\n{'─' * 48}")
            print(f"模型: {model}")
            print(f"{'─' * 48}")
            results = []
            for lead in lead_hours:
                r = self.verify_lead_for_model(model, init_date_str, lead)
                if r:
                    results.append(r)
            all_results[model] = results

        if not any(all_results.values()):
            print("\n无可用检验结果")
            return None

        # 保存每个模型CSV + 合并对比CSV
        self._save_multi_model_csv(all_results, init_date_str)
        # 多模型对比图
        self._plot_multi_model(all_results, init_date_str)
        print("\n" + "=" * 72)
        print("多模型检验完成, 结果已保存至:", self.output_dir)
        print("=" * 72)
        return all_results

    def _save_multi_model_csv(self, all_results, init_date_str):
        """保存多模型检验结果: 每个模型一个CSV + 合并对比CSV。"""
        # 1) 每个模型单独的 CSV
        for model, results in all_results.items():
            if not results:
                continue
            df = self._results_to_df(results)
            safe_name = model.replace('-', '_').replace('.', '_')
            csv = self.output_dir / f'ensemble_verification_{safe_name}_{init_date_str}.csv'
            df.to_csv(csv, index=False, encoding='utf-8')
            print(f"保存CSV: {csv}")

        # 2) 合并对比 CSV (所有模型在同一表中)
        all_dfs = []
        for model, results in all_results.items():
            if not results:
                continue
            df = self._results_to_df(results)
            all_dfs.append(df)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            csv = self.output_dir / f'ensemble_verification_multi_model_{init_date_str}.csv'
            combined.to_csv(csv, index=False, encoding='utf-8')
            print(f"保存对比CSV: {csv}")

    def _results_to_df(self, results):
        """将结果列表转为 DataFrame。"""
        rows = []
        for r in results:
            row = {
                'model': r.get('model', ''),
                'lead_hour': r['lead_hour'],
                'valid_time': r['valid_time'],
                'n_members': r.get('n_members', np.nan),
                'n_stations': r.get('n_stations', np.nan),
                'crps': r.get('crps', np.nan),
                'spread': r.get('spread', np.nan),
                'rmse': r.get('rmse', np.nan),
                'spread_error_ratio': r.get('spread_error_ratio', np.nan),
                'bss': r.get('bss', np.nan),
                'aroc': r.get('aroc', np.nan),
            }
            for i, thr in enumerate(PRECIP_THRESHOLDS):
                bss_list = r.get('bss_by_threshold')
                aroc_list = r.get('aroc_by_threshold')
                row[f'bss_{thr}mm'] = bss_list[i] if bss_list else np.nan
                row[f'aroc_{thr}mm'] = aroc_list[i] if aroc_list else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _plot_multi_model(self, all_results, init_date_str):
        """生成多模型对比图: 每个指标一张图, 每条线代表一个模型。"""
        if not HAS_MPL:
            print("matplotlib 未安装, 跳过绘图(CSV已正常输出)。")
            return

        # 将各模型结果转为 DataFrame
        model_dfs = {}
        for model, results in all_results.items():
            if results:
                model_dfs[model] = self._results_to_df(results)
        if not model_dfs:
            return

        metrics = [
            ('crps', 'CRPS (m)', _LABELS['crps'], '#1f77b4'),
            ('spread_error_ratio', 'Spread / RMSE', _LABELS['ssr'], '#ff7f0e'),
            ('spread', 'm', _LABELS['spread_rmse'], '#2ca02c'),
            ('rmse', 'm', _LABELS['rmse'], '#d62728'),
            ('bss', 'BSS', _LABELS['bss'], '#9467bd'),
            ('aroc', 'AROC', _LABELS['aroc'], '#17becf'),
        ]

        for metric_key, ylabel, title, default_color in metrics:
            # 检查是否有可用数据
            has_data = False
            for df in model_dfs.values():
                if metric_key in df.columns and not df[metric_key].isna().all():
                    has_data = True
                    break
            if not has_data:
                continue

            fig, ax = plt.subplots(figsize=(12, 7))
            for model, df in model_dfs.items():
                if metric_key not in df.columns or df[metric_key].isna().all():
                    continue
                style = self.model_styles.get(model, {})
                ax.plot(df['lead_hour'], df[metric_key],
                        marker=style.get('marker', 'o'),
                        color=style.get('color', default_color),
                        linestyle=style.get('linestyle', '-'),
                        linewidth=style.get('linewidth', 2),
                        label=model)

            if metric_key == 'spread_error_ratio':
                ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1,
                           alpha=0.7, label=_LABELS['ratio1'])
            if metric_key == 'bss':
                ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1,
                           alpha=0.5)
            if metric_key == 'aroc':
                ax.set_ylim(0, 1.05)

            ax.set_xlabel(_LABELS['lead_hour'])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {_LABELS['init_label']} {init_date_str}Z",
                         fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
            # X 轴刻度取所有模型时效的并集
            all_hours = sorted(set().union(
                *(set(df['lead_hour'].values) for df in model_dfs.values())))
            ax.set_xticks(all_hours)
            plt.tight_layout()
            out = self.output_dir / f'multi_model_{metric_key}_{init_date_str}.png'
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"保存图: {out}")

    # ------------------------------------------------------------------
    # 单模型批量检验 + 保存 + 绘图 (保留原有接口)
    # ------------------------------------------------------------------
    def run(self, init_date_str, lead_hours):
        print("\n" + "=" * 64)
        print("集合预报检验 (CRPS / Spread-Error Ratio / BSS / AROC)")
        print("=" * 64)
        print(f"起报时间: {init_date_str}Z")
        print(f"预报时效: {lead_hours}")
        print(f"预报路径: {self.forecast_root}")
        print(f"实况路径: {self.cra_root}")
        print("=" * 64)

        results = []
        for lead in lead_hours:
            print("-" * 48)
            r = self.verify_lead(init_date_str, lead)
            if r:
                results.append(r)

        if not results:
            print("\n无可用检验结果")
            return None

        df = self._to_dataframe(results)
        self._save_csv(df, init_date_str)
        self._plot(df, init_date_str)
        print("\n" + "=" * 64)
        print("检验完成，结果已保存至:", self.output_dir)
        print("=" * 64)
        return df

    def _to_dataframe(self, results):
        rows = []
        for r in results:
            row = {
                'lead_hour': r['lead_hour'],
                'valid_time': r['valid_time'],
                'n_stations': r.get('n_stations', np.nan),
                'crps': r.get('crps', np.nan),
                'spread': r.get('spread', np.nan),
                'rmse': r.get('rmse', np.nan),
                'spread_error_ratio': r.get('spread_error_ratio', np.nan),
                'bss': r.get('bss', np.nan),
                'aroc': r.get('aroc', np.nan),
            }
            for i, thr in enumerate(PRECIP_THRESHOLDS):
                bss_list = r.get('bss_by_threshold')
                aroc_list = r.get('aroc_by_threshold')
                row[f'bss_{thr}mm'] = bss_list[i] if bss_list else np.nan
                row[f'aroc_{thr}mm'] = aroc_list[i] if aroc_list else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _save_csv(self, df, init_date_str):
        csv = self.output_dir / f'ensemble_verification_{init_date_str}.csv'
        df.to_csv(csv, index=False, encoding='utf-8')
        print(f"保存CSV: {csv}")

    def _plot(self, df, init_date_str):
        if not HAS_MPL:
            print("matplotlib 未安装，跳过绘图(指标计算与CSV已正常输出)。")
            return
        hours = df['lead_hour'].values

        # 1) CRPS
        if 'crps' in df and not df['crps'].isna().all():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(hours, df['crps'], marker='o', color='#1f77b4', linewidth=2)
            ax.set_xlabel(_LABELS['lead_hour'])
            ax.set_ylabel('CRPS (m)')
            ax.set_title(f"{_LABELS['crps']} — {_LABELS['init_label']} "
                         f"{init_date_str}Z", fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(hours)
            plt.tight_layout()
            out = self.output_dir / f'crps_{init_date_str}.png'
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"保存图: {out}")

        # 2) Spread-Error Ratio
        if 'spread_error_ratio' in df and not df['spread_error_ratio'].isna().all():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(hours, df['spread_error_ratio'], marker='s', color='#ff7f0e',
                    linewidth=2, label='Spread-Error Ratio')
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7,
                       label=_LABELS['ratio1'])
            ax.set_xlabel(_LABELS['lead_hour'])
            ax.set_ylabel('Spread / RMSE')
            ax.set_title(f"{_LABELS['ssr']} — {_LABELS['init_label']} "
                         f"{init_date_str}Z", fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
            ax.set_xticks(hours)
            plt.tight_layout()
            out = self.output_dir / f'spread_error_ratio_{init_date_str}.png'
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"保存图: {out}")

        # 3) Spread & RMSE 对照
        if ('spread' in df and 'rmse' in df and
                not df['spread'].isna().all()):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(hours, df['spread'], marker='^', color='#2ca02c', linewidth=2,
                    label=_LABELS['spread'])
            ax.plot(hours, df['rmse'], marker='v', color='#d62728', linewidth=2,
                    label=_LABELS['rmse_label'])
            ax.set_xlabel(_LABELS['lead_hour'])
            ax.set_ylabel('m')
            ax.set_title(f"{_LABELS['spread_rmse']} — {_LABELS['init_label']} "
                         f"{init_date_str}Z", fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
            ax.set_xticks(hours)
            plt.tight_layout()
            out = self.output_dir / f'spread_rmse_{init_date_str}.png'
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"保存图: {out}")

        # 4) BSS
        if 'bss' in df and not df['bss'].isna().all():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(hours, df['bss'], marker='o', color='#9467bd', linewidth=2,
                    label=_LABELS['bss_avg'])
            for thr in PRECIP_THRESHOLDS:
                col = f'bss_{thr}mm'
                if col in df:
                    ax.plot(hours, df[col], marker='.', linestyle=':',
                            label=f'BSS >= {thr}mm', alpha=0.7)
            ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_xlabel(_LABELS['lead_hour'])
            ax.set_ylabel('BSS')
            ax.set_title(f"{_LABELS['bss_title']} — {_LABELS['init_label']} "
                         f"{init_date_str}Z", fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
            ax.set_xticks(hours)
            plt.tight_layout()
            out = self.output_dir / f'bss_{init_date_str}.png'
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"保存图: {out}")

        # 5) AROC
        if 'aroc' in df and not df['aroc'].isna().all():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(hours, df['aroc'], marker='o', color='#17becf', linewidth=2,
                    label=_LABELS['aroc_avg'])
            for thr in PRECIP_THRESHOLDS:
                col = f'aroc_{thr}mm'
                if col in df:
                    ax.plot(hours, df[col], marker='.', linestyle=':',
                            label=f'AROC >= {thr}mm', alpha=0.7)
            ax.set_xlabel(_LABELS['lead_hour'])
            ax.set_ylabel('AROC')
            ax.set_title(f"{_LABELS['aroc_title']} — {_LABELS['init_label']} "
                         f"{init_date_str}Z", fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1.05)
            ax.legend(loc='best')
            ax.set_xticks(hours)
            plt.tight_layout()
            out = self.output_dir / f'aroc_{init_date_str}.png'
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"保存图: {out}")


def get_config(fcstdata_root=None, cra_root=None, forecast_root=None,
               forecast_type='ens', obs_rain_root=None, accum_hours=6):
    """获取检验配置。

    支持两种路径模式:
      1. 生产模式: fcstdata_root=/gpu/zhaochy/fdp2/FCSTDATA
         → 文件路径: {fcstdata_root}/{MODEL}/{ENS|DF}/{date}/
      2. 本地模式: forecast_root=项目目录 (仅 Fengqing 可用)
         → 文件路径: {forecast_root}/{date}/

    参数:
      fcstdata_root:   生产环境 FCSTDATA 根目录
      cra_root:        实况数据根目录 (CMA-RA, 用于500hPa位势高度检验)
      forecast_root:   本地预报根目录 (Fengqing 测试用)
      forecast_type:   'ens' (集合预报) 或 'df' (确定性预报)
      obs_rain_root:   站点降水实况根目录 (diamond 3 格式 *.000 文件)
                       生产环境: /gpu/zhaochy/fdp2/RDATA/rain
      accum_hours:     降水累积小时数 (默认6)
    """
    return {
        'cra_root': cra_root or '/Users/allen/zhaochy/2026/xmetai_verify/CMA-RA',
        'forecast_root': forecast_root or '/Users/allen/zhaochy/2026/xmetai_verify',
        'fcstdata_root': fcstdata_root or '/gpu/zhaochy/fdp2/FCSTDATA',
        'forecast_type': forecast_type,
        'models': list(MODEL_CONFIGS.keys()),
        'obs_rain_root': obs_rain_root or '/gpu/zhaochy/fdp2/RDATA/rain',
        'accum_hours': accum_hours,
    }


def main():
    parser = argparse.ArgumentParser(
        description='集合预报检验指标算法 (CRPS/Spread-Error Ratio/BSS/AROC) — 支持多模型')
    parser.add_argument('--date', type=str, required=True,
                        help='起报时间 YYYYMMDDHH (例如 2026082000)')
    parser.add_argument('--forecast-hours', type=int, nargs='+',
                        default=list(range(6, 43, 6)),
                        help='预报时效列表 (默认 6-42 每6小时)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=None,
                        help='检验模型列表 (默认全部: Fengqing NJU-Earth PuYun YJ-TianJi W2S)')
    parser.add_argument('--multi', action='store_true', default=True,
                        help='多模型对比模式 (默认开启)')
    parser.add_argument('--single', action='store_true', default=False,
                        help='单模型模式 (仅 Fengqing, 使用本地路径, 兼容旧版)')
    parser.add_argument('--forecast-type', type=str, default='ens',
                        choices=['ens', 'df'],
                        help='预报类型: ens=集合预报, df=确定性预报 (默认 ens)')
    parser.add_argument('--fcstdata-root', type=str, default=None,
                        help='生产环境 FCSTDATA 根目录 (默认 /gpu/zhaochy/fdp2/FCSTDATA)')
    parser.add_argument('--cra-root', type=str, default=None,
                        help='实况数据根目录 (默认本地 CMA-RA)')
    parser.add_argument('--forecast-root', type=str, default=None,
                        help='本地预报根目录 (Fengqing 测试用)')
    parser.add_argument('--obs-rain-root', type=str, default=None,
                        help='站点降水实况根目录 (*.000 文件, '
                             '默认 /gpu/zhaochy/fdp2/RDATA/rain)')
    parser.add_argument('--accum-hours', type=int, default=6,
                        help='降水累积小时数 (默认6, 仅1时次文件可用时设1)')
    parser.add_argument('--output-dir', type=str,
                        default='./ensemble_verification_results',
                        help='输出目录')
    args = parser.parse_args()

    if len(args.date) != 10:
        print("Error: --date 必须为 YYYYMMDDHH 格式")
        return

    config = get_config(
        fcstdata_root=args.fcstdata_root,
        cra_root=args.cra_root,
        forecast_root=args.forecast_root,
        forecast_type=args.forecast_type,
        obs_rain_root=args.obs_rain_root,
        accum_hours=args.accum_hours,
    )

    # 如果指定了模型列表, 覆盖默认
    if args.models:
        # 验证模型名
        invalid = [m for m in args.models if m not in MODEL_CONFIGS]
        if invalid:
            print(f"Error: 未知模型 {invalid}")
            print(f"可用模型: {list(MODEL_CONFIGS.keys())}")
            return
        config['models'] = args.models

    verifier = EnsembleVerifier(config=config, output_dir=args.output_dir)

    if args.single:
        # 兼容旧版: 单模型 (Fengqing 本地路径)
        verifier.run(args.date, args.forecast_hours)
    else:
        # 多模型对比模式
        verifier.run_multi_model(args.date, args.forecast_hours,
                                  model_names=config['models'])


if __name__ == '__main__':
    main()
