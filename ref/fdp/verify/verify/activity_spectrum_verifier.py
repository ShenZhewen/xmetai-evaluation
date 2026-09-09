#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z500 预报活跃度 (Activity Ratio) 与功率谱 (Power Spectrum) 检验

算法来源:
  《第二届人工智能气象预报模型示范计划技术方案-短中期方向》
  - 3.2.1 确定性预报 / 500hPa 位势高度
  - Activity 比值 = std(Forecast Anomaly) / std(Observation Anomaly)
  - Power Spectrum P(k) = Σ_{l=0}^{M-1} [A_t(l,k)² + B_t(l,k)²]  (二维 FFT)
    A_t(l,k) = Σ_{j=1}^{M} Σ_{i=1}^{N} Z_{j,i} cos(2π[l(j-i)/M + k(i-1)/N])
    B_t(l,k) = Σ_{j=1}^{M} Σ_{i=1}^{N} Z_{j,i} sin(2π[l(j-i)/M + k(i-1)/N])
    k=纬向波数, l=经向波数; N=经度格点数, M=纬度格点数

支持模型: Fengqing, NJU-Earth, PuYun, YJ-TianJi, W2S
模型数据路径: /gpu/zhaochy/fdp2/FCSTDATA/{ModelName}/DF/{YYYYMMDD}/
实况路径: /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026/{YYYYMMDD}/
气候态路径: /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR/

数据读取/处理参考 multi_model_verifier_fix.py
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.ioff()

# ======================================================================
# 中文字体检测 (Linux 服务器兼容)
#   扫描系统字体, 找到可用的 CJK 字体后设为默认, 避免中文标题/标签乱码
# ======================================================================
_CHINESE_FONT_CANDIDATES = [
    # macOS
    'PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS',
    'Songti SC', 'STSong', 'Hiragino Sans GB',
    # Linux (生产服务器常见)
    'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP',
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
    'Source Han Sans SC', 'Source Han Sans CN',
    'Source Han Sans CJK SC', 'SimHei', 'Microsoft YaHei',
    'Droid Sans Fallback', 'AR PL UMing CN', 'AR PL UKai CN',
]
_font_names = set()
for _fp in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    try:
        _font_names.add(fm.FontProperties(fname=_fp).get_name())
    except Exception:
        continue
_CN_FONT = None
for _candidate in _CHINESE_FONT_CANDIDATES:
    if _candidate in _font_names:
        _CN_FONT = _candidate
        break
if _CN_FONT:
    plt.rcParams['font.sans-serif'] = [_CN_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[font] 使用中文字体: {_CN_FONT}")
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("[font] 警告: 未找到中文字体, 图表中文将显示为方块。"
          "请安装 CJK 字体: yum install -y wqy-zenhei-fonts "
          "或 apt-get install -y fonts-noto-cjk")

GRAVITY = 9.80665


# ======================================================================
# 模型配置 — 文件命名要素
#   标准: {Model}_{Ver}_GLB_PLEVELS_{ResTag}_6HOR_{DFtag}_{init}_{lead:03d}.nc
#   W2S legacy: W2S_V1_GLB_0P25_HOUR_{init}_{lead:03d}.nc
#   路径: /gpu/zhaochy/fdp2/FCSTDATA/{ModelName}/DF/{YYYYMMDD}/
# ======================================================================
MODEL_CONFIGS = {
    'Fengqing': {
        'version': '1.0',
        'resolution_tag': 'OP25',
        'df_tag': 'FCST',
    },
    'NJU-Earth': {
        'version': 'v1',
        'resolution_tag': '0P25',
        'df_tag': 'FCST',
    },
    'PuYun': {
        'version': 'v1',
        'resolution_tag': 'OP25',
        'df_tag': 'FCST',
    },
    'YJ-TianJi': {
        'version': 'v0.1',
        'resolution_tag': '0P25',
        'df_tag': 'FCST',
    },
    'W2S': {
        'version': 'V1',
        'resolution_tag': '0P25',
        'df_tag': 'FCST',
        # 确定性 (member=1): W2S_V1_GLB_0P25_HOUR_{init}_{lead:03d}.nc
        # 集合 (member>1):   W2S_V1_GLB_0P25_ENS_HOUR_{init}_{lead:03d}.nc
        'alt_naming': 'W2S_V1_GLB_0P25_HOUR_{init}_{lead:03d}.nc',
        'alt_naming_ens': 'W2S_V1_GLB_0P25_ENS_HOUR_{init}_{lead:03d}.nc',
        'channel_based': True,
    },
}


# ======================================================================
# 变量配置 — z500
# ======================================================================
VAR_CONFIG = {
    'var_candidates': ['Z500', 'z500', 'Z', 'gh', 'GH', 'H500'],
    'level': 500,
    'truth_keys': {'shortName': 'gh', 'typeOfLevel': 'isobaricInhPa', 'level': 500},
    'climate_var': 'gh',
    'full_name': '500hPa Geopotential Height',
    'units': 'm',
    # W2S channel name for z500
    'w2s_channel': 'z500',
}


# ======================================================================
# 工具函数 (ported from multi_model_verifier_fix.py)
# ======================================================================
def detect_variable_name(ds, candidates):
    """按候选名列表查找变量 (大小写不敏感回退)。"""
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower_map = {v.lower(): v for v in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def normalize_unit_string(unit_str):
    if unit_str is None:
        return ''
    u = unit_str.strip().lower().replace(' ', '').replace('**', '^')
    equivalences = {
        'm^2/s^2': 'm2/s2', 'm^2s^-2': 'm2/s2', 'm2s-2': 'm2/s2',
        'meter': 'm', 'meters': 'm', 'gpm': 'm',
    }
    return equivalences.get(u, u)


def convert_z500(data, source_unit):
    """将 Z500 转换为位势高度 (m)。"""
    su = normalize_unit_string(source_unit)
    if su in ('m', 'gpm', 'meter', 'meters'):
        return data
    if su in ('m2/s2',):
        return data / GRAVITY
    mean_val = float(np.nanmean(data))
    if mean_val > 10000:
        print(f"      ⚠ Z500 单位未知 ({source_unit!r}), 均值 {mean_val:.1f} > 10000, "
              f"按 m²/s² 处理, 除以 g")
        return data / GRAVITY
    return data


class ActivitySpectrumVerifier:
    """Z500 预报活跃度与功率谱检验器"""

    def __init__(self, config, output_dir='./activity_spectrum_results'):
        self.cra_root = Path(config.get('cra_root',
              '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026'))
        self.cli_root = Path(config.get('cli_root',
              '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR'))
        self.model_data_root = Path(config.get('model_data_root',
              '/gpu/zhaochy/fdp2/FCSTDATA'))
        self.models = config.get('models', list(MODEL_CONFIGS.keys()))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_styles = {
            'Fengqing':  {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-',  'linewidth': 2},
            'NJU-Earth': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'linewidth': 2},
            'PuYun':     {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'linewidth': 2},
            'YJ-TianJi': {'color': '#d62728', 'marker': 'D', 'linestyle': ':',  'linewidth': 2},
            'W2S':       {'color': '#9467bd', 'marker': 'v', 'linestyle': '-',  'linewidth': 2},
        }
        # 观测/气候态样式
        self.obs_style = {'color': '#000000', 'marker': '*', 'linestyle': '-', 'linewidth': 2.5}

    # ==================================================================
    # 实况 (CMA-RA) 数据读取 — GRIB2
    # ==================================================================
    def find_cra_file(self, date_str, data_type='pressure'):
        """查找 CRA40 实况文件 — date_str = YYYYMMDDHH。"""
        date_only = date_str[:8]
        date_dir = self.cra_root / date_only
        if not date_dir.exists():
            return None
        if data_type == 'pressure':
            filename = f"ART_ATM_GLB_0P25_6HOR_ANAL_{date_str}.grib2"
            cand = date_dir / filename
            if cand.exists():
                return cand
            files = list(date_dir.glob(f"*{date_str}*.grib2"))
            return files[0] if files else None
        return None

    def find_climate_file(self, month, day, hour='00'):
        """查找气候态文件 (按月日时匹配)。"""
        mmdd = f"{int(month):02d}{int(day):02d}"
        filename = f"ART_ATM_GLB_0P25_CLI_ANAL_{mmdd}{hour}.grib2"
        cand = self.cli_root / filename
        if cand.exists():
            return cand
        files = list(self.cli_root.glob(f"*{mmdd}*.grib2"))
        return files[0] if files else None

    def read_climate_data(self, filepath, var_name='gh', level=500):
        """读取气候态 Z500 (gh@500hPa, 单位 m)。"""
        if not filepath or not filepath.exists():
            return None
        try:
            keys = {'shortName': var_name, 'typeOfLevel': 'isobaricInhPa', 'level': level}
            ds = xr.open_dataset(str(filepath), engine='cfgrib',
                                 backend_kwargs={'filter_by_keys': keys})
            data_vars = [v for v in ds.data_vars if v not in
                         ['latitude', 'longitude', 'valid_time', 'step', 'time']]
            if data_vars:
                data = ds[data_vars[0]]
                if 'latitude' in data.coords and 'longitude' in data.coords:
                    data = data.rename({'latitude': 'lat', 'longitude': 'lon'})
                # 纬度降序
                lat_values = data.lat.values
                if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                    data = data.isel(lat=slice(None, None, -1))
                return data
        except Exception as e:
            print(f"      读取气候态数据失败: {e}")
        return None

    def read_truth_data(self, filepath, var_cfg):
        """读取实况 Z500 (gh@500hPa, 单位 m)。"""
        if not filepath or not filepath.exists():
            return None
        keys = var_cfg['truth_keys']
        level = var_cfg.get('level')
        try:
            if level is not None and 'level' not in keys:
                keys = {**keys, 'level': level}
            ds = xr.open_dataset(str(filepath), engine='cfgrib',
                                 backend_kwargs={'filter_by_keys': keys})
            data_vars = [v for v in ds.data_vars if v not in
                         ['latitude', 'longitude', 'valid_time', 'step', 'time']]
            if data_vars:
                data = ds[data_vars[0]]
                if 'latitude' in data.coords and 'longitude' in data.coords:
                    data = data.rename({'latitude': 'lat', 'longitude': 'lon'})
                lat_values = data.lat.values
                if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                    reversed_indices = np.arange(len(data.lat))[::-1]
                    data = data.isel(lat=reversed_indices)
                return data
        except Exception as e:
            print(f"      读取实况数据失败: {e}")
        return None

    # ==================================================================
    # 模型预报数据读取 — NetCDF (标准 + W2S channel 格式)
    # ==================================================================
    def find_model_file(self, model_name, date_str, forecast_hour):
        """查找模型预报文件 (z500 = PLEVELS 类型)。

        路径: {model_data_root}/{ModelName}/DF/{YYYYMMDD}/{filename}
        """
        if model_name not in MODEL_CONFIGS:
            return None
        cfg = MODEL_CONFIGS[model_name]
        date_only = date_str[:8]
        date_dir = self.model_data_root / model_name / 'DF' / date_only
        if not date_dir.exists():
            return None

        lead_str = f"{forecast_hour:03d}"

        # 1) W2S legacy 命名 (单文件包含所有变量, 无 PLEVELS/SURFACE 区分)
        if cfg.get('channel_based') and 'alt_naming' in cfg:
            # 确定性 (member=1): 无 ENS_ 前缀
            alt_fname = cfg['alt_naming'].format(init=date_str, lead=forecast_hour)
            alt_cand = date_dir / alt_fname
            if alt_cand.exists():
                return alt_cand
            # 集合 (member>1): 有 ENS_ 前缀
            if 'alt_naming_ens' in cfg:
                ens_fname = cfg['alt_naming_ens'].format(
                    init=date_str, lead=forecast_hour)
                ens_cand = date_dir / ens_fname
                if ens_cand.exists():
                    return ens_cand
            # 宽松 glob
            alt_files = list(date_dir.glob(
                f"*{model_name}*{date_str}_{lead_str}*.nc"))
            if alt_files:
                return alt_files[0]

        # 2) 标准命名 (PLEVELS 文件)
        type_tag = 'PLEVELS'
        fname = (f"{model_name}_{cfg['version']}_GLB_{type_tag}_"
                 f"{cfg['resolution_tag']}_6HOR_{cfg['df_tag']}_"
                 f"{date_str}_{lead_str}.nc")
        cand = date_dir / fname
        if cand.exists():
            return cand

        # 3) 宽松 glob
        pat = f"*{model_name}*{type_tag}*{date_str}_{lead_str}*.nc"
        files = list(date_dir.glob(pat))
        if files:
            return files[0]

        # 4) 更宽松
        pat2 = f"*{model_name}*{date_str}_{lead_str}*.nc"
        files = list(date_dir.glob(pat2))
        if files:
            return files[0]

        return None

    def read_model_data(self, filepath, model_name, debug=False):
        """读取模型 Z500 数据, 返回位势高度 (m), 纬度降序。

        支持标准 NetCDF (变量名自动检测) 和 W2S channel 格式。
        """
        if filepath is None:
            return None
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        try:
            ds = xr.open_dataset(str(filepath))

            # --- W2S channel-based 格式 ---
            if 'channel' in ds and 'data' in ds.data_vars:
                data = self._read_w2s_channel(ds, model_name, debug)
                ds.close()
                return data

            # --- 标准 NetCDF ---
            var_name = detect_variable_name(ds, VAR_CONFIG['var_candidates'])
            if var_name is None:
                print(f"      ✗ [{model_name}] Z500 未找到. 候选: {VAR_CONFIG['var_candidates']}, "
                      f"可用: {list(ds.data_vars)}")
                ds.close()
                return None

            data = ds[var_name]
            print(f"      [{model_name}] Z500: var='{var_name}', shape={data.shape}")

            # squeeze 单例维度
            for d in ('level', 'time', 'dtime', 'isobaricInhPa', 'lev'):
                if d in data.dims and data.sizes[d] == 1:
                    data = data.squeeze(dim=d, drop=True)
            for d in ('member', 'ensemble', 'ens', 'number', 'realization'):
                if d in data.dims and data.sizes[d] == 1:
                    data = data.squeeze(dim=d, drop=True)

            if 'latitude' in data.coords:
                data = data.rename({'latitude': 'lat', 'longitude': 'lon'})

            # 单位转换
            source_unit = data.attrs.get('units', None)
            data = convert_z500(data, source_unit)

            if debug:
                print(f"      [DEBUG] Z500 转换后 — "
                      f"min={float(data.values.min()):.2f}, "
                      f"max={float(data.values.max()):.2f}, "
                      f"mean={float(data.values.mean()):.2f}")

            # 纬度降序
            if 'lat' in data.coords:
                lat_values = data.lat.values
                if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                    print(f"      [{model_name}] 纬度升序→降序")
                    data = data.isel(lat=slice(None, None, -1))

            ds.close()
            return data

        except Exception as e:
            print(f"      [{model_name}] 读取 Z500 失败: {e}")
            return None

    def _read_w2s_channel(self, ds, model_name, debug=False):
        """读取 W2S channel-based NetCDF (z500 存储为 channel)。"""
        channel_names = ds['channel'].values
        channel_list = []
        for name in channel_names:
            if isinstance(name, bytes):
                channel_list.append(name.decode('utf-8'))
            else:
                channel_list.append(str(name))

        target_channel = VAR_CONFIG['w2s_channel']  # 'z500'
        if target_channel not in channel_list:
            print(f"      ✗ [{model_name}] channel '{target_channel}' 未找到. "
                  f"可用: {channel_list}")
            return None

        idx = channel_list.index(target_channel)
        data = ds['data'].isel(channel=idx, time=0)

        # 压缩 member 维度:
        #   确定性预报 (member=1): (1, lat, lon) → (lat, lon)
        #   集合预报 (member>1):  取 ensemble mean → (lat, lon)
        if 'member' in data.dims:
            if data.sizes['member'] == 1:
                data = data.squeeze(dim='member', drop=True)
            else:
                data = data.mean(dim='member')

        print(f"      [{model_name}] W2S channel: '{target_channel}', shape={data.shape}")

        if 'latitude' in data.coords:
            data = data.rename({'latitude': 'lat', 'longitude': 'lon'})

        # W2S z500 单位转换: 可能是 m²/s² (位势) 或 m (位势高度)
        # 使用统一的 convert_z500 函数处理所有情况 (含未知单位时按数值推断)
        source_unit = data.attrs.get('units', None)
        mean_before = float(np.nanmean(data.values))
        data = convert_z500(data, source_unit)
        mean_after = float(np.nanmean(data.values))
        if abs(mean_before - mean_after) > 1.0:
            print(f"      [{model_name}] W2S z500: 单位转换 "
                  f"({source_unit!r}) 均值 {mean_before:.1f} → {mean_after:.1f} (m)")

        if debug:
            print(f"      [DEBUG] W2S z500 转换后 — "
                  f"min={float(data.values.min()):.2f}, "
                  f"max={float(data.values.max()):.2f}, "
                  f"mean={float(data.values.mean()):.2f}")

        # 纬度降序
        if 'lat' in data.coords:
            lat_values = data.lat.values
            if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                print(f"      [{model_name}] 纬度升序→降序")
                data = data.isel(lat=slice(None, None, -1))

        return data

    # ==================================================================
    # 网格对齐
    # ==================================================================
    def interpolate_to_common_grid(self, source_data, target_lat, target_lon):
        """将 source_data 插值到 target 网格。"""
        try:
            if 'lat' not in source_data.coords:
                if 'latitude' in source_data.coords:
                    source_data = source_data.rename(
                        {'latitude': 'lat', 'longitude': 'lon'})
                else:
                    return None
            source_lat = source_data.lat.values
            if len(source_lat) > 1 and len(target_lat) > 1:
                source_desc = source_lat[0] > source_lat[-1]
                target_desc = target_lat[0] > target_lat[-1]
                if source_desc != target_desc:
                    source_data = source_data.isel(lat=slice(None, None, -1))
            return source_data.interp(
                lat=target_lat, lon=target_lon,
                kwargs={"fill_value": "extrapolate"})
        except Exception as e:
            print(f"      网格插值失败: {e}")
            return None

    # ==================================================================
    # 指标 1: Activity 比值
    #   Activity Ratio = std_weighted(Forecast Anomaly) / std_weighted(Obs Anomaly)
    #   Anomaly = Field - Climate
    #   std_weighted = sqrt(Σ w_i (x_i - w_mean)² / Σ w_i)
    #   w_i = cos(lat_i)
    # ==================================================================
    def calculate_activity_ratio(self, forecast_data, truth_data, climate_data):
        """计算纬度加权的活跃度比值。

        公式:
          numerator   = sqrt(Σ w ((F-C) - w_mean(F-C))² / Σ w)
          denominator = sqrt(Σ w ((O-C) - w_mean(O-C))² / Σ w)
          Activity Ratio = numerator / denominator
        """
        try:
            if (forecast_data.shape != truth_data.shape or
                    forecast_data.shape != climate_data.shape):
                print(f"      Shape 不匹配: fc={forecast_data.shape}, "
                      f"obs={truth_data.shape}, clim={climate_data.shape}")
                return None

            lat = forecast_data.lat.values
            w = np.cos(np.deg2rad(lat))
            if len(forecast_data.shape) == 2:
                w_2d = np.broadcast_to(w[:, np.newaxis], forecast_data.shape)
            else:
                w_2d = w

            fc_anom = forecast_data.values - climate_data.values
            obs_anom = truth_data.values - climate_data.values

            w_sum = np.nansum(w_2d * ~np.isnan(fc_anom))
            if w_sum == 0:
                return None

            fc_wmean = np.nansum(w_2d * fc_anom) / w_sum
            obs_wmean = np.nansum(w_2d * obs_anom) / w_sum

            fc_std = np.sqrt(np.nansum(w_2d * (fc_anom - fc_wmean) ** 2) / w_sum)
            obs_std = np.sqrt(np.nansum(w_2d * (obs_anom - obs_wmean) ** 2) / w_sum)

            if obs_std == 0:
                return None

            activity_ratio = float(fc_std / obs_std)
            return {
                'activity_ratio': activity_ratio,
                'fc_activity': float(fc_std),
                'obs_activity': float(obs_std),
            }
        except Exception as e:
            print(f"      Activity 比值计算失败: {e}")
            return None

    # ==================================================================
    # 指标 2: 功率谱 (Power Spectrum) — 二维傅里叶变换法
    #   公式来源: 《功率谱计算公式.docx》
    #   P(k) = Σ_{l=0}^{M-1} [A_t(l,k)² + B_t(l,k)²]
    #   A_t(l,k) = Σ_{j=1}^{M} Σ_{i=1}^{N} Z_{j,i} cos(2π[l(j-i)/M + k(i-1)/N])
    #   B_t(l,k) = Σ_{j=1}^{M} Σ_{i=1}^{N} Z_{j,i} sin(2π[l(j-i)/M + k(i-1)/N])
    #   k=纬向波数, l=经向波数; N=经度格点数, M=纬度格点数
    #
    #   高效实现 (FFT):
    #     令 G(l,i) = IFFT_j(Z)(l,i) · exp(-2πi·l·i/M)
    #         F(l,k) = IFFT_i(G)(l,k) = (1/(MN)) · [A_t(l,k) + i·B_t(l,k)]
    #     则 |A_t|² + |B_t|² = (MN)² · |F(l,k)|²
    #         P(k) = (MN)² · Σ_{l=0}^{M-1} |F(l,k)|²
    #     实数输入共轭对称: P(k)=P(N-k), 取单边谱 k=0..N//2
    # ==================================================================
    def calculate_power_spectrum(self, data, max_wavenumber=None):
        """计算纬向波数功率谱 P(k) — 二维 FFT 法。

        公式 (《功率谱计算公式.docx》):
          P(k) = Σ_{l=0}^{M-1} [A_t(l,k)² + B_t(l,k)²]
          A_t(l,k) = Σ_{j=1}^{M} Σ_{i=1}^{N} Z_{j,i} cos(2π[l(j-i)/M + k(i-1)/N])
          B_t(l,k) = Σ_{j=1}^{M} Σ_{i=1}^{N} Z_{j,i} sin(2π[l(j-i)/M + k(i-1)/N])

        参数:
          data: xarray DataArray (lat, lon), 纬度降序或升序均可
          max_wavenumber: 返回的最大波数 (None=自动取 N//2)

        返回:
          wavenumbers: np.array, [0, 1, 2, ..., K]
          power: np.array, 对应 P(k)
        """
        try:
            if 'lat' not in data.coords:
                return None, None

            values = data.values
            M, N = values.shape  # M=纬度方向格点数, N=经度方向格点数
            n_freq = N // 2 + 1  # 单边谱长度

            # 打印数据范围以辅助诊断量级 (Z500 应在 ~5000-6000 m)
            data_mean = float(np.nanmean(values))
            data_std = float(np.nanstd(values))
            print(f"      [功率谱输入] shape={values.shape}, M(纬度)={M}, N(经度)={N}, "
                  f"mean={data_mean:.1f}, std={data_std:.1f}")
            if data_mean > 10000:
                print(f"      ⚠ 均值 {data_mean:.1f} > 10000, 疑似未转换单位 "
                      f"(m²/s² 而非 m), 功率谱量级将偏大约 g²≈{GRAVITY**2:.1f} 倍")

            if max_wavenumber is None:
                max_k = n_freq - 1
            else:
                max_k = min(max_wavenumber, n_freq - 1)

            # 处理 NaN (用全场均值填充)
            if np.any(np.isnan(values)):
                values = np.where(np.isnan(values), np.nanmean(values), values)

            # --- 高效计算 A_t + i·B_t ---
            # 公式: A_t + i·B_t = Σ_j Σ_i Z_{j,i} · exp(2πi·[l(j-i)/M + k·i/N])
            # (0-indexed: j=0..M-1, i=0..N-1, 故 (i-1)→i, j-i 不变)

            # Step 1: IFFT 沿 j (纬度, axis=0)
            #   IFFT_j(Z)[l,i] = (1/M) Σ_j Z[j,i]·exp(2πi·l·j/M)
            Z_ifft_j = np.fft.ifft(values, axis=0)  # (M, N)

            # Step 2: 相位调制 G(l,i) = IFFT_j(Z)[l,i] · exp(-2πi·l·i/M)
            l_idx = np.arange(M)[:, None]   # (M, 1)
            i_idx = np.arange(N)[None, :]   # (1, N)
            phase_mod = np.exp(-2j * np.pi * l_idx * i_idx / M)
            G = Z_ifft_j * phase_mod        # (M, N)

            # Step 3: IFFT 沿 i (经度, axis=1)
            #   F(l,k) = (1/N) Σ_i G(l,i)·exp(2πi·k·i/N)
            #          = (1/(MN)) · [A_t(l,k) + i·B_t(l,k)]
            F = np.fft.ifft(G, axis=1)      # (M, N)

            # Step 4: |A_t|² + |B_t|² = (MN)² · |F(l,k)|²
            P_lk = (M * N) ** 2 * np.abs(F) ** 2  # (M, N)

            # Step 5: P(k) = Σ_{l=0}^{M-1} [(A_t)² + (B_t)²]
            P_k_full = P_lk.sum(axis=0)     # (N,)

            # 单边谱合并: 实数输入共轭对称 P(k) = P(N-k)
            # 合并正负频率: 1 ≤ k < N/2 时 P_one(k) = P_full(k) + P_full(N-k) = 2·P_full(k)
            P_k = np.zeros(max_k + 1)
            for k in range(max_k + 1):
                if k == 0 or (N % 2 == 0 and k == N // 2):
                    P_k[k] = P_k_full[k]
                else:
                    P_k[k] = P_k_full[k] + P_k_full[N - k]

            wavenumbers = np.arange(max_k + 1)
            return wavenumbers, P_k
        except Exception as e:
            print(f"      功率谱计算失败: {e}")
            return None, None

    # ==================================================================
    # 检验流程
    # ==================================================================
    def verify_single_forecast(self, model_name, date_str, forecast_hour,
                               debug=False):
        """检验单个模型/时效的 Z500 activity 和功率谱。"""
        print(f"  检验 {model_name}: {forecast_hour:03d}h")

        # 读取模型数据
        model_file = self.find_model_file(model_name, date_str, forecast_hour)
        if not model_file:
            print(f"    ✗✗✗✗ 模型文件未找到")
            return None
        print(f"    ✓ 模型文件: {model_file.name}")

        model_data = self.read_model_data(model_file, model_name, debug=debug)
        if model_data is None:
            print(f"    ✗✗✗✗ 无法读取模型数据")
            return None

        # 计算实况时间
        start_dt = datetime.strptime(date_str, "%Y%m%d%H")
        truth_dt = start_dt + timedelta(hours=forecast_hour)
        truth_date = truth_dt.strftime("%Y%m%d%H")
        print(f"    ✓ 实况时间: {truth_date}")

        # 读取实况数据
        truth_file = self.find_cra_file(truth_date, data_type='pressure')
        if not truth_file:
            print(f"    ✗✗✗✗ 实况文件未找到")
            return None
        print(f"    ✓ 实况文件: {truth_file.name}")

        truth_data = self.read_truth_data(truth_file, VAR_CONFIG)
        if truth_data is None:
            print(f"    ✗✗✗✗ 无法读取实况数据")
            return None

        # 读取气候态数据
        climate_file = self.find_climate_file(
            truth_dt.month, truth_dt.day, f"{truth_dt.hour:02d}")
        if not climate_file:
            print(f"    ✗✗✗✗ 气候态文件未找到")
            return None
        print(f"    ✓ 气候态文件: {climate_file.name}")

        climate_data = self.read_climate_data(
            climate_file, VAR_CONFIG['climate_var'], VAR_CONFIG['level'])
        if climate_data is None:
            print(f"    ✗✗✗✗ 无法读取气候态数据")
            return None

        # 网格对齐: 实况、气候态 → 模型网格
        if model_data.shape != truth_data.shape:
            print(f"      网格不匹配: model={model_data.shape}, "
                  f"truth={truth_data.shape}, 插值实况...")
            truth_data = self.interpolate_to_common_grid(
                truth_data, model_data.lat.values, model_data.lon.values)
            if truth_data is None:
                print(f"      ✗✗✗✗ 实况插值失败")
                return None

        if model_data.shape != climate_data.shape:
            print(f"      网格不匹配: model={model_data.shape}, "
                  f"clim={climate_data.shape}, 插值气候态...")
            climate_data = self.interpolate_to_common_grid(
                climate_data, model_data.lat.values, model_data.lon.values)
            if climate_data is None:
                print(f"      ✗✗✗✗ 气候态插值失败")
                return None

        # --- 计算 Activity 比值 ---
        activity_result = self.calculate_activity_ratio(
            model_data, truth_data, climate_data)

        if activity_result is not None:
            print(f"    ✓ Activity 比值: {activity_result['activity_ratio']:.4f}")
            print(f"      (预报活跃度={activity_result['fc_activity']:.2f}, "
                  f"观测活跃度={activity_result['obs_activity']:.2f})")
        else:
            print(f"    ✗✗✗✗ Activity 计算失败")

        # --- 计算功率谱 ---
        max_k = 30  # 通常关注 0-30 波 (行星波+天气尺度波)
        fc_wavenumbers, fc_power = self.calculate_power_spectrum(
            model_data, max_wavenumber=max_k)
        obs_wavenumbers, obs_power = self.calculate_power_spectrum(
            truth_data, max_wavenumber=max_k)

        if fc_power is not None and obs_power is not None:
            print(f"    ✓ 功率谱: {len(fc_power)} 个波数 (0-{max_k})")
        else:
            print(f"    ✗✗✗✗ 功率谱计算失败")

        return {
            'activity': activity_result,
            'spectrum': {
                'wavenumbers': fc_wavenumbers,
                'fc_power': fc_power,
                'obs_power': obs_power,
            } if fc_power is not None and obs_power is not None else None,
            'forecast_hour': forecast_hour,
            'truth_date': truth_date,
        }

    def run_verification(self, date_str, model_names=None,
                         forecast_hours=None):
        """批量检验多个模型/时效。"""
        if forecast_hours is None:
            forecast_hours = list(range(6, 361, 6))
        if model_names is None:
            model_names = self.models

        display_date = f"{date_str}Z"
        print("\n" + "=" * 60)
        print("Z500 预报活跃度与功率谱检验")
        print("=" * 60)
        print(f"初始时间: {display_date}")
        print(f"模型: {model_names}")
        print(f"检验时效: {forecast_hours}")
        print(f"模型数据根目录: {self.model_data_root}")
        print(f"实况根目录: {self.cra_root}")
        print(f"气候态根目录: {self.cli_root}")
        print("=" * 60)

        # results 结构: {model: {hours:[], activity_ratio:[], fc_activity:[],
        #                        obs_activity:[], spectra:{hour: {wn, fc, obs}}}}
        all_results = {}
        for model_name in model_names:
            all_results[model_name] = {
                'hours': [],
                'activity_ratio': [],
                'fc_activity': [],
                'obs_activity': [],
                'spectra': {},  # {hour: {'wavenumbers':, 'fc_power':, 'obs_power':}}
            }

        for model_name in model_names:
            print(f"\n模型: {model_name}")
            print("-" * 40)
            valid_count = 0

            for f_hour in forecast_hours:
                result = self.verify_single_forecast(
                    model_name, date_str, f_hour)

                if result is None:
                    print(f"  {f_hour:03d}h: ✗ 失败")
                    continue

                valid_count += 1
                all_results[model_name]['hours'].append(f_hour)

                if result['activity'] is not None:
                    all_results[model_name]['activity_ratio'].append(
                        result['activity']['activity_ratio'])
                    all_results[model_name]['fc_activity'].append(
                        result['activity']['fc_activity'])
                    all_results[model_name]['obs_activity'].append(
                        result['activity']['obs_activity'])
                else:
                    all_results[model_name]['activity_ratio'].append(np.nan)
                    all_results[model_name]['fc_activity'].append(np.nan)
                    all_results[model_name]['obs_activity'].append(np.nan)

                if result['spectrum'] is not None:
                    all_results[model_name]['spectra'][f_hour] = result['spectrum']

            print(f"  共计: {valid_count}/{len(forecast_hours)} 个时效检验成功")

        return all_results, display_date

    # ==================================================================
    # 结果输出 (CSV)
    # ==================================================================
    def save_results(self, all_results, date_str):
        """保存 Activity 比值到 CSV; 功率谱单独保存。"""
        # --- Activity CSV ---
        activity_dfs = []
        for model_name, model_data in all_results.items():
            if not model_data['hours']:
                continue
            df = pd.DataFrame({
                'forecast_hour': model_data['hours'],
                'activity_ratio': model_data['activity_ratio'],
                'fc_activity': model_data['fc_activity'],
                'obs_activity': model_data['obs_activity'],
            })
            df['model'] = model_name
            df['variable'] = 'z500'
            df['variable_name'] = VAR_CONFIG['full_name']
            df['units'] = VAR_CONFIG['units']
            activity_dfs.append(df)

        if activity_dfs:
            df_all = pd.concat(activity_dfs, ignore_index=True)
            csv_path = self.output_dir / f'activity_ratio_{date_str}.csv'
            df_all.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"已保存: {csv_path}")

        # --- 功率谱 CSV (每个时效一行, 列为波数) ---
        spectrum_dfs = []
        for model_name, model_data in all_results.items():
            for f_hour, spec in model_data['spectra'].items():
                if spec is None:
                    continue
                wn = spec['wavenumbers']
                fc_p = spec['fc_power']
                obs_p = spec['obs_power']
                row_fc = {'model': model_name, 'forecast_hour': f_hour,
                          'type': 'forecast'}
                row_obs = {'model': model_name, 'forecast_hour': f_hour,
                           'type': 'observation'}
                for k, (fp, op) in enumerate(zip(fc_p, obs_p)):
                    row_fc[f'P{k}'] = fp
                    row_obs[f'P{k}'] = op
                spectrum_dfs.append(row_fc)
                spectrum_dfs.append(row_obs)

        if spectrum_dfs:
            df_spec = pd.DataFrame(spectrum_dfs)
            csv_spec = self.output_dir / f'power_spectrum_{date_str}.csv'
            df_spec.to_csv(csv_spec, index=False, encoding='utf-8')
            print(f"已保存: {csv_spec}")

    # ==================================================================
    # 可视化
    # ==================================================================
    def plot_results(self, all_results, date_str, display_date,
                     spectrum_hours=None):
        """绘制 Activity 比值时序图与功率谱对比图。

        spectrum_hours: 指定哪些时效绘制功率谱对比图 (默认取 72h 和 360h)
        """
        if not all_results:
            print("无数据可绘图")
            return

        # --- Activity 比值时序图 ---
        self._plot_activity_ratio(all_results, date_str, display_date)

        # --- 功率谱对比图 ---
        if spectrum_hours is None:
            # 自动选取: 72h, 168h, 240h, 360h (如存在)
            candidate_hours = [72, 168, 240, 360]
            spectrum_hours = []
            for h in candidate_hours:
                for model_data in all_results.values():
                    if h in model_data['spectra']:
                        spectrum_hours.append(h)
                        break
            if not spectrum_hours:
                # 取第一个有谱数据的时效
                for model_data in all_results.values():
                    if model_data['spectra']:
                        spectrum_hours = [min(model_data['spectra'].keys())]
                        break

        for f_hour in spectrum_hours:
            self._plot_power_spectrum(all_results, f_hour, date_str, display_date)

    def _plot_activity_ratio(self, all_results, date_str, display_date):
        """Activity 比值 vs 预报时效。"""
        fig, ax = plt.subplots(figsize=(12, 6))
        has_data = False

        for model_name, model_data in all_results.items():
            hours = model_data['hours']
            ratios = model_data['activity_ratio']
            if not hours or not ratios:
                continue
            # 过滤 NaN
            valid = [(h, r) for h, r in zip(hours, ratios)
                     if not np.isnan(r)]
            if not valid:
                continue
            vh, vr = zip(*valid)
            vh = np.array(vh)
            vr = np.array(vr)
            si = np.argsort(vh)
            has_data = True
            style = self.model_styles.get(model_name, {})
            ax.plot(vh[si], vr[si],
                    marker=style.get('marker', 'o'),
                    linestyle=style.get('linestyle', '-'),
                    linewidth=style.get('linewidth', 2),
                    markersize=6,
                    color=style.get('color', '#1f77b4'),
                    label=model_name)

        if has_data:
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='Activity=1 (理想)')
            ax.set_xlabel('预报时效 (小时)', fontsize=12)
            ax.set_ylabel('Activity 比值 (预报/观测)', fontsize=12)
            ax.set_title(f'Z500 预报活跃度比值 - {display_date}',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='best')
            plt.tight_layout()
            pf = self.output_dir / f'activity_ratio_{date_str}.png'
            plt.savefig(pf, dpi=300, bbox_inches='tight')
            print(f"已保存: {pf}")
        else:
            print("无 Activity 数据可绘图")
        plt.close(fig)

        # --- 预报/观测活跃度对比图 ---
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        has_data2 = False
        # 观测活跃度 (所有模型应相同, 取第一个有效值)
        obs_hours = []
        obs_values = []
        for model_name, model_data in all_results.items():
            hours = model_data['hours']
            obs_act = model_data['obs_activity']
            if not hours or not obs_act:
                continue
            valid = [(h, v) for h, v in zip(hours, obs_act)
                     if not np.isnan(v)]
            if not valid:
                continue
            vh, vv = zip(*valid)
            obs_hours = np.array(vh)
            obs_values = np.array(vv)
            si = np.argsort(obs_hours)
            obs_hours = obs_hours[si]
            obs_values = obs_values[si]
            has_data2 = True
            break  # 观测活跃度对所有模型相同

        if has_data2:
            ax2.plot(obs_hours, obs_values, color='black', linewidth=2.5,
                     marker='*', markersize=8, label='观测活跃度')

        for model_name, model_data in all_results.items():
            hours = model_data['hours']
            fc_act = model_data['fc_activity']
            if not hours or not fc_act:
                continue
            valid = [(h, v) for h, v in zip(hours, fc_act)
                     if not np.isnan(v)]
            if not valid:
                continue
            vh, vv = zip(*valid)
            vh = np.array(vh)
            vv = np.array(vv)
            si = np.argsort(vh)
            has_data2 = True
            style = self.model_styles.get(model_name, {})
            ax2.plot(vh[si], vv[si],
                     marker=style.get('marker', 'o'),
                     linestyle=style.get('linestyle', '--'),
                     linewidth=style.get('linewidth', 2),
                     markersize=6,
                     color=style.get('color', '#1f77b4'),
                     label=f'{model_name} 预报')

        if has_data2:
            ax2.set_xlabel('预报时效 (小时)', fontsize=12)
            ax2.set_ylabel('活跃度 (m)', fontsize=12)
            ax2.set_title(f'Z500 预报与观测活跃度 - {display_date}',
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=10, loc='best')
            plt.tight_layout()
            pf2 = self.output_dir / f'activity_fc_obs_{date_str}.png'
            plt.savefig(pf2, dpi=300, bbox_inches='tight')
            print(f"已保存: {pf2}")
        plt.close(fig2)

    def _plot_power_spectrum(self, all_results, forecast_hour, date_str,
                             display_date):
        """功率谱 P(k) vs 波数 k 对比图 (预报 vs 观测)。"""
        fig, ax = plt.subplots(figsize=(12, 6))
        has_data = False

        # 观测功率谱 (所有模型相同, 取第一个有效值)
        obs_wn = None
        obs_power = None
        for model_name, model_data in all_results.items():
            spec = model_data['spectra'].get(forecast_hour)
            if spec is not None:
                obs_wn = spec['wavenumbers']
                obs_power = spec['obs_power']
                break

        if obs_wn is not None and obs_power is not None:
            # 跳过 k=0 (直流分量, 为均值平方)
            mask = obs_wn > 0
            has_data = True
            ax.semilogy(obs_wn[mask], obs_power[mask],
                        color='black', linewidth=2.5,
                        marker='*', markersize=8,
                        label='观测 (CMA-RA)')

        # 各模型预报功率谱
        for model_name, model_data in all_results.items():
            spec = model_data['spectra'].get(forecast_hour)
            if spec is None:
                continue
            wn = spec['wavenumbers']
            fc_power = spec['fc_power']
            mask = wn > 0
            has_data = True
            style = self.model_styles.get(model_name, {})
            ax.semilogy(wn[mask], fc_power[mask],
                        marker=style.get('marker', 'o'),
                        linestyle=style.get('linestyle', '-'),
                        linewidth=style.get('linewidth', 2),
                        markersize=5,
                        color=style.get('color', '#1f77b4'),
                        label=f'{model_name} 预报')

        if has_data:
            ax.set_xlabel('纬向波数 k', fontsize=12)
            ax.set_ylabel('功率谱 P(k) (m²)', fontsize=12)
            ax.set_title(f'Z500 功率谱 - {forecast_hour}h - {display_date}',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', which='both')
            ax.legend(fontsize=10, loc='best')
            plt.tight_layout()
            pf = self.output_dir / f'power_spectrum_{forecast_hour}h_{date_str}.png'
            plt.savefig(pf, dpi=300, bbox_inches='tight')
            print(f"已保存: {pf}")
        else:
            print(f"无功率谱数据: {forecast_hour}h")
        plt.close(fig)

        # --- 功率谱比值图 (预报/观测) ---
        if obs_wn is not None and obs_power is not None:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            has_ratio = False
            mask = obs_wn > 0
            obs_safe = np.where(obs_power > 0, obs_power, np.nan)

            for model_name, model_data in all_results.items():
                spec = model_data['spectra'].get(forecast_hour)
                if spec is None:
                    continue
                fc_power = spec['fc_power']
                ratio = fc_power[mask] / obs_safe[mask]
                has_ratio = True
                style = self.model_styles.get(model_name, {})
                ax2.plot(spec['wavenumbers'][mask], ratio,
                         marker=style.get('marker', 'o'),
                         linestyle=style.get('linestyle', '-'),
                         linewidth=style.get('linewidth', 2),
                         markersize=5,
                         color=style.get('color', '#1f77b4'),
                         label=f'{model_name}')

            if has_ratio:
                ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                            alpha=0.7, label='比值=1 (理想)')
                ax2.set_xlabel('纬向波数 k', fontsize=12)
                ax2.set_ylabel('功率谱比值 (预报/观测)', fontsize=12)
                ax2.set_title(f'Z500 功率谱比值 - {forecast_hour}h - {display_date}',
                              fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.legend(fontsize=10, loc='best')
                ax2.set_ylim(0, 3)
                plt.tight_layout()
                pf2 = self.output_dir / f'power_spectrum_ratio_{forecast_hour}h_{date_str}.png'
                plt.savefig(pf2, dpi=300, bbox_inches='tight')
                print(f"已保存: {pf2}")
            plt.close(fig2)


# ======================================================================
# 配置
# ======================================================================
def get_config(model_data_root=None, cra_root=None, cli_root=None, models=None):
    """生产环境配置。"""
    return {
        'cra_root': cra_root or
            '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026',
        'cli_root': cli_root or
            '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR',
        'model_data_root': model_data_root or
            '/gpu/zhaochy/fdp2/FCSTDATA',
        'models': models or list(MODEL_CONFIGS.keys()),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Z500 预报活跃度与功率谱检验 '
                    '(Fengqing / NJU-Earth / PuYun / YJ-TianJi / W2S)')
    parser.add_argument('--date', type=str, required=True,
                        help='预报起始时间 YYYYMMDDHH (e.g. 2026070500)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='检验模型 (默认: 全部 5 个)')
    parser.add_argument('--forecast-hours', type=int, nargs='+',
                        default=list(range(6, 361, 6)),
                        help='预报时效 (默认 6-360 步长 6)')
    parser.add_argument('--output-dir', type=str,
                        default='./activity_spectrum_results',
                        help='输出目录')
    parser.add_argument('--model-data-root', type=str, default=None,
                        help='FCSTDATA 根目录 (默认 /gpu/zhaochy/fdp2/FCSTDATA, '
                             '需有 {ModelName}/DF/{YYYYMMDD}/ 子目录)')
    parser.add_argument('--cra-root', type=str, default=None,
                        help='CRA 实况根目录')
    parser.add_argument('--cli-root', type=str, default=None,
                        help='气候态数据根目录')
    parser.add_argument('--spectrum-hours', type=int, nargs='+', default=None,
                        help='绘制功率谱对比图的时效 (默认 72 168 240 360)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试输出')
    args = parser.parse_args()

    if len(args.date) != 10:
        print("错误: --date 必须为 YYYYMMDDHH")
        return

    if args.models:
        invalid = [m for m in args.models if m not in MODEL_CONFIGS]
        if invalid:
            print(f"错误: 未知模型 {invalid}")
            print(f"可用: {list(MODEL_CONFIGS.keys())}")
            return

    config = get_config(
        model_data_root=args.model_data_root,
        cra_root=args.cra_root,
        cli_root=args.cli_root,
        models=args.models,
    )
    verifier = ActivitySpectrumVerifier(config=config, output_dir=args.output_dir)

    date_part = args.date[:8]
    hour_part = args.date[8:10]
    display_date = f"{date_part}{hour_part}Z"

    print("启动 Z500 预报活跃度与功率谱检验...")
    print(f"模型数据根目录: {config['model_data_root']}")
    print(f"实况根目录: {config['cra_root']}")
    print(f"气候态根目录: {config['cli_root']}")
    print(f"输出目录: {args.output_dir}")
    print(f"初始时间: {display_date}")
    print(f"预报时效: {args.forecast_hours}")
    print("=" * 60)

    # 路径检查
    print("\n检查路径...")
    model_root = Path(config['model_data_root'])
    print(f"  模型数据根目录: "
          f"{'存在' if model_root.exists() else '未找到'} ({model_root})")
    cra_root = Path(config['cra_root'])
    print(f"  CRA 根目录: "
          f"{'存在' if cra_root.exists() else '未找到'} ({cra_root})")
    cli_root = Path(config['cli_root'])
    print(f"  气候态根目录: "
          f"{'存在' if cli_root.exists() else '未找到'} ({cli_root})")

    all_results, display_date = verifier.run_verification(
        date_str=args.date,
        model_names=config['models'],
        forecast_hours=args.forecast_hours,
    )

    if all_results:
        print("\n保存结果到 CSV...")
        verifier.save_results(all_results, args.date)

        print("\n生成图表...")
        verifier.plot_results(all_results, args.date, display_date,
                              spectrum_hours=args.spectrum_hours)

        print("\n" + "=" * 60)
        print("检验完成!")
        print(f"结果保存至: {args.output_dir}")
        print("=" * 60)
    else:
        print("\n检验失败, 无有效结果")


if __name__ == "__main__":
    main()
