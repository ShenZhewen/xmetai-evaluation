#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-model deterministic forecast verification script
Supports: Fengqing, NJU-Earth, PuYun, YJ-TianJi, W2S

Model data path:  /gpu/zhaochy/fdp2/FCSTDATA/{ModelName}/DF/YYYYMMDD/
Observation path: /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026 (business env)
Climate path:      /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR  (business env)

Verifies surface variables (t2m, msl, u10, v10) and pressure level variable (z500).
Calculates RMSE, Bias (all variables) and ACC (z500 only).

Data format / naming convention follows ensemble_verifier.py:
  File: {ModelName}_{Version}_GLB_{PLEVELS|SURFACE}_{ResTag}_6HOR_{DFtag}_{init}_{lead:03d}.nc
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.ioff()

GRAVITY = 9.80665


# ======================================================================
# Model configurations — file naming elements per model
#   Standard pattern: {Model}_{Ver}_GLB_{PLEVELS|SURFACE}_{ResTag}_6HOR_{Tag}_{init}_{lead:03d}.nc
#     - DF (deterministic):  Tag = df_tag  ('FCST')
#     - ENS (ensemble):      Tag = ens_tag ('ENS_FCST')
#   Directory: /gpu/zhaochy/fdp2/FCSTDATA/{Model}/{DF|ENS}/{YYYYMMDD}/
#   Strategy: try DF first; if empty, fall back to ENS and use ensemble mean.
# ======================================================================
MODEL_CONFIGS = {
    'Fengqing': {
        'version': '1.0',
        'resolution_tag': 'OP25',   # letter O
        'df_tag': 'FCST',
        'ens_tag': 'ENS_FCST',
    },
    'NJU-Earth': {
        'version': 'v1',
        'resolution_tag': '0P25',   # digit 0
        'df_tag': 'FCST',
        'ens_tag': 'ENS_FCST',
    },
    'PuYun': {
        'version': 'v1',
        'resolution_tag': 'OP25',
        'df_tag': 'FCST',
        'ens_tag': 'ENS_FCST',
    },
    'YJ-TianJi': {
        'version': 'v0.1',
        'resolution_tag': '0P25',
        'df_tag': 'FCST',
        'ens_tag': 'ENS_FCST',
    },
    # W2S (onnx_infer_dfens.py 输出):
    #   文件名: W2S_V1_GLB_0P25_{ENS_}HOUR_{init}_{lead:03d}.nc
    #   不区分 PLEVELS/SURFACE, 所有通道在同一文件
    #   纬度升序(-90→90), read 时自动反转为降序以匹配实况
    #   z500 通道: 位势(m²/s²) → convert_z500 自动除以 g 得 m
    #   t2m/msl/u10/v10 通道: 单位与实况一致, 无需转换
    'W2S': {
        'version': 'V1',
        'resolution_tag': '0P25',
        'df_tag': 'HOUR',            # 确定性: W2S_V1_GLB_0P25_HOUR_...
        'ens_tag': 'ENS_HOUR',       # 集合预报: W2S_V1_GLB_0P25_ENS_HOUR_...
        'channel_based': True,       # 所有变量在单一文件, 用 'channel' 维度区分
    },
}


# ======================================================================
# Variable configurations — auto-detection candidates + truth GRIB keys
# ======================================================================
VAR_CONFIG = {
    # ---- pressure level variable (PLEVELS file) ----
    'z500': {
        'var_candidates': ['Z500', 'z500', 'Z', 'gh', 'GH', 'H500'],
        'model_data_type': 'pressure',
        'truth_data_type': 'pressure',
        'level': 500,
        'truth_keys': {'shortName': 'gh', 'typeOfLevel': 'isobaricInhPa', 'level': 500},
        'climate_var': 'gh',
        'full_name': '500hPa Geopotential Height',
        'units': 'm',
        'compute_acc': True,
    },
    # ---- surface variables (SURFACE file) ----
    't2m': {
        'var_candidates': ['T2M', 't2m', '2t', 'T_2M', 't2m_ave', 'TEMP2M'],
        'model_data_type': 'surface',
        'truth_data_type': 'surface',
        'level': 2,
        'truth_keys': {'shortName': '2t', 'typeOfLevel': 'heightAboveGround', 'level': 2},
        'climate_var': None,
        'full_name': '2m Temperature',
        'units': 'K',
        'compute_acc': False,
    },
    'msl': {
        'var_candidates': ['MSL', 'msl', 'prmsl', 'MSLP', 'slp', 'PSL'],
        'model_data_type': 'surface',
        'truth_data_type': 'pressure',   # MSL lives in ART_ATM (pressure) GRIB file
        'level': 0,
        'truth_keys': [
            {'shortName': 'msl', 'typeOfLevel': 'meanSea', 'level': 0},
            {'shortName': 'prmsl', 'typeOfLevel': 'meanSea', 'level': 0},
        ],
        'climate_var': None,
        'full_name': 'Mean Sea Level Pressure',
        'units': 'Pa',
        'compute_acc': False,
    },
    'u10': {
        'var_candidates': ['U10', 'u10', '10u', 'U_10M', 'UGRD'],
        'model_data_type': 'surface',
        'truth_data_type': 'surface',
        'level': 10,
        'truth_keys': {'shortName': '10u', 'typeOfLevel': 'heightAboveGround', 'level': 10},
        'climate_var': None,
        'full_name': '10m U Wind',
        'units': 'm/s',
        'compute_acc': False,
    },
    'v10': {
        'var_candidates': ['V10', 'v10', '10v', 'V_10M', 'VGRD'],
        'model_data_type': 'surface',
        'truth_data_type': 'surface',
        'level': 10,
        'truth_keys': {'shortName': '10v', 'typeOfLevel': 'heightAboveGround', 'level': 10},
        'climate_var': None,
        'full_name': '10m V Wind',
        'units': 'm/s',
        'compute_acc': False,
    },
}


# ======================================================================
# Auto-detection / unit-conversion utility functions
#   (ported from ensemble_verifier.py)
# ======================================================================
def detect_variable_name(ds, candidates):
    """Search xarray Dataset for a variable matching *candidates* (case-insensitive fallback)."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower_map = {v.lower(): v for v in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def normalize_unit_string(unit_str):
    """Normalise a unit string for comparison."""
    if unit_str is None:
        return ''
    u = unit_str.strip().lower().replace(' ', '').replace('**', '^')
    equivalences = {
        'm^2/s^2': 'm2/s2', 'm^2s^-2': 'm2/s2', 'm2s-2': 'm2/s2',
        'm^2s^(-2)': 'm2/s2',
        'kg/m^2': 'kg/m2', 'kgm-2': 'kg/m2', 'kgm^-2': 'kg/m2',
        'm/s': 'm/s', 'ms-1': 'm/s', 'm*s^-1': 'm/s',
        'millimeter': 'mm', 'millimeters': 'mm',
        'meter': 'm', 'meters': 'm', 'gpm': 'm',
    }
    return equivalences.get(u, u)


def convert_z500(data, source_unit):
    """Convert Z500 to geopotential height (m)."""
    su = normalize_unit_string(source_unit)
    if su in ('m', 'gpm', 'meter', 'meters'):
        return data
    if su in ('m2/s2',):
        return data / GRAVITY
    mean_val = float(np.nanmean(data))
    if mean_val > 10000:
        print(f"      \u26a0 Z500 unit unknown ({source_unit!r}), mean={mean_val:.1f} > 10000, "
              f"treating as m\u00b2/s\u00b2, dividing by g")
        return data / GRAVITY
    return data


def convert_t2m(data, source_unit):
    """Convert T2M to Kelvin."""
    su = normalize_unit_string(source_unit)
    if su in ('k', 'kelvin'):
        return data
    if su in ('c', 'celsius', 'degc'):
        return data + 273.15
    mean_val = float(np.nanmean(data))
    if mean_val < 100:
        print(f"      \u26a0 T2M unit unknown ({source_unit!r}), mean={mean_val:.1f} < 100, "
              f"treating as \u00b0C, converting to K")
        return data + 273.15
    return data


def convert_msl(data, source_unit):
    """Convert MSL to Pascals."""
    su = normalize_unit_string(source_unit)
    if su in ('pa', 'pascals'):
        return data
    if su in ('hpa', 'millibar', 'mb'):
        return data * 100.0
    mean_val = float(np.nanmean(data))
    if 800 < mean_val < 1500:
        print(f"      \u26a0 MSL unit unknown ({source_unit!r}), mean={mean_val:.1f}, "
              f"treating as hPa, converting to Pa")
        return data * 100.0
    return data


def convert_units(data, var_key, source_unit):
    """Dispatch unit conversion based on variable key."""
    if var_key == 'z500':
        return convert_z500(data, source_unit)
    if var_key == 't2m':
        return convert_t2m(data, source_unit)
    if var_key == 'msl':
        return convert_msl(data, source_unit)
    # u10, v10 — no conversion (m/s)
    return data


class MultiModelVerifier:
    def __init__(self, config, output_dir='./multi_model_results'):
        self.cra_root = Path(config.get('cra_root',
              '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026'))
        self.cli_root = Path(config.get('cli_root',
              '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR'))
        # FCSTDATA 根目录, 每个模型在自己子目录下:
        #   {model_data_root}/{ModelName}/DF/{YYYYMMDD}/
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

    # ==================================================================
    # Observation (CMA-RA / CRA40) data — GRIB2  (unchanged from original)
    # ==================================================================
    def find_cra_file(self, date_str, data_type='surface'):
        """Find CRA40 truth data file — date_str = YYYYMMDDHH."""
        date_only = date_str[:8]
        date_dir = self.cra_root / date_only
        if not date_dir.exists():
            return None

        if data_type == 'surface':
            filename = f"CRA40LAND_SURFACE_{date_str}_GLB_0P25_HOUR_V1_0_0.grib"
            cand = date_dir / filename
            if cand.exists():
                return cand
            filename2 = f"ART_ATM_GLB_0P25_6HOR_ANAL_{date_str}.grib2"
            cand2 = date_dir / filename2
            if cand2.exists():
                return cand2
            files = list(date_dir.glob(f"*SURFACE*{date_str}*.grib*")) + \
                    list(date_dir.glob(f"*{date_str}*.grib2"))
            return files[0] if files else None
        else:  # pressure
            filename = f"ART_ATM_GLB_0P25_6HOR_ANAL_{date_str}.grib2"
            cand = date_dir / filename
            if cand.exists():
                return cand
            files = list(date_dir.glob(f"*{date_str}*.grib2"))
            return files[0] if files else None

    def find_climate_file(self, month, day, hour='00'):
        mmdd = f"{int(month):02d}{int(day):02d}"
        filename = f"ART_ATM_GLB_0P25_CLI_ANAL_{mmdd}{hour}.grib2"
        cand = self.cli_root / filename
        if cand.exists():
            return cand
        files = list(self.cli_root.glob(f"*{mmdd}*.grib2"))
        return files[0] if files else None

    def read_climate_data(self, filepath, var_name, level):
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
                return data
        except Exception as e:
            print(f"      Error reading climate data: {e}")
        return None

    def read_grib_data(self, filepath, keys_to_try, data_type='truth', level=None):
        """Read data from GRIB file, trying multiple key combinations."""
        if not filepath or not filepath.exists():
            return None
        if not isinstance(keys_to_try, list):
            keys_to_try = [keys_to_try]

        for keys in keys_to_try:
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
                    if 'lat' in data.coords:
                        lat_values = data.lat.values
                        if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                            reversed_indices = np.arange(len(data.lat))[::-1]
                            data = data.isel(lat=reversed_indices)
                    return data
            except Exception:
                continue
        return None

    def read_truth_data(self, filepath, var_cfg):
        """Read truth (observation) data using var_cfg truth_keys."""
        keys = var_cfg['truth_keys']
        level = var_cfg.get('level')
        return self.read_grib_data(filepath, keys, 'truth', level)

    # ==================================================================
    # Model forecast data — NetCDF with auto-detection
    #   (replaces old read_netcdf_data / read_fengqing_netcdf_data /
    #    model-specific read_grib_data calls)
    # ==================================================================
    def find_model_file(self, model_name, date_str, forecast_hour, data_type='surface'):
        """Find model forecast file.

        Strategy: try DF (deterministic) directory first; if not found,
        fall back to ENS (ensemble) directory. For ENS data with multiple
        members, the caller (read_model_data) will compute the ensemble mean.

        Path:  {model_data_root}/{ModelName}/{DF|ENS}/{YYYYMMDD}/{filename}
        File:  {Model}_{Ver}_GLB_{PLEVELS|SURFACE}_{ResTag}_6HOR_{Tag}_{init}_{lead:03d}.nc
               Tag = df_tag ('FCST') for DF, ens_tag ('ENS_FCST') for ENS
        W2S legacy: W2S_V1_GLB_0P25_HOUR_{init}_{lead:03d}.nc
        """
        if model_name not in MODEL_CONFIGS:
            return None
        cfg = MODEL_CONFIGS[model_name]
        date_only = date_str[:8]
        type_tag = 'SURFACE' if data_type == 'surface' else 'PLEVELS'
        lead_str = f"{forecast_hour:03d}"

        # Try DF first, then ENS (e.g. W2S has empty DF, data only in ENS)
        for fcst_type, tag_key in [('DF', 'df_tag'), ('ENS', 'ens_tag')]:
            date_dir = self.model_data_root / model_name / fcst_type / date_only
            if not date_dir.exists():
                continue

            # 1) W2S channel-based naming (single file, no PLEVELS/SURFACE/6HOR tag)
            #    W2S_V1_GLB_0P25_{ENS_}HOUR_{init}_{lead:03d}.nc
            if cfg.get('channel_based'):
                ens_prefix = "ENS_" if fcst_type == 'ENS' else ""
                alt_fname = (f"{model_name}_{cfg['version']}_GLB_"
                             f"{cfg['resolution_tag']}_{ens_prefix}HOUR_"
                             f"{date_str}_{lead_str}.nc")
                alt_cand = date_dir / alt_fname
                if alt_cand.exists():
                    return alt_cand
                # loose glob (ens_prefix distinguishes ENS_HOUR vs HOUR)
                for pat in [
                    f"*{model_name}*{ens_prefix}HOUR*{date_str}_{lead_str}*.nc",
                    f"*{model_name}*HOUR*{date_str}_{lead_str}*.nc",
                    f"*{model_name}*{date_str}_{lead_str}*.nc",
                ]:
                    alt_files = list(date_dir.glob(pat))
                    if alt_files:
                        return alt_files[0]

            # 2) standard naming
            tag = cfg.get(tag_key, 'FCST')
            fname = (f"{model_name}_{cfg['version']}_GLB_{type_tag}_"
                     f"{cfg['resolution_tag']}_6HOR_{tag}_"
                     f"{date_str}_{lead_str}.nc")
            cand = date_dir / fname
            if cand.exists():
                return cand

            # 3) loose glob (ignore OP25 vs 0P25, FCST vs ENS_FCST)
            for pat in [
                f"*{model_name}*{type_tag}*6HOR*{date_str}_{lead_str}*.nc",
                f"*{model_name}*{type_tag}*{date_str}_{lead_str}*.nc",
                f"*{model_name}*{date_str}_{lead_str}*.nc",
            ]:
                files = list(date_dir.glob(pat))
                if files:
                    return files[0]

        return None

    def read_model_data(self, filepath, var_key, model_name, debug=False):
        """Generic NetCDF reader for model forecast data.

        Handles both standard variable-based NetCDF and W2S channel-based NetCDF.
        Auto-detects variable name, converts units, normalises latitude to descending.
        """
        if filepath is None:
            return None
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        var_cfg = VAR_CONFIG.get(var_key)
        if var_cfg is None:
            return None

        try:
            ds = xr.open_dataset(str(filepath))

            # --- W2S channel-based format ---
            if 'channel' in ds and 'data' in ds.data_vars:
                data = self._read_channel_data(ds, var_cfg, var_key, model_name, debug)
                ds.close()
                return data

            # --- standard NetCDF: auto-detect variable ---
            var_name = detect_variable_name(ds, var_cfg['var_candidates'])
            if var_name is None:
                print(f"      \u2717 [{model_name}] '{var_key}' not found. "
                      f"Candidates: {var_cfg['var_candidates']}, "
                      f"Available: {list(ds.data_vars)}")
                ds.close()
                return None

            data = ds[var_name]
            print(f"      [{model_name}] {var_key}: var='{var_name}', shape={data.shape}")

            # squeeze singleton dims
            for d in ('level', 'time', 'dtime', 'isobaricInhPa', 'lev'):
                if d in data.dims and data.sizes[d] == 1:
                    data = data.squeeze(dim=d, drop=True)
            # handle ensemble member dim: squeeze singleton, mean for multi-member
            member_dims = [d for d in list(data.dims)
                           if d.lower() in ('member', 'ensemble', 'ens',
                                             'number', 'realization')]
            for d in member_dims:
                if data.sizes[d] == 1:
                    data = data.squeeze(dim=d, drop=True)
                else:
                    print(f"      [{model_name}] 集合预报 {data.sizes[d]} 成员 → 集合平均")
                    data = data.mean(dim=d)

            # normalise coords
            if 'latitude' in data.coords:
                data = data.rename({'latitude': 'lat', 'longitude': 'lon'})

            # unit conversion
            source_unit = data.attrs.get('units', None)
            data = convert_units(data, var_key, source_unit)

            if debug:
                print(f"      [DEBUG] {var_key} after conversion — "
                      f"min={float(data.values.min()):.2f}, "
                      f"max={float(data.values.max()):.2f}, "
                      f"mean={float(data.values.mean()):.2f}")

            # latitude descending
            if 'lat' in data.coords:
                lat_values = data.lat.values
                if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                    print(f"      [{model_name}] Reversing latitude to descending")
                    data = data.isel(lat=slice(None, None, -1))

            ds.close()
            return data

        except Exception as e:
            print(f"      [{model_name}] Error reading {var_key}: {e}")
            return None

    def _read_channel_data(self, ds, var_cfg, var_key, model_name, debug=False):
        """Read W2S channel-based NetCDF (variables stored as channels).

        W2S stores all variables in a single file with a 'channel' dimension.
        Unit conversion is delegated to convert_units() which handles z500,
        t2m, msl and falls through for u10/v10 (already in m/s).
        """
        channel_names = ds['channel'].values
        channel_list = []
        for name in channel_names:
            if isinstance(name, bytes):
                channel_list.append(name.decode('utf-8'))
            else:
                channel_list.append(str(name))

        # find matching channel by candidates
        channel_name = None
        for cand in var_cfg['var_candidates']:
            if cand in channel_list:
                channel_name = cand
                break
        if channel_name is None:
            lower_map = {c.lower(): c for c in channel_list}
            for cand in var_cfg['var_candidates']:
                if cand.lower() in lower_map:
                    channel_name = lower_map[cand.lower()]
                    break

        if channel_name is None:
            print(f"      \u2717 [{model_name}] Channel not found for "
                  f"{var_cfg['full_name']}. Available: {channel_list}")
            return None

        idx = channel_list.index(channel_name)
        data = ds['data'].isel(channel=idx, time=0)
        print(f"      [{model_name}] {var_key}: channel='{channel_name}', shape={data.shape}")

        # handle ensemble member dim: squeeze singleton, mean for multi-member
        # (DF files have member=1 → squeezed; ENS fallback has multi-member → mean)
        member_dims = [d for d in list(data.dims)
                       if d.lower() in ('member', 'ensemble', 'ens',
                                         'number', 'realization')]
        for d in member_dims:
            if data.sizes[d] == 1:
                data = data.squeeze(dim=d, drop=True)
            else:
                print(f"      [{model_name}] 集合预报 {data.sizes[d]} 成员 → 集合平均")
                data = data.mean(dim=d)

        if 'latitude' in data.coords:
            data = data.rename({'latitude': 'lat', 'longitude': 'lon'})

        # unit conversion via convert_units (handles z500/t2m/msl; u10/v10 passthrough)
        source_unit = data.attrs.get('units', None)
        data = convert_units(data, var_key, source_unit)

        if debug:
            print(f"      [DEBUG] channel '{channel_name}' ({var_key}) after conversion — "
                  f"min={float(data.values.min()):.2f}, "
                  f"max={float(data.values.max()):.2f}, "
                  f"mean={float(data.values.mean()):.2f}")

        # latitude descending
        if 'lat' in data.coords:
            lat_values = data.lat.values
            if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                print(f"      [{model_name}] Reversing latitude to descending")
                data = data.isel(lat=slice(None, None, -1))

        return data

    # ==================================================================
    # Grid interpolation + metric calculations  (unchanged)
    # ==================================================================
    def interpolate_to_common_grid(self, source_data, target_lat, target_lon):
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
                    print("      [GRID] Latitude order mismatch, reversing source")
                    source_data = source_data.isel(lat=slice(None, None, -1))

            return source_data.interp(
                lat=target_lat, lon=target_lon,
                kwargs={"fill_value": "extrapolate"})
        except Exception as e:
            print(f"      Error in grid interpolation: {e}")
            return None

    def calculate_weighted_metrics(self, forecast_data, truth_data):
        """Latitude-weighted RMSE and Bias."""
        try:
            if forecast_data.shape != truth_data.shape:
                return None, None
            if 'lat' in forecast_data.coords:
                lat = forecast_data.lat.values
            elif 'latitude' in forecast_data.coords:
                lat = forecast_data.latitude.values
            else:
                return None, None

            w = np.cos(np.deg2rad(lat))
            if len(forecast_data.shape) == 2:
                w_2d = np.broadcast_to(w[:, np.newaxis], forecast_data.shape)
            else:
                w_2d = w

            diff = forecast_data.values - truth_data.values
            weight_sum = np.nansum(w_2d * ~np.isnan(diff))
            if weight_sum == 0:
                return 0.0, 0.0
            bias = np.nansum(w_2d * diff) / weight_sum
            rmse = np.sqrt(np.nansum(w_2d * diff ** 2) / weight_sum)
            return float(rmse), float(bias)
        except Exception as e:
            print(f"      Error calculating metrics: {e}")
            return None, None

    def calculate_acc(self, forecast_data, truth_data, climate_data):
        """Anomaly Correlation Coefficient with latitude weighting."""
        try:
            if (forecast_data.shape != truth_data.shape or
                    forecast_data.shape != climate_data.shape):
                print(f"      Shape mismatch: fc={forecast_data.shape}, "
                      f"truth={truth_data.shape}, clim={climate_data.shape}")
                return None

            lat = forecast_data.lat.values
            w = np.cos(np.deg2rad(lat))
            if len(forecast_data.shape) == 2:
                w_2d = np.broadcast_to(w[:, np.newaxis], forecast_data.shape)
            else:
                w_2d = w

            fc_anom = forecast_data.values - climate_data.values
            obs_anom = truth_data.values - climate_data.values

            w_sum = w_2d.sum()
            fc_mean = (w_2d * fc_anom).sum() / w_sum
            obs_mean = (w_2d * obs_anom).sum() / w_sum

            num = (w_2d * (fc_anom - fc_mean) * (obs_anom - obs_mean)).sum()
            den = np.sqrt(
                (w_2d * (fc_anom - fc_mean) ** 2).sum() *
                (w_2d * (obs_anom - obs_mean) ** 2).sum())
            if den == 0:
                return None
            return float(num / den)
        except Exception as e:
            print(f"      Error calculating ACC: {e}")
            return None

    # ==================================================================
    # Verification flow
    # ==================================================================
    def verify_single_forecast(self, model_name, date_str, forecast_hour,
                              data_type='surface'):
        """Verify a single model / forecast hour for all variables of *data_type*."""
        print(f"  Verifying {model_name} {data_type}: {forecast_hour:03d}h")

        model_file = self.find_model_file(
            model_name, date_str, forecast_hour, data_type)
        if not model_file:
            print(f"    \u2717\u2717\u2717\u2717 Model file not found")
            return None
        print(f"    \u2713 Model file: {model_file.name}")

        start_dt = datetime.strptime(date_str, "%Y%m%d%H")
        truth_dt = start_dt + timedelta(hours=forecast_hour)
        truth_date = truth_dt.strftime("%Y%m%d%H")
        print(f"    \u2713 Truth time: {truth_date}")

        vars_to_verify = {k: v for k, v in VAR_CONFIG.items()
                          if v['model_data_type'] == data_type}
        results = {}

        for var_key, var_cfg in vars_to_verify.items():
            print(f"    Variable: {var_cfg['full_name']} ({var_key})")

            # --- read model data ---
            model_data = self.read_model_data(
                model_file, var_key, model_name, debug=(var_key == 'msl'))
            if model_data is None:
                print(f"      \u2717\u2717\u2717\u2717 Cannot read model data")
                continue

            # --- read truth data ---
            truth_dt_type = var_cfg.get('truth_data_type', data_type)
            truth_file = self.find_cra_file(truth_date, data_type=truth_dt_type)
            if not truth_file:
                print(f"    \u2717\u2717\u2717\u2717 Truth file not found")
                continue
            print(f"    \u2713 Truth file: {truth_file.name}")

            truth_data = self.read_truth_data(truth_file, var_cfg)
            if truth_data is None:
                print(f"      \u2717\u2717\u2717\u2717 Cannot read truth data")
                continue

            # --- grid alignment ---
            if model_data.shape != truth_data.shape:
                print(f"      Grid mismatch: model={model_data.shape}, "
                      f"truth={truth_data.shape}, interpolating truth...")
                truth_data = self.interpolate_to_common_grid(
                    truth_data, model_data.lat.values, model_data.lon.values)
                if truth_data is None:
                    print(f"      \u2717\u2717\u2717\u2717 Interpolation failed")
                    continue

            # --- RMSE & Bias ---
            rmse, bias = self.calculate_weighted_metrics(model_data, truth_data)

            # --- ACC (z500 only) ---
            acc = None
            if var_cfg.get('compute_acc') and rmse is not None:
                climate_file = self.find_climate_file(
                    truth_dt.month, truth_dt.day, f"{truth_dt.hour:02d}")
                if climate_file:
                    print(f"    \u2713 Climate file: {climate_file.name}")
                    climate_data = self.read_climate_data(
                        climate_file, var_cfg.get('climate_var', 'gh'),
                        var_cfg.get('level', 500))
                    if climate_data is not None:
                        if climate_data.shape != model_data.shape:
                            climate_data = self.interpolate_to_common_grid(
                                climate_data, model_data.lat.values,
                                model_data.lon.values)
                        if climate_data is not None:
                            acc = self.calculate_acc(
                                model_data, truth_data, climate_data)
                            if acc is not None:
                                print(f"      \u2713 ACC: {acc:.4f}")
                            else:
                                print(f"      \u2717\u2717\u2717\u2717 ACC failed")
                        else:
                            print(f"      \u2717\u2717\u2717\u2717 Climate interpolation failed")
                    else:
                        print(f"      \u2717\u2717\u2717\u2717 Cannot read climate data")
                else:
                    print(f"      \u2717\u2717\u2717\u2717 Climate file not found")

            if rmse is not None and bias is not None:
                results[var_key] = {
                    'rmse': rmse,
                    'bias': bias,
                    'units': var_cfg['units'],
                    'full_name': var_cfg['full_name'],
                }
                if acc is not None:
                    results[var_key]['acc'] = acc
                print(f"      \u2713 RMSE: {rmse:.4f} {var_cfg['units']}")
                print(f"      \u2713 Bias: {bias:+.4f} {var_cfg['units']}")
            else:
                print(f"      \u2717\u2717\u2717\u2717 Calculation failed")

        if results:
            print(f"    \u2713 Verified {len(results)} variables")
            return results
        print(f"    \u2717\u2717\u2717\u2717 No variables verified")
        return None

    def run_verification(self, date_str, model_names=None,
                         forecast_hours=None,
                         verify_surface=True, verify_pressure=True):
        if forecast_hours is None:
            forecast_hours = list(range(6, 25, 6))
        if model_names is None:
            model_names = self.models

        display_date = f"{date_str}Z"
        print("\n" + "=" * 60)
        print("Multi-Model Deterministic Forecast Verification")
        print("=" * 60)
        print(f"Initial Time: {display_date}")
        print(f"Models: {model_names}")
        print(f"Forecast Hours: {forecast_hours}")
        print(f"Surface: {verify_surface}  Pressure: {verify_pressure}")
        print(f"Model data root: {self.model_data_root}")
        print(f"CRA root: {self.cra_root}")
        print("=" * 60)

        all_results = {}
        for model_name in model_names:
            all_results[model_name] = {'surface': {}, 'pressure': {}}
            # init surface vars
            for vk, vc in VAR_CONFIG.items():
                if vc['model_data_type'] == 'surface':
                    all_results[model_name]['surface'][vk] = {
                        'hours': [], 'rmse': [], 'bias': []}
            # init pressure vars
            for vk, vc in VAR_CONFIG.items():
                if vc['model_data_type'] == 'pressure':
                    entry = {'hours': [], 'rmse': [], 'bias': []}
                    if vc.get('compute_acc'):
                        entry['acc'] = []
                    all_results[model_name]['pressure'][vk] = entry

        for model_name in model_names:
            print(f"\nModel: {model_name}")
            print("-" * 40)
            valid_count = 0

            for f_hour in forecast_hours:
                surface_results = None
                pressure_results = None
                if verify_surface:
                    surface_results = self.verify_single_forecast(
                        model_name, date_str, f_hour, 'surface')
                if verify_pressure:
                    pressure_results = self.verify_single_forecast(
                        model_name, date_str, f_hour, 'pressure')

                if surface_results or pressure_results:
                    valid_count += 1
                    if surface_results:
                        for vk, vr in surface_results.items():
                            all_results[model_name]['surface'][vk]['hours'].append(f_hour)
                            all_results[model_name]['surface'][vk]['rmse'].append(vr['rmse'])
                            all_results[model_name]['surface'][vk]['bias'].append(vr['bias'])
                    if pressure_results:
                        for vk, vr in pressure_results.items():
                            all_results[model_name]['pressure'][vk]['hours'].append(f_hour)
                            all_results[model_name]['pressure'][vk]['rmse'].append(vr['rmse'])
                            all_results[model_name]['pressure'][vk]['bias'].append(vr['bias'])
                            if 'acc' in vr and 'acc' in all_results[model_name]['pressure'][vk]:
                                all_results[model_name]['pressure'][vk]['acc'].append(vr['acc'])
                else:
                    print(f"  {f_hour:03d}h: \u2717 Failed")

            print(f"  Total: {valid_count}/{len(forecast_hours)} verified")

        return all_results, display_date

    # ==================================================================
    # CSV output
    # ==================================================================
    def save_results(self, all_results, date_str):
        for model_name, model_results in all_results.items():
            # surface
            surface_dfs = []
            for vk, vc in VAR_CONFIG.items():
                if vc['model_data_type'] != 'surface':
                    continue
                data = model_results['surface'].get(vk, {})
                if data.get('hours'):
                    df = pd.DataFrame({
                        'forecast_hour': data['hours'],
                        'rmse': data['rmse'],
                        'bias': data['bias'],
                    })
                    df['variable'] = vk
                    df['variable_name'] = vc['full_name']
                    df['units'] = vc['units']
                    df['data_type'] = 'surface'
                    surface_dfs.append(df)
            if surface_dfs:
                df = pd.concat(surface_dfs, ignore_index=True)
                csv = self.output_dir / f'{model_name}_surface_verification_{date_str}.csv'
                df.to_csv(csv, index=False, encoding='utf-8')
                print(f"Saved: {csv}")

            # pressure
            pressure_dfs = []
            for vk, vc in VAR_CONFIG.items():
                if vc['model_data_type'] != 'pressure':
                    continue
                data = model_results['pressure'].get(vk, {})
                if data.get('hours'):
                    cols = {
                        'forecast_hour': data['hours'],
                        'rmse': data['rmse'],
                        'bias': data['bias'],
                    }
                    if 'acc' in data and data['acc']:
                        cols['acc'] = data['acc']
                    df = pd.DataFrame(cols)
                    df['variable'] = vk
                    df['variable_name'] = vc['full_name']
                    df['units'] = vc['units']
                    df['data_type'] = 'pressure_level'
                    pressure_dfs.append(df)
            if pressure_dfs:
                df = pd.concat(pressure_dfs, ignore_index=True)
                csv = self.output_dir / f'{model_name}_pressure_verification_{date_str}.csv'
                df.to_csv(csv, index=False, encoding='utf-8')
                print(f"Saved: {csv}")

    # ==================================================================
    # Plotting
    # ==================================================================
    def plot_results(self, all_results, date_str, display_date,
                     verify_surface=True, verify_pressure=True):
        if not all_results:
            print("No data to plot")
            return
        model_names = list(all_results.keys())

        if verify_surface:
            print("\nPlotting surface variables...")
            self._plot_summary(all_results, model_names, date_str, display_date,
                               data_type='surface')
            for vk, vc in VAR_CONFIG.items():
                if vc['model_data_type'] == 'surface':
                    self._plot_single_variable(
                        all_results, model_names, vk, vc,
                        date_str, display_date, 'surface')

        if verify_pressure:
            print("\nPlotting pressure level variables...")
            for vk, vc in VAR_CONFIG.items():
                if vc['model_data_type'] == 'pressure':
                    self._plot_single_variable(
                        all_results, model_names, vk, vc,
                        date_str, display_date, 'pressure')
                    if vc.get('compute_acc'):
                        self._plot_acc(all_results, model_names, vk, vc,
                                       date_str, display_date)

    def _plot_summary(self, all_results, model_names, date_str, display_date,
                      data_type='surface'):
        """Summary comparison plot for all variables of a data_type."""
        vars_in_type = [(vk, vc) for vk, vc in VAR_CONFIG.items()
                        if vc['model_data_type'] == data_type]
        n_vars = len(vars_in_type)
        if n_vars == 0:
            return
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_vars > 1 else [axes]

        for idx, (vk, vc) in enumerate(vars_in_type):
            if idx >= len(axes):
                break
            ax = axes[idx]
            has_data = False
            for model_name in model_names:
                data = all_results.get(model_name, {}).get(data_type, {}).get(vk, {})
                hours = data.get('hours', [])
                rmse = data.get('rmse', [])
                if hours and rmse:
                    has_data = True
                    si = np.argsort(hours)
                    sh = np.array(hours)[si]
                    sr = np.array(rmse)[si]
                    style = self.model_styles.get(model_name, {})
                    ax.plot(sh, sr, marker=style.get('marker', 'o'),
                            linestyle=style.get('linestyle', '-'),
                            linewidth=style.get('linewidth', 2),
                            markersize=6, color=style.get('color', '#1f77b4'),
                            label=model_name)
            if has_data:
                ax.set_xlabel('Forecast Hour', fontsize=12)
                ax.set_ylabel(f'RMSE ({vc["units"]})', fontsize=12)
                ax.set_title(vc['full_name'], fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                if idx == 0:
                    ax.legend(fontsize=10, loc='best')
                if 'sh' in dir() and len(sh) > 0:
                    ax.set_xticks(sh[::4] if len(sh) > 20 else sh)
                    ax.set_xlim(min(sh) - 1, max(sh) + 1)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_title(vc['full_name'], fontsize=13, fontweight='bold')

        for idx in range(n_vars, len(axes)):
            axes[idx].set_visible(False)

        tag = 'Surface' if data_type == 'surface' else 'Pressure'
        plt.suptitle(f'Multi-Model {tag} RMSE - {display_date}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_file = self.output_dir / f'{data_type}_summary_rmse_{date_str}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot_file}")

    def _plot_single_variable(self, all_results, model_names, var_key, var_cfg,
                              date_str, display_date, data_type):
        """RMSE + Bias plots for a single variable."""
        # --- RMSE ---
        fig_r, ax_r = plt.subplots(figsize=(10, 6))
        has_r = False
        for model_name in model_names:
            data = all_results.get(model_name, {}).get(data_type, {}).get(var_key, {})
            hours = data.get('hours', [])
            rmse = data.get('rmse', [])
            if hours and rmse:
                has_r = True
                si = np.argsort(hours)
                sh = np.array(hours)[si]
                sr = np.array(rmse)[si]
                style = self.model_styles.get(model_name, {})
                ax_r.plot(sh, sr, marker=style.get('marker', 'o'),
                          linestyle=style.get('linestyle', '-'),
                          linewidth=style.get('linewidth', 2),
                          markersize=6, color=style.get('color', '#1f77b4'),
                          label=model_name)
        if has_r:
            ax_r.set_xlabel('Forecast Hour', fontsize=12)
            ax_r.set_ylabel(f'RMSE ({var_cfg["units"]})', fontsize=12)
            ax_r.set_title(f'{var_cfg["full_name"]} RMSE - {display_date}',
                           fontsize=14, fontweight='bold')
            ax_r.grid(True, alpha=0.3, linestyle='--')
            ax_r.legend(fontsize=10, loc='best')
            if 'sh' in dir() and len(sh) > 0:
                ax_r.set_xticks(sh[::4] if len(sh) > 20 else sh)
                ax_r.set_xlim(min(sh) - 1, max(sh) + 1)
            plt.tight_layout()
            pf = self.output_dir / f'{var_key}_{data_type}_rmse_{date_str}.png'
            plt.savefig(pf, dpi=300, bbox_inches='tight')
            print(f"Saved: {pf}")
        else:
            print(f"No RMSE data for {var_key}")
        plt.close(fig_r)

        # --- Bias ---
        fig_b, ax_b = plt.subplots(figsize=(10, 6))
        has_b = False
        for model_name in model_names:
            data = all_results.get(model_name, {}).get(data_type, {}).get(var_key, {})
            hours = data.get('hours', [])
            bias = data.get('bias', [])
            if hours and bias:
                has_b = True
                si = np.argsort(hours)
                sh = np.array(hours)[si]
                sb = np.array(bias)[si]
                style = self.model_styles.get(model_name, {})
                ax_b.plot(sh, sb, marker=style.get('marker', 'o'),
                          linestyle=style.get('linestyle', '-'),
                          linewidth=style.get('linewidth', 2),
                          markersize=6, color=style.get('color', '#1f77b4'),
                          label=model_name)
        if has_b:
            ax_b.set_xlabel('Forecast Hour', fontsize=12)
            ax_b.set_ylabel(f'Bias ({var_cfg["units"]})', fontsize=12)
            ax_b.set_title(f'{var_cfg["full_name"]} Bias - {display_date}',
                           fontsize=14, fontweight='bold')
            ax_b.grid(True, alpha=0.3, linestyle='--')
            ax_b.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax_b.legend(fontsize=10, loc='best')
            if 'sh' in dir() and len(sh) > 0:
                ax_b.set_xticks(sh[::4] if len(sh) > 20 else sh)
                ax_b.set_xlim(min(sh) - 1, max(sh) + 1)
            plt.tight_layout()
            pf = self.output_dir / f'{var_key}_{data_type}_bias_{date_str}.png'
            plt.savefig(pf, dpi=300, bbox_inches='tight')
            print(f"Saved: {pf}")
        else:
            print(f"No Bias data for {var_key}")
        plt.close(fig_b)

    def _plot_acc(self, all_results, model_names, var_key, var_cfg,
                  date_str, display_date):
        """ACC plot for a single variable (typically z500)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        for model_name in model_names:
            data = all_results.get(model_name, {}).get('pressure', {}).get(var_key, {})
            hours = data.get('hours', [])
            acc = data.get('acc', [])
            if hours and acc and len(hours) == len(acc):
                has_data = True
                si = np.argsort(hours)
                sh = np.array(hours)[si]
                sa = np.array(acc)[si]
                style = self.model_styles.get(model_name, {})
                ax.plot(sh, sa, marker=style.get('marker', 'o'),
                        linestyle=style.get('linestyle', '-'),
                        linewidth=style.get('linewidth', 2),
                        markersize=6, color=style.get('color', '#1f77b4'),
                        label=model_name)
        if has_data:
            ax.set_xlabel('Forecast Hour', fontsize=12)
            ax.set_ylabel('ACC', fontsize=12)
            ax.set_title(f'{var_cfg["full_name"]} ACC - {display_date}',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1,
                       alpha=0.7, label='ACC=0.6')
            ax.legend(fontsize=10, loc='best')
            ax.set_ylim(-0.1, 1.1)
            if 'sh' in dir() and len(sh) > 0:
                ax.set_xticks(sh[::4] if len(sh) > 20 else sh)
                ax.set_xlim(min(sh) - 1, max(sh) + 1)
            plt.tight_layout()
            pf = self.output_dir / f'{var_key}_acc_{date_str}.png'
            plt.savefig(pf, dpi=300, bbox_inches='tight')
            print(f"Saved: {pf}")
        else:
            print(f"No ACC data for {var_key}")
        plt.close(fig)


# ======================================================================
# Configuration
# ======================================================================
def get_config(model_data_root=None, cra_root=None, cli_root=None, models=None):
    """Production configuration.

    Model data path:  /gpu/zhaochy/fdp2/FCSTDATA/{ModelName}/DF/YYYYMMDD/
    Observation path: /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026  (unchanged)
    Climate path:     /gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR  (unchanged)
    """
    return {
        'cra_root': cra_root or
            '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026',
        'cli_root': cli_root or
            '/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR',
        # FCSTDATA 根目录, 路径下需有 {ModelName}/DF/{YYYYMMDD}/ 结构
        'model_data_root': model_data_root or
            '/gpu/zhaochy/fdp2/FCSTDATA',
        'models': models or list(MODEL_CONFIGS.keys()),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Multi-model deterministic forecast verification '
                    '(Fengqing / NJU-Earth / PuYun / YJ-TianJi / W2S)')
    parser.add_argument('--date', type=str, required=True,
                        help='Forecast start YYYYMMDDHH (e.g. 2026082000)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to verify (default: all 5)')
    parser.add_argument('--forecast-hours', type=int, nargs='+',
                        default=list(range(6, 361, 6)),
                        help='Forecast hours (default 6-360 step 6)')
    parser.add_argument('--output-dir', type=str,
                        default='./verification_results',
                        help='Output directory')
    parser.add_argument('--surface-only', action='store_true',
                        help='Verify only surface variables')
    parser.add_argument('--pressure-only', action='store_true',
                        help='Verify only pressure level variables')
    parser.add_argument('--model-data-root', type=str, default=None,
                        help='FCSTDATA root (default '
                             '/gpu/zhaochy/fdp2/FCSTDATA, expects '
                             '{ModelName}/DF/{YYYYMMDD}/ subdirs)')
    parser.add_argument('--cra-root', type=str, default=None,
                        help='CRA observation root')
    parser.add_argument('--cli-root', type=str, default=None,
                        help='Climate data root')
    args = parser.parse_args()

    if len(args.date) != 10:
        print("Error: --date must be YYYYMMDDHH")
        return

    verify_surface = not args.pressure_only
    verify_pressure = not args.surface_only

    # validate model names
    if args.models:
        invalid = [m for m in args.models if m not in MODEL_CONFIGS]
        if invalid:
            print(f"Error: unknown models {invalid}")
            print(f"Available: {list(MODEL_CONFIGS.keys())}")
            return

    config = get_config(
        model_data_root=args.model_data_root,
        cra_root=args.cra_root,
        cli_root=args.cli_root,
        models=args.models,
    )
    verifier = MultiModelVerifier(config=config, output_dir=args.output_dir)

    date_part = args.date[:8]
    hour_part = args.date[8:10]
    display_date = f"{date_part}{hour_part}Z"

    print("Starting multi-model deterministic forecast verification...")
    print(f"Model data root: {config['model_data_root']}")
    print(f"Observation root: {config['cra_root']}")
    print(f"Climate root: {config['cli_root']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Initial Time: {display_date}")
    print(f"Forecast hours: {args.forecast_hours}")
    print(f"Surface: {verify_surface}  Pressure: {verify_pressure}")
    print("=" * 60)

    # check paths
    print("\nChecking paths...")
    model_root = Path(config['model_data_root'])
    print(f"  Model data root: "
          f"{'exists' if model_root.exists() else 'NOT FOUND'} ({model_root})")
    cra_root = Path(config['cra_root'])
    print(f"  CRA root: "
          f"{'exists' if cra_root.exists() else 'NOT FOUND'} ({cra_root})")
    cli_root = Path(config['cli_root'])
    print(f"  Climate root: "
          f"{'exists' if cli_root.exists() else 'NOT FOUND'} ({cli_root})")

    all_results, display_date = verifier.run_verification(
        date_str=args.date,
        model_names=config['models'],
        forecast_hours=args.forecast_hours,
        verify_surface=verify_surface,
        verify_pressure=verify_pressure,
    )

    if all_results:
        print("\nSaving results to CSV...")
        verifier.save_results(all_results, args.date)

        print("\nGenerating plots...")
        verifier.plot_results(all_results, args.date, display_date,
                              verify_surface=verify_surface,
                              verify_pressure=verify_pressure)

        print("\n" + "=" * 60)
        print("Verification completed!")
        print(f"Results saved to: {args.output_dir}")
        print("=" * 60)
    else:
        print("\nVerification failed, no valid results")


if __name__ == "__main__":
    main()
