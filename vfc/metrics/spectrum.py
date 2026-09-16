# -*- coding: utf-8 -*-
"""连续量(regression) 指标：纬向 FFT 功率谱（能量谱；球谐升级备忘见原文档）。"""
from __future__ import annotations
import numpy as np
from ._weights import lat_weights

def zonal_spectrum(field, lat, weighted=True, remove_zonal_mean=True):
    """(T, lat, lon) -> (T, k) 纬向功率谱，k=0..nlon//2（k=0 可置零）。"""
    x = np.asarray(field, dtype="f8")
    nlon = x.shape[-1]
    if remove_zonal_mean:
        x = x - x.mean(axis=-1, keepdims=True)
    F = np.fft.rfft(x, axis=-1)
    psd = (F.real ** 2 + F.imag ** 2) / (nlon ** 2)
    if remove_zonal_mean:
        psd[..., 0] = 0.0
    w = lat_weights(lat, weighted)
    return (psd * w[:, None]).sum(axis=-2) / w.sum()

def wavenumber_axis(nlon) -> np.ndarray:
    return np.arange(nlon // 2 + 1)

def wavelength_km(k) -> np.ndarray:
    k = np.asarray(k, dtype="f8")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 0, 40075.0 / np.maximum(k, 1e-9), np.nan)
