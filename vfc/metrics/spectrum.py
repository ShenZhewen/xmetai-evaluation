# -*- coding: utf-8 -*-
"""连续量(regression) 指标：纬向 FFT 谱（zonal）与球谐带功率（spherical）。

两套定义不同，勿混：
- zonal_spectrum / zonal_band_power：沿每个纬圈 1D rFFT（纬向波数谱 / 波数带功率）；
- spherical_band_power：associated Legendre + 经向 rFFT 的球谐带功率，
  等价参考 xmetai.loss.SphericalBandPowerLoss.band_power。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._weights import lat_weights

# 与参考 xmetai ForecastFidelity 保持一致的球谐总阶数分段。
DEFAULT_BANDS = ((1, 4), (5, 20), (21, 40), (41, 64), (65, 128))


def zonal_spectrum(field, lat, weighted=True, remove_zonal_mean=True):
    """(T, lat, lon) -> (T, k) 纬向波数功率谱（zonal_spectrum；非球谐）。

    沿每个纬圈做 1D rFFT，k=0..nlon//2；cos(lat) 加权；k=0(纬向平均)可置零。
    """
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


def zonal_band_power(field, lat, bands=DEFAULT_BANDS,
                     weighted=True, remove_zonal_mean=True):
    """纬向 FFT 谱按波数带求和，返回 (..., nband)。"""
    psd = zonal_spectrum(field, lat, weighted=weighted,
                         remove_zonal_mean=remove_zonal_mean)
    return np.stack(
        [psd[..., int(lo):int(hi) + 1].sum(axis=-1)
         for lo, hi in bands],
        axis=-1,
    )


class _SphericalBandPower:
    """球谐带功率计算器（等价于参考 SphericalBandPowerLoss.band_power）。"""

    def __init__(self, nlat: int, nlon: int, bands: Sequence[tuple[int, int]]):
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.bands = tuple((int(lo), int(hi)) for lo, hi in bands)
        if not self.bands:
            raise ValueError("bands must not be empty")
        if any(lo < 0 or hi < lo for lo, hi in self.bands):
            raise ValueError("invalid spectral band: %r" % (self.bands,))
        self.lmax = self.bands[-1][1]
        n = self.nlat - 1
        if n < 2 or 2 * self.lmax >= n or self.lmax >= self.nlon // 2:
            raise ValueError(
                "grid too small for lmax=%d: nlat=%d nlon=%d"
                % (self.lmax, self.nlat, self.nlon)
            )

        # Clenshaw-Curtis quadrature，与参考实现一致。
        theta = np.arange(self.nlat, dtype=np.float64) * (np.pi / n)
        weights = np.zeros(self.nlat, dtype=np.float64)
        if n % 2 == 0:
            weights[[0, -1]] = 1.0 / (n * n - 1)
        else:
            weights[[0, -1]] = 1.0 / (n * n)
        interior = np.ones(n - 1, dtype=np.float64)
        for k in range(1, (n + 1) // 2):
            interior -= 2.0 * np.cos(2.0 * k * theta[1:-1]) / (4.0 * k * k - 1.0)
        if n % 2 == 0:
            interior -= np.cos(n * theta[1:-1]) / (n * n - 1.0)
        weights[1:-1] = 2.0 * interior / n
        self.latitude_weights = (weights / 2.0).astype(np.float32)

        # 稳定的 associated Legendre 递推。
        basis = np.zeros((self.lmax + 1, self.lmax + 1, self.nlat),
                         dtype=np.float64)
        basis[0, 0] = 1.0 / np.sqrt(4.0 * np.pi)
        x = np.cos(theta)
        sint = np.sin(theta)
        for m in range(self.lmax + 1):
            if m:
                basis[m, m] = (
                    -np.sqrt((2.0 * m + 1.0) / (2.0 * m))
                    * sint * basis[m - 1, m - 1]
                )
            if m < self.lmax:
                basis[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * basis[m, m]
            for degree in range(m + 2, self.lmax + 1):
                a = np.sqrt(
                    (4.0 * degree * degree - 1.0)
                    / (degree * degree - m * m)
                )
                b = np.sqrt(
                    ((degree - 1) ** 2 - m * m)
                    / (4.0 * (degree - 1) ** 2 - 1.0)
                )
                basis[degree, m] = a * (
                    x * basis[degree - 1, m] - b * basis[degree - 2, m]
                )
        self.weighted_basis = (basis * weights).astype(np.float32)
        multiplicity = np.full(self.lmax + 1, 2.0, dtype=np.float32)
        multiplicity[0] = 1.0
        self.multiplicity = multiplicity

    def power(self, field, block: int = 16):
        """(..., nlat, nlon) -> (..., nband) 球谐带能量。"""
        arr = np.asarray(field, dtype=np.float32)
        if arr.ndim < 2:
            raise ValueError("field must have at least 2 dimensions")
        if tuple(arr.shape[-2:]) != (self.nlat, self.nlon):
            raise ValueError(
                "expected field grid (%d, %d), got %r"
                % (self.nlat, self.nlon, tuple(arr.shape[-2:]))
            )
        out_shape = tuple(arr.shape[:-2])
        x = arr.reshape(-1, self.nlat, self.nlon)
        nfield = x.shape[0]
        chunks = []
        blk = max(1, int(block))
        for i0 in range(0, nfield, blk):
            xb = x[i0:i0 + blk]
            mean = (xb.mean(axis=-1) * self.latitude_weights).sum(axis=-1)
            xb = xb - mean[:, None, None]
            # np.fft 无 forward norm，除以 nlon 等价于 norm="forward"。
            fft = np.fft.rfft(xb, axis=-1) / float(self.nlon)
            fft = fft[..., :self.lmax + 1]
            real = np.einsum(
                "lmh,nhm->nlm", self.weighted_basis, fft.real,
                optimize=True,
            )
            imag = np.einsum(
                "lmh,nhm->nlm", self.weighted_basis, fft.imag,
                optimize=True,
            )
            power = (
                (real * real + imag * imag) * self.multiplicity[None, None, :]
            ).sum(axis=-1) * (2.0 * np.pi) ** 2
            bands = np.stack(
                [power[:, lo:hi + 1].sum(axis=-1)
                 for lo, hi in self.bands],
                axis=-1,
            )
            chunks.append(bands)
        result = np.concatenate(chunks, axis=0).reshape(
            out_shape + (len(self.bands),)
        )
        return result


_SPH_CACHE: dict[tuple[int, int, tuple[tuple[int, int], ...]], _SphericalBandPower] = {}


def validate_global_lat(lat, atol: float = 1e-6) -> np.ndarray:
    """校验「全球、等距、含南北极」的纬度网格，返回 theta（弧度）。

    球谐展开只对全球球面有定义：连带勒让德函数只在整个纬圈上正交，
    把纬度截断成区域后 _SphericalBandPower 仍会把它当成"从极到极"的
    球面来算，结果是错的且不会报错。因此这里显式拒绝区域 /
    非等距 / 不含极点的 lat（调用方应降级为只算 zonal）。

    注意：带功率对纬度「行序翻转」是恒等的
    （P_lm(cos(pi-theta)) = (-1)^(l+m) P_lm(cos(theta))，模平方不变），
    所以这里不限制纬度是升序还是降序。

    Parameters
    ----------
    lat : array_like
        纬度，单位**度**、一维；要求 [-90, 90]、等距、首末点为两极。
    atol : float
        浮点容差。

    Returns
    -------
    theta : np.ndarray
        colatitude（弧度）。
    """
    lat = np.asarray(lat, dtype=np.float64)
    if lat.ndim != 1 or lat.size < 3 or not np.all(np.isfinite(lat)):
        raise ValueError("lat 必须是长度>=3 的一维有限数组")
    if np.max(np.abs(lat)) > 90.0 + atol:
        raise ValueError(
            "lat 必须是「度」且 |lat| <= 90；当前 max|lat|=%.6f（可能传了弧度？）"
            % np.max(np.abs(lat)))
    d = np.diff(lat)
    if d.size < 2 or not np.allclose(d, d.mean(), rtol=1e-6, atol=atol):
        raise ValueError("lat 必须是等距网格（dlat 不是常数）")
    global_ends = (
        (abs(lat[0] + 90.0) <= atol and abs(lat[-1] - 90.0) <= atol)
        or (abs(lat[0] - 90.0) <= atol and abs(lat[-1] + 90.0) <= atol)
    )
    if not global_ends:
        raise ValueError(
            "球谐带功率要求全球含极网格（-90..+90）；当前 lat=[%.4f, %.4f]"
            "（n=%d）。区域/bbox 检验请只用 zonal band power。"
            % (lat[0], lat[-1], lat.size))
    return np.deg2rad(90.0 - lat)


def spherical_band_power(field, lat=None, bands=DEFAULT_BANDS, block: int = 16):
    """球谐带功率（spherical band power），返回 (..., nband)。

    与 zonal_spectrum 不同：不是逐纬圈 1D FFT，而是球谐展开的带功率。

    公式与参考 xmetai.loss.SphericalBandPowerLoss.band_power 保持一致：
    去全球面积加权平均、经向 rFFT、associated Legendre basis 展开、
    按球谐总阶数区间求和。

    ``lat`` 是**必需**参数（单位：度），用于校验输入场确实是全球含极
    网格。球谐基函数只在全球球面上正交，区域裁剪（如 bbox 子集）
    得到的不是球谐谱，这里直接抛 ValueError；调用方（regr_pair）
    捕获后降级为只输出 zonal 谱。
    """
    arr = np.asarray(field)
    if arr.ndim < 2:
        raise ValueError("field must have at least 2 dimensions")
    if lat is None:
        raise ValueError(
            "spherical_band_power 需要 lat（度）来校验全球网格；请传 lat=file.lat")
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lat_arr.size != arr.shape[-2]:
        raise ValueError(
            "lat 长度 %d != field 纬度维 %d" % (lat_arr.size, arr.shape[-2]))
    validate_global_lat(lat_arr)          # 区域/非全球 -> raise，调用方降级
    bands0 = tuple((int(lo), int(hi)) for lo, hi in bands)
    key = (int(arr.shape[-2]), int(arr.shape[-1]), bands0)
    obj = _SPH_CACHE.get(key)
    if obj is None:
        obj = _SphericalBandPower(key[0], key[1], bands0)
        _SPH_CACHE[key] = obj
    return obj.power(arr, block=block)


def band_power_arrays(pred, obs, lat, bands=DEFAULT_BANDS, block: int = 16,
                      weighted: bool = True):
    """一次算齐 pred/obs 的纬向与球谐带功率。

    纬向带功率对任何网格都有定义；球谐只对「全球含极」网格有定义，区域/bbox
    裁剪会在这里被 ``spherical_band_power`` 拦下——此时 ``sp``/``so`` 返回
    None 而不是抛错，让调用方降级成只输出 zonal（批量路径与 ``regr_pair``
    都按这个约定走）。

    Returns
    -------
    zp, zo, sp, so : np.ndarray | None
        形状都是 ``(..., nband)``；``sp``/``so`` 见上。
    note : str
        球谐是否可用的说明，写进产物/归档便于事后排查。
    """
    zp = zonal_band_power(pred, lat, bands=bands, weighted=weighted)
    zo = zonal_band_power(obs, lat, bands=bands, weighted=weighted)
    try:
        sp = spherical_band_power(pred, lat=lat, bands=bands, block=block)
        so = spherical_band_power(obs, lat=lat, bands=bands, block=block)
    except ValueError as exc:
        return zp, zo, None, None, "skipped: %s" % exc
    return zp, zo, sp, so, "global: spherical band power computed"


def band_power_frame(pred, obs, lat, leads, bands=DEFAULT_BANDS, block: int = 16,
                     weighted: bool = True, prefix: str = "") -> pd.DataFrame:
    """带功率表：行=lead，列=各频带。

    列名 ``{prefix}{zonal|spherical}_{pred|obs|ratio}_{lo}_{hi}`` 是**归档契约**，
    批量路径的 ``zonal_bands_<date>_<var>.csv`` / ``spherical_bands_<date>_<var>.csv``
    与 ``regr_pair`` 的成对产物共用它，报告端只按这一套名字取数。

    ``prefix`` 只在集合多成员时需要（``"det_"`` 之类）；单成员留空，与
    ``regr_pair`` 的产物逐列一致。
    """
    zp, zo, sp, so, note = band_power_arrays(pred, obs, lat, bands=bands,
                                             block=block, weighted=weighted)
    data = {}
    for i, (lo, hi) in enumerate(bands):
        tag = "%d_%d" % (lo, hi)
        data[prefix + "zonal_pred_" + tag] = zp[..., i]
        data[prefix + "zonal_obs_" + tag] = zo[..., i]
        with np.errstate(divide="ignore", invalid="ignore"):
            data[prefix + "zonal_ratio_" + tag] = zp[..., i] / zo[..., i]
        if sp is not None and so is not None:
            data[prefix + "spherical_pred_" + tag] = sp[..., i]
            data[prefix + "spherical_obs_" + tag] = so[..., i]
            with np.errstate(divide="ignore", invalid="ignore"):
                data[prefix + "spherical_ratio_" + tag] = sp[..., i] / so[..., i]
    frame = pd.DataFrame(data, index=pd.Index(leads, name="lead_h"))
    frame.attrs["spherical_note"] = note
    return frame


def wavenumber_axis(nlon) -> np.ndarray:
    return np.arange(nlon // 2 + 1)


def wavelength_km(k) -> np.ndarray:
    k = np.asarray(k, dtype="f8")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 0, 40075.0 / np.maximum(k, 1e-9), np.nan)
