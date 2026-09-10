"""
网格插值 Transform

将预报场插值到观测场的网格，支持：
- 双线性插值（默认）
- 最近邻插值
- 保守重映射（需要额外库）

按照 README 第 13.3 节：Transform 是可复用的数据运算，输入和输出是 DataBundle。
"""

from typing import Dict, Any, Optional
import xarray as xr
import numpy as np

from xmetai_evaluation.core.contracts import DataBundle
from xmetai_evaluation.core.errors import ContractError


def regrid_to_target(
    source: DataBundle,
    target: DataBundle,
    method: str = "linear",
    drop_source_coords: bool = True,
) -> DataBundle:
    """
    将源数据插值到目标网格

    Args:
        source: 源数据（通常是预报）
        target: 目标数据（通常是观测），提供目标网格
        method: 插值方法（'linear', 'nearest'）
        drop_source_coords: 是否丢弃源坐标（默认 True）

    Returns:
        插值后的 DataBundle（网格与 target 一致）
    """
    source_ds = source.payload if isinstance(source.payload, xr.Dataset) else source.payload.to_dataset()
    target_ds = target.payload if isinstance(target.payload, xr.Dataset) else target.payload.to_dataset()

    # 检查坐标
    if "lat" not in target_ds.coords or "lon" not in target_ds.coords:
        raise ContractError("Target must have 'lat' and 'lon' coordinates")

    if "lat" not in source_ds.coords or "lon" not in source_ds.coords:
        raise ContractError("Source must have 'lat' and 'lon' coordinates")

    # 获取目标网格
    target_lat = target_ds.coords["lat"]
    target_lon = target_ds.coords["lon"]

    # 执行插值
    regridded_ds = source_ds.interp(
        lat=target_lat,
        lon=target_lon,
        method=method,
        kwargs={"fill_value": None},  # 保留 NaN
    )

    # 更新元数据
    new_provenance = source.provenance
    new_provenance.transform_chain = source.provenance.transform_chain + [
        f"regrid_{method}_to_{target.source_id}"
    ]

    return DataBundle(
        payload=regridded_ds,
        kind=source.kind,
        source_id=source.source_id,
        standard_vars=source.standard_vars,
        semantic=source.semantic,
        provenance=new_provenance,
        quality=source.quality,
    )


def compute_ensemble_mean(bundle: DataBundle, member_dim: str = "member") -> DataBundle:
    """
    计算集合均值

    Args:
        bundle: 包含集合成员的数据
        member_dim: 成员维度名（默认 'member'）

    Returns:
        集合均值 DataBundle（不含 member 维度）
    """
    ds = bundle.payload if isinstance(bundle.payload, xr.Dataset) else bundle.payload.to_dataset()

    if member_dim not in ds.dims:
        # 已经是确定性场，直接返回
        return bundle

    # 计算均值
    mean_ds = ds.mean(dim=member_dim)

    # 更新元数据
    new_semantic = bundle.semantic
    new_semantic.member_count = None  # 不再有成员

    new_provenance = bundle.provenance
    new_provenance.transform_chain = bundle.provenance.transform_chain + [
        f"ensemble_mean_over_{member_dim}"
    ]

    return DataBundle(
        payload=mean_ds,
        kind=bundle.kind,
        source_id=bundle.source_id,
        standard_vars=bundle.standard_vars,
        semantic=new_semantic,
        provenance=new_provenance,
        quality=bundle.quality,
    )


def compute_latitude_weights(ds: xr.Dataset) -> xr.DataArray:
    """
    计算纬度余弦权重

    Args:
        ds: 包含 'lat' 坐标的数据集

    Returns:
        权重 DataArray，维度与 (lat, lon) 一致
    """
    if "lat" not in ds.coords:
        raise ContractError("Dataset must have 'lat' coordinate")

    lat = ds.coords["lat"]
    weights = np.cos(np.deg2rad(lat))

    # 扩展到 (lat, lon)
    if "lon" in ds.coords:
        weights, _ = xr.broadcast(weights, ds.coords["lon"])

    return weights
