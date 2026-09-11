"""
网格插值 Transform

将预报场插值到观测场的网格，支持：
- 双线性插值（默认）
- 最近邻插值
- 保守重映射（需要额外库）

按照 README 第 13.3 节：Transform 是可复用的数据运算，输入和输出是 DataBundle。
"""

import xarray as xr
import numpy as np

from xmetai_evaluation.core.contracts import DataBundle
from xmetai_evaluation.core.errors import ContractError


def compute_ensemble_mean(
    bundle: DataBundle, member_dim: str = "member", skipna: bool = False
) -> DataBundle:
    """
    计算集合均值

    Args:
        bundle: 包含集合成员的数据
        member_dim: 成员维度名（默认 'member'）
        skipna: 是否跳过缺测成员。默认 False：任一成员缺测则整点缺测，
            与参考实现 ``members.mean(axis=0)`` 的传播语义一致，避免"缺员还当有效"。

    Returns:
        集合均值 DataBundle（不含 member 维度）
    """
    ds = bundle.payload if isinstance(bundle.payload, xr.Dataset) else bundle.payload.to_dataset()

    if member_dim not in ds.dims:
        # 已经是确定性场，直接返回
        return bundle

    # 计算均值
    mean_ds = ds.mean(dim=member_dim, skipna=skipna)

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


class EnsembleMeanTransform:
    """集合均值变换：输入和输出都是 DataBundle。

    连续场评测在配对前先降到集合均值；确定性场原样返回。
    """

    def __init__(self, member_dim: str = "member"):
        self.member_dim = member_dim

    def transform(self, bundle: DataBundle) -> DataBundle:
        return compute_ensemble_mean(bundle, member_dim=self.member_dim)


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
