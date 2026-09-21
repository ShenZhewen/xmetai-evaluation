"""
Reader 基类和 Catalog 协议

按照 README 第 12 节实现数据源抽象层。
"""

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Iterator

from xmetai_evaluation.core.contracts import (
    DataRequest,
    DataIndex,
    DataBundle,
)

#: 底层走 libhdf5 的后端。**HDF5 不是线程安全的**——多个线程同时打开/读取
#: 会破坏它的进程级内部状态：轻则读属性报 ``NetCDF: Can't open HDF5
#: attribute`` 这种看着像文件损坏的假错，重则堆损坏直接段错误
#: （实测：fengqing 目录下 4 线程并发 open 不同的 .nc，第 3 轮就
#: ``malloc(): unaligned fastbin chunk detected`` core dump；同样这 8 个
#: 文件顺序开 5 遍则一次都不出错）。cfgrib 走 eccodes、zarr 自管块索引，
#: 都不经 HDF5，所以只有这两个后端要排队。
_HDF5_ENGINES = frozenset({"netcdf", "h5netcdf"})

#: 进程内唯一的 HDF5 访问锁。HDF5 的状态本来就是进程级的，所以只有一把，
#: 不按文件/数据源细分。
_HDF5_LOCK = threading.Lock()


def uses_hdf5(engine: Optional[str], engine_fallback: Optional[str] = None) -> bool:
    """该读取器是否可能用 HDF5 系后端打开文件（含回退后端）。"""
    return (
        str(engine or "").strip() in _HDF5_ENGINES
        or str(engine_fallback or "").strip() in _HDF5_ENGINES
    )


@contextmanager
def hdf5_guard(
    engine: Optional[str], engine_fallback: Optional[str] = None
) -> Iterator[None]:
    """HDF5 系后端的**串行**访问窗口。

    ``threads`` 形态下工作块是并发跑的，文件访问必须排队；只包住
    "打开-读取-关闭"这一小段，归一化/换算/指标计算仍在锁外并发，所以
    并发收益没有全丢。``serial`` / ``processes`` 形态下没有竞争（进程各有
    各的 HDF5 状态），这把锁不产生额外代价。

    非 HDF5 后端（cfgrib / zarr）直接放行——不要为了"保险"把它们也串行化。
    """
    if not uses_hdf5(engine, engine_fallback):
        yield
        return
    with _HDF5_LOCK:
        yield


class DataCatalog(ABC):
    """
    数据目录协议

    按照 README 第 12.1 节：Catalog 负责文件发现，不执行 I/O。
    """

    @abstractmethod
    def discover(self, request: DataRequest) -> DataIndex:
        """
        发现符合请求的文件

        Args:
            request: 数据请求

        Returns:
            DataIndex（文件清单）

        Raises:
            DiscoveryError: 文件缺失、路径歧义等
        """
        pass


class Reader(ABC):
    """
    数据读取器基类

    按照 README 第 12.2 节：Reader 负责解码、单位转换、归一化。
    """

    def __init__(self, source_id: str, version: str):
        """
        Args:
            source_id: 数据源标识
            version: Reader 版本
        """
        self.source_id = source_id
        self.version = version

    @abstractmethod
    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        """
        读取数据

        Args:
            request: 数据请求
            index: 文件索引（来自 Catalog.discover）

        Returns:
            DataBundle

        Raises:
            DecodeError: 文件损坏、格式错误
            ContractError: 单位/维度不符
        """
        pass
