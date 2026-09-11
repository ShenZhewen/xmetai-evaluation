"""
Reader 基类和 Catalog 协议

按照 README 第 12 节实现数据源抽象层。
"""

from abc import ABC, abstractmethod

from xmetai_evaluation.core.contracts import (
    DataRequest,
    DataIndex,
    DataBundle,
)


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
