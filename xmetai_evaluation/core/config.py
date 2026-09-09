"""
配置加载与校验

按照 README 第 15 节实现配置解析、schema 版本检查、环境变量解析、
路径解析和 resolved config 输出。
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from copy import deepcopy

from xmetai_evaluation.core.errors import ConfigError


class ConfigLoader:
    """
    配置加载器

    按照 README 第 15.1 节：
    - 所有配置顶层必须有 schema_version
    - 配置结构变更时递增版本
    - 加载器只负责支持明确列出的版本
    """

    SUPPORTED_SCHEMA_VERSIONS = [1]

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Args:
            base_dir: 配置文件基础目录，用于解析相对路径
        """
        self.base_dir = base_dir or Path.cwd()

    def load(self, config_path: Path) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            解析后的配置字典

        Raises:
            ConfigError: 如果文件不存在或格式错误
        """
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Cannot read config file {config_path}: {e}")

        if not isinstance(config, dict):
            raise ConfigError(f"Config must be a dictionary, got {type(config).__name__}")

        # 验证 schema_version
        if "schema_version" not in config:
            raise ConfigError(
                f"Config file {config_path} missing required field 'schema_version'. "
                f"All configs must declare their schema version."
            )

        schema_version = config["schema_version"]
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise ConfigError(
                f"Unsupported schema_version {schema_version} in {config_path}. "
                f"Supported versions: {self.SUPPORTED_SCHEMA_VERSIONS}"
            )

        # 保存配置文件路径，用于后续相对路径解析
        config["_config_file"] = str(config_path.resolve())

        return config

    def resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析环境变量

        按照 README 第 7.1 节：路径支持环境变量

        Args:
            config: 配置字典

        Returns:
            解析后的配置字典

        Raises:
            ConfigError: 如果引用的环境变量不存在
        """
        resolved = deepcopy(config)
        self._resolve_env_recursive(resolved, path=[], root=resolved)
        return resolved

    def _resolve_env_recursive(self, obj: Any, path: List[str], root: Any = None) -> Any:
        """递归解析环境变量，返回解析后的值"""
        if root is None:
            root = obj

        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._resolve_env_recursive(value, path + [key], root)
            return obj
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._resolve_env_recursive(item, path + [f"[{i}]"], root)
            return obj
        elif isinstance(obj, str):
            # 检查是否包含环境变量引用 ${VAR}
            if "${" in obj:
                resolved_str = obj
                import re

                env_vars = re.findall(r"\$\{([^}]+)\}", obj)
                for var_name in env_vars:
                    if var_name not in os.environ:
                        path_str = ".".join(path)
                        raise ConfigError(
                            f"Environment variable '${{{var_name}}}' not found "
                            f"(referenced at {path_str})"
                        )
                    resolved_str = resolved_str.replace(
                        f"${{{var_name}}}", os.environ[var_name]
                    )
                return resolved_str
            return obj
        else:
            return obj

    def resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析相对路径

        按照 README 第 7.1 节：配置内相对路径相对于所属配置文件解析，
        解析后的绝对路径进入运行快照。

        Args:
            config: 配置字典

        Returns:
            解析后的配置字典
        """
        if "_config_file" not in config:
            # 如果没有配置文件信息，使用 base_dir
            config_dir = self.base_dir
        else:
            config_dir = Path(config["_config_file"]).parent

        resolved = deepcopy(config)
        self._resolve_paths_recursive(resolved, config_dir, path=[])
        return resolved

    def _resolve_paths_recursive(self, obj: Any, config_dir: Path, path: List[str]) -> Any:
        """递归解析路径，返回解析后的值"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    # 对于字符串值，检查是否看起来像路径（包含 / 或 .）且不是环境变量
                    if ("/" in value or value.startswith(".")) and not value.startswith("${"):
                        # 常见的路径字段名，或值看起来像路径
                        if key in ["path", "root", "file", "dir", "directory", "input", "output"] or "/" in value or value.startswith("."):
                            p = Path(value)
                            if not p.is_absolute():
                                obj[key] = str((config_dir / p).resolve())
                elif isinstance(value, (dict, list)):
                    obj[key] = self._resolve_paths_recursive(value, config_dir, path + [key])
            return obj
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._resolve_paths_recursive(item, config_dir, path + [f"[{i}]"])
            return obj
        else:
            return obj

    def validate_required_fields(self, config: Dict[str, Any], required: List[str]) -> None:
        """
        验证必填字段

        Args:
            config: 配置字典
            required: 必填字段列表

        Raises:
            ConfigError: 如果缺少必填字段
        """
        missing = []
        for field in required:
            if field not in config:
                missing.append(field)

        if missing:
            raise ConfigError(
                f"Missing required fields: {', '.join(missing)}. "
                f"Available fields: {', '.join(config.keys())}"
            )

    def merge_with_defaults(
        self, config: Dict[str, Any], defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合并默认配置

        按照 README 第 15.2 节优先级：defaults < 被引用配置 < task配置 < 命令行

        Args:
            config: 用户配置
            defaults: 默认配置

        Returns:
            合并后的配置
        """
        merged = deepcopy(defaults)
        self._deep_merge(merged, config)
        return merged

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """深度合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


def load_and_resolve_config(config_path: Path) -> Dict[str, Any]:
    """
    便捷函数：加载并完整解析配置

    Args:
        config_path: 配置文件路径

    Returns:
        完整解析后的配置

    Raises:
        ConfigError: 如果配置无效
    """
    loader = ConfigLoader(base_dir=config_path.parent)
    config = loader.load(config_path)
    config = loader.resolve_env_vars(config)
    config = loader.resolve_paths(config)
    return config
