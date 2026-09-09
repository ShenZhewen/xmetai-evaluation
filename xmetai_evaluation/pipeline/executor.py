"""
Pipeline 执行器

按照 README 第 14 节实现最小执行流程：
- 状态转换
- 步骤编排
- 缓存检查
- 原子写入

Change 4 只实现骨架，不包含真实 Reader、真实指标或报告系统。
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import time

from xmetai_evaluation.pipeline.state import (
    ExecutionState,
    StepStatus,
    StepResult,
    ExecutionResult,
)
from xmetai_evaluation.core.errors import EvaluationError, ConfigError
from xmetai_evaluation.core.registry import Registry


class PipelineExecutor:
    """
    Pipeline 执行器

    按照 README 第 14 节：
    - 解析协议配置
    - 按依赖顺序执行步骤
    - 检查缓存
    - 原子写入结果
    - 管理状态转换
    """

    def __init__(
        self,
        registry: Registry,
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Args:
            registry: 组件注册表
            cache_dir: 缓存目录
            output_dir: 输出目录
        """
        self.registry = registry
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")

        # 确保目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        protocol_config: Dict[str, Any],
        enable_cache: bool = True,
    ) -> ExecutionResult:
        """
        执行评测协议

        Args:
            protocol_config: 协议配置
            enable_cache: 是否启用缓存

        Returns:
            ExecutionResult
        """
        protocol_id = protocol_config.get("protocol_id", "unknown")

        # 初始化结果
        result = ExecutionResult(
            protocol_id=protocol_id,
            state=ExecutionState.PLANNED,
            started_at=datetime.now(),
            output_dir=self.output_dir / protocol_id,
        )

        try:
            # 状态转换：PLANNED → RUNNING
            result.state = ExecutionState.RUNNING

            # 解析步骤
            steps = protocol_config.get("steps", [])
            if not steps:
                raise ConfigError("Protocol has no steps")

            # 执行步骤
            for step_config in steps:
                step_result = self._execute_step(
                    step_config,
                    protocol_config,
                    enable_cache=enable_cache,
                )
                result.add_step_result(step_result)

                # 如果步骤失败，停止执行
                if step_result.status == StepStatus.FAILED:
                    result.state = ExecutionState.FAILED
                    result.error = f"Step '{step_result.step_id}' failed: {step_result.error}"
                    break

            # 如果所有步骤完成，标记为成功
            if result.state == ExecutionState.RUNNING:
                result.state = ExecutionState.SUCCEEDED

        except Exception as e:
            result.state = ExecutionState.FAILED
            result.error = str(e)

        finally:
            # 记录完成时间
            result.completed_at = datetime.now()
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

        return result

    def _execute_step(
        self,
        step_config: Dict[str, Any],
        protocol_config: Dict[str, Any],
        enable_cache: bool,
    ) -> StepResult:
        """
        执行单个步骤

        Args:
            step_config: 步骤配置
            protocol_config: 协议配置（提供上下文）
            enable_cache: 是否启用缓存

        Returns:
            StepResult
        """
        step_id = step_config.get("id", "unknown")
        step_type = step_config.get("type", "unknown")

        # 初始化结果
        step_result = StepResult(
            step_id=step_id,
            status=StepStatus.PENDING,
            started_at=datetime.now(),
        )

        try:
            step_result.status = StepStatus.RUNNING

            # 计算输入指纹
            fingerprint = self._compute_fingerprint(step_config, protocol_config)
            step_result.fingerprint = fingerprint

            # 检查缓存
            if enable_cache:
                cached_output = self._check_cache(fingerprint)
                if cached_output:
                    step_result.status = StepStatus.CACHED
                    step_result.cache_hit = True
                    step_result.output_path = cached_output
                    step_result.completed_at = datetime.now()
                    step_result.duration_seconds = 0.0
                    return step_result

            # 实际执行（Change 4 只是模拟）
            output_path = self._run_step(step_config, protocol_config)
            step_result.output_path = output_path

            # 写入缓存
            if enable_cache and output_path:
                self._write_cache(fingerprint, output_path)

            # 标记完成
            step_result.status = StepStatus.COMPLETED

        except Exception as e:
            step_result.status = StepStatus.FAILED
            step_result.error = str(e)

        finally:
            step_result.completed_at = datetime.now()
            if step_result.started_at:
                step_result.duration_seconds = (
                    step_result.completed_at - step_result.started_at
                ).total_seconds()

        return step_result

    def _compute_fingerprint(
        self,
        step_config: Dict[str, Any],
        protocol_config: Dict[str, Any],
    ) -> str:
        """
        计算步骤输入指纹

        按照 README 第 14 节：指纹包括步骤配置、组件版本、依赖输出。

        Args:
            step_config: 步骤配置
            protocol_config: 协议配置

        Returns:
            十六进制指纹字符串
        """
        # 提取关键信息
        fingerprint_data = {
            "step_id": step_config.get("id"),
            "step_type": step_config.get("type"),
            "step_params": step_config.get("params", {}),
            "component_version": step_config.get("component_version"),
            # 简化：不递归依赖
        }

        # 序列化为 JSON（排序键以确保一致性）
        fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)

        # 计算 SHA256
        return hashlib.sha256(fingerprint_json.encode()).hexdigest()

    def _check_cache(self, fingerprint: str) -> Optional[Path]:
        """
        检查缓存是否存在

        Args:
            fingerprint: 输入指纹

        Returns:
            缓存路径，如果存在；否则 None
        """
        cache_file = self.cache_dir / f"{fingerprint}.json"
        if cache_file.exists():
            return cache_file
        return None

    def _write_cache(self, fingerprint: str, output_path: Path) -> None:
        """
        写入缓存

        按照 README 第 14 节：使用原子写入（写临时文件，然后重命名）。

        Args:
            fingerprint: 输入指纹
            output_path: 输出路径
        """
        cache_file = self.cache_dir / f"{fingerprint}.json"
        temp_file = cache_file.with_suffix(".tmp")

        try:
            # 写入临时文件
            with open(temp_file, "w") as f:
                json.dump(
                    {
                        "fingerprint": fingerprint,
                        "output_path": str(output_path),
                        "created_at": datetime.now().isoformat(),
                    },
                    f,
                )

            # 原子重命名
            temp_file.replace(cache_file)

        except Exception:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _run_step(
        self,
        step_config: Dict[str, Any],
        protocol_config: Dict[str, Any],
    ) -> Path:
        """
        运行步骤（Change 4 模拟实现）

        Args:
            step_config: 步骤配置
            protocol_config: 协议配置

        Returns:
            输出路径
        """
        step_id = step_config.get("id", "unknown")
        step_type = step_config.get("type", "unknown")

        # 模拟执行时间
        time.sleep(0.01)

        # 创建模拟输出
        output_dir = self.output_dir / protocol_config.get("protocol_id", "unknown")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{step_id}.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "step_id": step_id,
                    "step_type": step_type,
                    "status": "completed",
                    "simulated": True,
                },
                f,
            )

        return output_path
