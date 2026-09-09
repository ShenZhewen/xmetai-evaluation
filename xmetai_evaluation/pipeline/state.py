"""
Pipeline 执行状态机

按照 README 第 14 节实现状态转换和结果追踪。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class ExecutionState(Enum):
    """
    执行状态

    状态转换图（README 第 14 节）：
    PLANNED → RUNNING → {SUCCEEDED, FAILED, CANCELLED}
    """

    PLANNED = "planned"  # 已规划，未开始
    RUNNING = "running"  # 执行中
    SUCCEEDED = "succeeded"  # 成功完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 用户取消


class StepStatus(Enum):
    """步骤状态"""

    PENDING = "pending"  # 等待执行
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 完成
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 跳过（依赖失败）
    CACHED = "cached"  # 使用缓存


@dataclass
class StepResult:
    """
    单步执行结果

    记录步骤执行的详细信息，用于追踪和调试。
    """

    step_id: str
    status: StepStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_path: Optional[Path] = None
    fingerprint: Optional[str] = None  # 输入指纹，用于缓存
    cache_hit: bool = False
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """
    Pipeline 执行结果

    按照 README 第 14 节：记录状态、步骤结果、输出路径、错误信息。
    """

    protocol_id: str
    state: ExecutionState
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    steps: Dict[str, StepResult] = field(default_factory=dict)  # step_id -> StepResult
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step_result(self, result: StepResult) -> None:
        """添加步骤结果"""
        self.steps[result.step_id] = result

    def get_failed_steps(self) -> List[StepResult]:
        """获取所有失败的步骤"""
        return [s for s in self.steps.values() if s.status == StepStatus.FAILED]

    def get_completed_steps(self) -> List[StepResult]:
        """获取所有完成的步骤"""
        return [s for s in self.steps.values() if s.status == StepStatus.COMPLETED]

    def get_cached_steps(self) -> List[StepResult]:
        """获取所有使用缓存的步骤"""
        return [s for s in self.steps.values() if s.cache_hit]

    def summary(self) -> Dict[str, Any]:
        """生成执行摘要"""
        total_steps = len(self.steps)
        completed = len(self.get_completed_steps())
        failed = len(self.get_failed_steps())
        cached = len(self.get_cached_steps())
        skipped = len([s for s in self.steps.values() if s.status == StepStatus.SKIPPED])

        return {
            "protocol_id": self.protocol_id,
            "state": self.state.value,
            "duration_seconds": self.duration_seconds,
            "total_steps": total_steps,
            "completed": completed,
            "failed": failed,
            "cached": cached,
            "skipped": skipped,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "has_error": self.error is not None,
            "warnings_count": len(self.warnings),
        }
