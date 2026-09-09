"""
测试 Pipeline 执行器和状态机

验证 README Change 4 要求：
- 状态转换正确
- 缓存机制工作
- 原子写入
- 步骤编排
"""

import pytest
import json
import time
from pathlib import Path
from datetime import datetime

from xmetai_evaluation.pipeline.state import (
    ExecutionState,
    StepStatus,
    StepResult,
    ExecutionResult,
)
from xmetai_evaluation.pipeline.executor import PipelineExecutor
from xmetai_evaluation.core.registry import Registry
from xmetai_evaluation.core.errors import ConfigError


class TestExecutionState:
    """测试执行状态"""

    def test_execution_states_exist(self):
        """测试所有状态定义"""
        assert ExecutionState.PLANNED
        assert ExecutionState.RUNNING
        assert ExecutionState.SUCCEEDED
        assert ExecutionState.FAILED
        assert ExecutionState.CANCELLED

    def test_step_statuses_exist(self):
        """测试步骤状态定义"""
        assert StepStatus.PENDING
        assert StepStatus.RUNNING
        assert StepStatus.COMPLETED
        assert StepStatus.FAILED
        assert StepStatus.SKIPPED
        assert StepStatus.CACHED


class TestStepResult:
    """测试步骤结果"""

    def test_step_result_minimal(self):
        """测试最小步骤结果"""
        result = StepResult(
            step_id="read_forecast",
            status=StepStatus.COMPLETED,
        )

        assert result.step_id == "read_forecast"
        assert result.status == StepStatus.COMPLETED
        assert result.started_at is None
        assert result.cache_hit is False
        assert result.warnings == []

    def test_step_result_with_timing(self):
        """测试带时间信息的步骤结果"""
        start = datetime.now()
        time.sleep(0.01)
        end = datetime.now()

        result = StepResult(
            step_id="compute_metric",
            status=StepStatus.COMPLETED,
            started_at=start,
            completed_at=end,
            duration_seconds=(end - start).total_seconds(),
        )

        assert result.duration_seconds > 0


class TestExecutionResult:
    """测试执行结果"""

    def test_execution_result_minimal(self):
        """测试最小执行结果"""
        result = ExecutionResult(
            protocol_id="test_protocol",
            state=ExecutionState.PLANNED,
            started_at=datetime.now(),
        )

        assert result.protocol_id == "test_protocol"
        assert result.state == ExecutionState.PLANNED
        assert result.steps == {}

    def test_add_step_result(self):
        """测试添加步骤结果"""
        exec_result = ExecutionResult(
            protocol_id="test",
            state=ExecutionState.RUNNING,
            started_at=datetime.now(),
        )

        step1 = StepResult(step_id="step1", status=StepStatus.COMPLETED)
        step2 = StepResult(step_id="step2", status=StepStatus.FAILED, error="test error")

        exec_result.add_step_result(step1)
        exec_result.add_step_result(step2)

        assert len(exec_result.steps) == 2
        assert exec_result.steps["step1"].status == StepStatus.COMPLETED
        assert exec_result.steps["step2"].status == StepStatus.FAILED

    def test_get_failed_steps(self):
        """测试获取失败步骤"""
        exec_result = ExecutionResult(
            protocol_id="test",
            state=ExecutionState.FAILED,
            started_at=datetime.now(),
        )

        exec_result.add_step_result(StepResult(step_id="s1", status=StepStatus.COMPLETED))
        exec_result.add_step_result(StepResult(step_id="s2", status=StepStatus.FAILED))
        exec_result.add_step_result(StepResult(step_id="s3", status=StepStatus.FAILED))

        failed = exec_result.get_failed_steps()
        assert len(failed) == 2
        assert all(s.status == StepStatus.FAILED for s in failed)

    def test_get_cached_steps(self):
        """测试获取缓存步骤"""
        exec_result = ExecutionResult(
            protocol_id="test",
            state=ExecutionState.SUCCEEDED,
            started_at=datetime.now(),
        )

        step1 = StepResult(step_id="s1", status=StepStatus.COMPLETED)
        step2 = StepResult(step_id="s2", status=StepStatus.CACHED, cache_hit=True)

        exec_result.add_step_result(step1)
        exec_result.add_step_result(step2)

        cached = exec_result.get_cached_steps()
        assert len(cached) == 1
        assert cached[0].step_id == "s2"

    def test_summary(self):
        """测试生成摘要"""
        exec_result = ExecutionResult(
            protocol_id="test_summary",
            state=ExecutionState.SUCCEEDED,
            started_at=datetime.now(),
            duration_seconds=10.5,
        )

        exec_result.add_step_result(StepResult(step_id="s1", status=StepStatus.COMPLETED))
        exec_result.add_step_result(StepResult(step_id="s2", status=StepStatus.CACHED, cache_hit=True))
        exec_result.add_step_result(StepResult(step_id="s3", status=StepStatus.FAILED))

        summary = exec_result.summary()

        assert summary["protocol_id"] == "test_summary"
        assert summary["state"] == "succeeded"
        assert summary["total_steps"] == 3
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["cached"] == 1
        assert summary["duration_seconds"] == 10.5


class TestPipelineExecutor:
    """测试 Pipeline 执行器"""

    @pytest.fixture
    def executor(self, tmp_path):
        """创建测试执行器"""
        registry = Registry()
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "outputs"

        return PipelineExecutor(
            registry=registry,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    def test_executor_init(self, executor, tmp_path):
        """测试执行器初始化"""
        assert executor.registry is not None
        assert executor.cache_dir == tmp_path / "cache"
        assert executor.output_dir == tmp_path / "outputs"
        assert executor.cache_dir.exists()
        assert executor.output_dir.exists()

    def test_execute_simple_protocol(self, executor):
        """测试执行简单协议"""
        protocol = {
            "protocol_id": "simple_test",
            "steps": [
                {"id": "step1", "type": "read", "params": {}},
                {"id": "step2", "type": "compute", "params": {}},
            ],
        }

        result = executor.execute(protocol, enable_cache=False)

        assert result.state == ExecutionState.SUCCEEDED
        assert len(result.steps) == 2
        assert result.steps["step1"].status == StepStatus.COMPLETED
        assert result.steps["step2"].status == StepStatus.COMPLETED
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0

    def test_execute_with_no_steps_fails(self, executor):
        """测试无步骤的协议应该失败"""
        protocol = {
            "protocol_id": "empty_test",
            "steps": [],
        }

        result = executor.execute(protocol)

        assert result.state == ExecutionState.FAILED
        assert "no steps" in result.error.lower()

    def test_execute_step_failure_stops_pipeline(self, executor, tmp_path):
        """测试步骤失败应该停止 pipeline"""
        protocol = {
            "protocol_id": "failure_test",
            "steps": [
                {"id": "step1", "type": "read"},
                {"id": "step2", "type": "compute"},
            ],
        }

        # 模拟步骤失败：通过修改执行器的 _run_step 方法
        original_run = executor._run_step

        def mock_run_step(step_config, protocol_config):
            if step_config["id"] == "step1":
                raise RuntimeError("Simulated failure")
            return original_run(step_config, protocol_config)

        executor._run_step = mock_run_step

        result = executor.execute(protocol, enable_cache=False)

        assert result.state == ExecutionState.FAILED
        assert "step1" in result.error.lower()
        # step2 不应该执行
        assert "step2" not in result.steps or result.steps.get("step2") is None

    def test_cache_mechanism(self, executor):
        """测试缓存机制"""
        protocol = {
            "protocol_id": "cache_test",
            "steps": [
                {"id": "cached_step", "type": "read", "params": {"key": "value"}},
            ],
        }

        # 第一次执行
        result1 = executor.execute(protocol, enable_cache=True)
        assert result1.state == ExecutionState.SUCCEEDED
        assert result1.steps["cached_step"].status == StepStatus.COMPLETED
        assert result1.steps["cached_step"].cache_hit is False

        # 第二次执行（应该使用缓存）
        result2 = executor.execute(protocol, enable_cache=True)
        assert result2.state == ExecutionState.SUCCEEDED
        assert result2.steps["cached_step"].status == StepStatus.CACHED
        assert result2.steps["cached_step"].cache_hit is True

    def test_cache_disabled(self, executor):
        """测试禁用缓存"""
        protocol = {
            "protocol_id": "no_cache_test",
            "steps": [
                {"id": "step1", "type": "read"},
            ],
        }

        # 执行两次，都不使用缓存
        result1 = executor.execute(protocol, enable_cache=False)
        result2 = executor.execute(protocol, enable_cache=False)

        assert result1.steps["step1"].cache_hit is False
        assert result2.steps["step1"].cache_hit is False

    def test_fingerprint_computation(self, executor):
        """测试指纹计算"""
        step1 = {"id": "step1", "type": "read", "params": {"a": 1}}
        step2 = {"id": "step1", "type": "read", "params": {"a": 1}}  # 相同
        step3 = {"id": "step1", "type": "read", "params": {"a": 2}}  # 不同

        protocol = {"protocol_id": "test"}

        fp1 = executor._compute_fingerprint(step1, protocol)
        fp2 = executor._compute_fingerprint(step2, protocol)
        fp3 = executor._compute_fingerprint(step3, protocol)

        # 相同配置应该产生相同指纹
        assert fp1 == fp2
        # 不同配置应该产生不同指纹
        assert fp1 != fp3

    def test_atomic_cache_write(self, executor):
        """测试原子写入缓存"""
        fingerprint = "test_fingerprint_12345"
        output_path = executor.output_dir / "test_output.json"
        output_path.write_text("{}")

        # 写入缓存
        executor._write_cache(fingerprint, output_path)

        # 验证缓存文件存在
        cache_file = executor.cache_dir / f"{fingerprint}.json"
        assert cache_file.exists()

        # 验证没有临时文件残留
        temp_file = cache_file.with_suffix(".tmp")
        assert not temp_file.exists()

        # 验证缓存内容
        cache_data = json.loads(cache_file.read_text())
        assert cache_data["fingerprint"] == fingerprint
        assert "output_path" in cache_data
        assert "created_at" in cache_data

    def test_output_directory_creation(self, executor):
        """测试输出目录创建"""
        protocol = {
            "protocol_id": "dir_test",
            "steps": [
                {"id": "step1", "type": "read"},
            ],
        }

        result = executor.execute(protocol, enable_cache=False)

        # 验证输出目录存在
        protocol_output_dir = executor.output_dir / "dir_test"
        assert protocol_output_dir.exists()

        # 验证步骤输出文件存在
        step_output = protocol_output_dir / "step1.json"
        assert step_output.exists()
