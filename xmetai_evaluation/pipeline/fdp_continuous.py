"""
FDP 连续场评测主流程

实现 z500 等大气变量的 RMSE / Bias / ACC 评测。

使用方式：
    from xmetai_evaluation.pipeline.fdp_continuous import FDPContinuousEvaluator

    evaluator = FDPContinuousEvaluator(config)
    results = evaluator.run()
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from xmetai_evaluation.io.fengqing_reader import FengqingReader, FengqingCatalog
from xmetai_evaluation.io.cra_reader import CRAReader, CRACatalog
from xmetai_evaluation.core.contracts import DataRequest, ResultBundle
from xmetai_evaluation.pipeline.matcher import Matcher
from xmetai_evaluation.metrics.rmse import RMSE
from xmetai_evaluation.metrics.bias import Bias
from xmetai_evaluation.metrics.acc import ACC
from xmetai_evaluation.core.errors import ConfigError


class FDPContinuousEvaluator:
    """
    FDP 连续场评测器

    支持的指标：
    - RMSE（均方根误差）
    - Bias（平均误差）
    - ACC（距平相关系数，需要气候态）
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 评测配置字典
        """
        self.config = config
        self._validate_config()

        # 初始化 Readers
        self.forecast_reader, self.forecast_catalog = FengqingReader.with_catalog(
            root_dir=Path(config["forecast"]["root"]),
            source_id=config["forecast"].get("reader", "fengqing"),
        )

        self.observation_reader, self.observation_catalog = CRAReader.with_catalog(
            root_dir=Path(config["observation"]["root"]),
            source_id=config["observation"].get("reader", "cra"),
        )

        # 初始化 Matcher
        self.matcher = Matcher(ensemble_reduction="mean")

        # 初始化 Metrics
        self.metrics = self._initialize_metrics()

    def _validate_config(self):
        """校验配置"""
        required_fields = ["forecast", "observation", "metrics", "output"]
        for field in required_fields:
            if field not in self.config:
                raise ConfigError(f"Missing required config field: {field}")

        forecast_required = ["root", "variables", "init_times", "lead_times"]
        for field in forecast_required:
            if field not in self.config["forecast"]:
                raise ConfigError(f"Missing required forecast field: {field}")

        observation_required = ["root", "variables"]
        for field in observation_required:
            if field not in self.config["observation"]:
                raise ConfigError(f"Missing required observation field: {field}")

    def _initialize_metrics(self) -> List:
        """初始化指标"""
        metrics = []
        for metric_config in self.config["metrics"]:
            name = metric_config["name"]
            params = metric_config.get("params", {})

            if name == "rmse":
                metrics.append(RMSE(params))
            elif name == "bias":
                metrics.append(Bias(params))
            elif name == "acc":
                metrics.append(ACC(params))
            else:
                raise ConfigError(f"Unknown metric: {name}")

        return metrics

    def run(self) -> ResultBundle:
        """
        运行评测

        Returns:
            ResultBundle 包含所有指标结果
        """
        print("=" * 80)
        print("FDP Continuous Field Evaluation")
        print("=" * 80)

        # 1. 读取预报数据
        print("\n[1/5] Reading forecast data...")
        forecast_request = DataRequest(
            source_id=self.config["forecast"].get("reader", "fengqing"),
            variables=self.config["forecast"]["variables"],
            init_times=[
                datetime.fromisoformat(t) for t in self.config["forecast"]["init_times"]
            ],
            lead_times=self.config["forecast"]["lead_times"],
        )

        forecast_index = self.forecast_catalog.discover(forecast_request)
        forecast_bundle = self.forecast_reader.read(forecast_request, forecast_index)
        print(f"   Loaded forecast: {forecast_bundle.payload.dims}")

        # 2. 读取观测数据
        print("\n[2/5] Reading observation data...")
        # 计算所需的观测时间（所有 valid_time）
        observation_times = self._compute_observation_times(
            forecast_request.init_times,
            forecast_request.lead_times,
        )

        observation_request = DataRequest(
            source_id=self.config["observation"].get("reader", "cra"),
            variables=self.config["observation"]["variables"],
            init_times=observation_times,  # 用作 valid_times
        )

        observation_index = self.observation_catalog.discover(observation_request)
        observation_bundle = self.observation_reader.read(observation_request, observation_index)
        print(f"   Loaded observation: {observation_bundle.payload.dims}")

        # 3. 配对数据
        print("\n[3/5] Matching forecast and observation...")
        batches = self.matcher.match(
            forecast=forecast_bundle,
            observation=observation_bundle,
            variables=self.config["forecast"]["variables"],
        )
        print(f"   Created {len(batches)} evaluation batches")

        # 4. 计算指标
        print("\n[4/5] Computing metrics...")
        all_results = []

        for metric in self.metrics:
            print(f"   Computing {metric.name}...")
            metric_results = []

            for batch in batches:
                # Accumulate
                state = metric.accumulate(batch)
                # Finalize immediately (no merging for simplicity)
                result = metric.finalize(state)
                metric_results.append(result)

            all_results.extend(metric_results)
            print(f"      {metric.name}: {len(metric_results)} results")

        # 5. 输出结果
        print("\n[5/5] Writing results...")
        output_path = self._write_results(all_results)
        print(f"   Results written to: {output_path}")

        print("\n" + "=" * 80)
        print("Evaluation complete!")
        print("=" * 80)

        # 构建 ResultBundle
        result_bundle = ResultBundle(
            run_id=self.config.get("name", "fdp_continuous"),
            results=all_results,
            manifest={
                "forecast": forecast_bundle.provenance.__dict__,
                "observation": observation_bundle.provenance.__dict__,
                "n_batches": len(batches),
            },
            resolved_config=self.config,
        )

        return result_bundle

    def _compute_observation_times(
        self,
        init_times: List[datetime],
        lead_times: List[int],
    ) -> List[datetime]:
        """计算所需的观测时间"""
        from datetime import timedelta

        observation_times = set()
        for init_time in init_times:
            for lead_hours in lead_times:
                valid_time = init_time + timedelta(hours=lead_hours)
                observation_times.add(valid_time)

        return sorted(observation_times)

    def _write_results(self, results: List) -> Path:
        """写入结果到 JSON"""
        output_config = self.config["output"]
        output_path = Path(output_config["path"])

        # 替换 {timestamp}
        if "{timestamp}" in str(output_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(str(output_path).replace("{timestamp}", timestamp))

        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 序列化结果
        results_data = []
        for result in results:
            result_dict = {
                "metric_name": result.metric_name,
                "metric_version": result.metric_version,
                "value": result.value,
                "status": result.status.value,
                "n_requested": result.n_requested,
                "n_valid": result.n_valid,
                "weights_sum": result.weights_sum,
                "aggregation": result.aggregation,
                "warnings": result.warnings,
            }
            results_data.append(result_dict)

        # 写入 JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": self.config,
                    "results": results_data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return output_path


def run_evaluation(config_path: Path) -> ResultBundle:
    """
    从配置文件运行评测

    Args:
        config_path: 配置文件路径（YAML 或 JSON）

    Returns:
        ResultBundle
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    evaluator = FDPContinuousEvaluator(config)
    return evaluator.run()
