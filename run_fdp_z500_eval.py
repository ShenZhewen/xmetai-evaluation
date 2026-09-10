#!/usr/bin/env python
"""
FDP z500 评测运行脚本

使用方法：
    python run_fdp_z500_eval.py [config_path]

默认使用 configs/fdp_continuous_z500.yaml
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from xmetai_evaluation.pipeline.fdp_continuous import run_evaluation


def main():
    """主函数"""
    # 获取配置文件路径
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = project_root / "configs" / "fdp_continuous_z500.yaml"

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Using config: {config_path}")
    print()

    try:
        # 运行评测
        result_bundle = run_evaluation(config_path)

        # 打印摘要
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        for result in result_bundle.results:
            print(f"\n{result.metric_name.upper()}:")
            print(f"  Value: {result.value:.6f}")
            print(f"  Status: {result.status.value}")
            print(f"  Valid points: {result.n_valid}/{result.n_requested}")
            if result.warnings:
                print(f"  Warnings: {', '.join(result.warnings)}")

        print("\n" + "=" * 80)
        print("Evaluation completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
