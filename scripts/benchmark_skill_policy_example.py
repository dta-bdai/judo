#!/usr/bin/env python3
"""Example script showing how to use the skill policy optimizer in benchmarks.

This demonstrates how to:
1. Load an ONNX model for the skill policy
2. Configure the SkillPolicyOptimizer
3. Run benchmarks comparing skill policy vs sample-based optimizers
"""

import onnxruntime
import numpy as np

from judo.controller import Controller, ControllerConfig
from judo.optimizers import SkillPolicyConfig, get_registered_optimizers
from judo.tasks import get_registered_tasks


def load_onnx_session(onnx_path: str) -> onnxruntime.InferenceSession:
    """Load an ONNX model for the skill policy.

    Args:
        onnx_path: Path to the ONNX model file

    Returns:
        ONNX runtime inference session
    """
    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    return session


def setup_skill_policy_benchmark(
    task_name: str,
    onnx_model_path: str,
    target_pos: np.ndarray | None = None,
    locomotion_only: bool = False,
) -> tuple:
    """Set up a benchmark using the skill policy optimizer.

    Args:
        task_name: Name of the registered task (e.g., "spot_barbell")
        onnx_model_path: Path to the ONNX skill policy model
        target_pos: Target position for the task (xyz in world frame)
        locomotion_only: Whether to use locomotion only (no arm commands)

    Returns:
        Tuple of (task, optimizer, controller, configs)
    """
    # Load the ONNX model
    print(f"Loading ONNX model from: {onnx_model_path}")
    onnx_session = load_onnx_session(onnx_model_path)

    # Get the task
    registered_tasks = get_registered_tasks()
    if task_name not in registered_tasks:
        raise ValueError(f"Task '{task_name}' not found. Available: {list(registered_tasks.keys())}")

    task_cls, task_config_cls = registered_tasks[task_name]
    task = task_cls()
    task_cfg = task_config_cls()

    # Set up default target position if not provided
    if target_pos is None:
        target_pos = np.array([0.0, 0.0, 0.5])  # Default target

    # Configure the skill policy optimizer
    skill_policy_cfg = SkillPolicyConfig(
        onnx_session=onnx_session,
        target_cmd=target_pos,
        locomotion_only=locomotion_only,
        num_rollouts=1,  # Skill policy only needs 1 "rollout"
        num_nodes=4,     # Can be adjusted based on horizon
        cutoff_time=0.2,
    )

    # Get optimizer class
    registered_optimizers = get_registered_optimizers()
    optimizer_cls, _ = registered_optimizers["skill_policy"]
    optimizer = optimizer_cls(skill_policy_cfg, task.nu)

    # Create controller config
    controller_cfg = ControllerConfig(
        max_opt_iters=1,  # Skill policy only needs 1 iteration (no optimization loop)
        horizon=1.0,
    )

    # Create controller
    controller = Controller(
        task=task,
        task_cfg=task_cfg,
        optimizer=optimizer,
        optimizer_cfg=skill_policy_cfg,
        controller_cfg=controller_cfg,
    )

    return task, optimizer, controller, (task_cfg, skill_policy_cfg, controller_cfg)


def benchmark_skill_policy_vs_sampling(
    task_name: str,
    onnx_model_path: str,
    sample_optimizer_name: str = "mppi",
    num_samples: int = 10,
):
    """Run a comparison benchmark between skill policy and a sampling-based optimizer.

    Args:
        task_name: Name of the task to benchmark
        onnx_model_path: Path to the skill policy ONNX model
        sample_optimizer_name: Name of sampling optimizer to compare against (e.g., "mppi", "cem")
        num_samples: Number of benchmark samples to collect
    """
    import time

    print(f"\n{'='*60}")
    print(f"Benchmarking: {task_name}")
    print(f"Skill Policy vs {sample_optimizer_name.upper()}")
    print(f"{'='*60}\n")

    # Set up skill policy
    task_skill, opt_skill, ctrl_skill, _ = setup_skill_policy_benchmark(
        task_name=task_name,
        onnx_model_path=onnx_model_path,
    )

    # Set up sampling optimizer for comparison
    registered_tasks = get_registered_tasks()
    registered_optimizers = get_registered_optimizers()

    task_cls, task_config_cls = registered_tasks[task_name]
    optimizer_cls, optimizer_config_cls = registered_optimizers[sample_optimizer_name]

    task_sample = task_cls()
    task_cfg_sample = task_config_cls()
    opt_cfg_sample = optimizer_config_cls()
    opt_sample = optimizer_cls(opt_cfg_sample, task_sample.nu)

    controller_cfg_sample = ControllerConfig(max_opt_iters=10, horizon=1.0)
    ctrl_sample = Controller(
        task=task_sample,
        task_cfg=task_cfg_sample,
        optimizer=opt_sample,
        optimizer_cfg=opt_cfg_sample,
        controller_cfg=controller_cfg_sample,
    )

    # Run benchmarks
    skill_times = []
    sample_times = []

    print(f"Running {num_samples} benchmark iterations...\n")

    for i in range(num_samples):
        # Reset tasks
        task_skill.reset()
        task_sample.reset()

        # Get initial states
        state_skill = np.concatenate([task_skill.data.qpos, task_skill.data.qvel])
        state_sample = np.concatenate([task_sample.data.qpos, task_sample.data.qvel])

        # Benchmark skill policy
        start = time.perf_counter()
        ctrl_skill.update_action(state_skill, 0.0)
        skill_time = time.perf_counter() - start
        skill_times.append(skill_time)

        # Benchmark sampling optimizer
        start = time.perf_counter()
        ctrl_sample.update_action(state_sample, 0.0)
        sample_time = time.perf_counter() - start
        sample_times.append(sample_time)

        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{num_samples} iterations")

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nSkill Policy:")
    print(f"  Mean:   {np.mean(skill_times)*1000:.2f} ms")
    print(f"  Median: {np.median(skill_times)*1000:.2f} ms")
    print(f"  Std:    {np.std(skill_times)*1000:.2f} ms")

    print(f"\n{sample_optimizer_name.upper()}:")
    print(f"  Mean:   {np.mean(sample_times)*1000:.2f} ms")
    print(f"  Median: {np.median(sample_times)*1000:.2f} ms")
    print(f"  Std:    {np.std(sample_times)*1000:.2f} ms")

    speedup = np.mean(sample_times) / np.mean(skill_times)
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"{'='*60}\n")


def main():
    """Example usage of skill policy in benchmarks."""

    # Example 1: Basic setup
    print("Example 1: Basic Skill Policy Setup")
    print("-" * 60)

    # NOTE: Replace with actual path to your ONNX model
    onnx_path = "/path/to/your/skill_policy_model.onnx"

    # Check if ONNX file exists
    import os
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found at: {onnx_path}")
        print("\nTo use this script:")
        print("1. Train a skill policy and export to ONNX format")
        print("2. Update 'onnx_path' variable with the path to your ONNX model")
        print("3. Run this script again")
        return

    try:
        task, optimizer, controller, configs = setup_skill_policy_benchmark(
            task_name="spot_barbell",  # Or any other Spot task
            onnx_model_path=onnx_path,
            target_pos=np.array([1.0, 0.0, 0.3]),  # Target object position
            locomotion_only=False,
        )

        print(f"✓ Successfully set up skill policy optimizer")
        print(f"  Task: {task.__class__.__name__}")
        print(f"  Optimizer: {optimizer.__class__.__name__}")
        print(f"  Control dims: {task.nu}")

    except FileNotFoundError:
        print(f"✗ ONNX model not found at: {onnx_path}")
        return

    # Example 2: Run comparison benchmark
    print("\n\nExample 2: Benchmark Skill Policy vs MPPI")
    print("-" * 60)

    try:
        benchmark_skill_policy_vs_sampling(
            task_name="spot_barbell",
            onnx_model_path=onnx_path,
            sample_optimizer_name="mppi",
            num_samples=20,
        )
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        print("\nMake sure:")
        print("  - ONNX model path is correct")
        print("  - Task name is valid")
        print("  - ONNX model outputs match expected format")


if __name__ == "__main__":
    main()
