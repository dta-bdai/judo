import datetime
import json
import os
from dataclasses import fields
from fractions import Fraction
from functools import reduce
from math import gcd, isclose
from textwrap import dedent
from typing import Any, Sequence

import h5py
import numpy as np
from mujoco import mj_step
from tqdm import tqdm

from judo.controller import Controller, ControllerConfig
from judo.optimizers import Optimizer, OptimizerConfig, get_registered_optimizers
from judo.tasks import Task, TaskConfig, get_registered_tasks

# Define the Spot SimBackend from the provided code
import onnxruntime
from scipy.spatial.transform import Rotation as R
from judo.utils.mujoco import SimBackendSpot
from judo.tasks.spot.spot_base import SpotBase
from pathlib import Path

# ##### #
# UTILS #
# ##### #


def _as_fraction(x: float, max_den: int = 10**4) -> Fraction:
    """Convert a float to a Fraction with limited denominator."""
    return Fraction(x).limit_denominator(max_den)


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _lcm_many(values: Sequence[int]) -> int:
    return reduce(_lcm, values)


def _gcd_many(values: Sequence[int]) -> int:
    return reduce(gcd, values)


def compute_common_step_many(dts: Sequence[float], *, max_den: int = 10**4) -> tuple[float, list[int]]:
    """Pick a loop step that evenly divides all given dts.

    Returns:
        step_dt: the loop step size (rational GCD of all dts).
        loop_steps: list of loop iterations per 1 dt, in the same order as `dts`.
    """
    assert len(dts) >= 1, "Provide at least one dt."
    assert all(dt > 0 for dt in dts), "Timesteps must be positive."

    fracs = [_as_fraction(dt, max_den) for dt in dts]

    # put all fractions over a common denominator L (integer grid)
    L = _lcm_many([f.denominator for f in fracs])
    nums = [f.numerator * (L // f.denominator) for f in fracs]  # each dt as n_i / L

    g = _gcd_many(nums)  # integer GCD on the common grid
    step_frac = Fraction(g, L)  # step_dt as a rational
    step_dt = float(step_frac)

    loop_steps = [n // g for n in nums]  # exact integer counts by construction

    # sanity checks (tolerant to float conversion)
    for dt_f, loops in zip(fracs, loop_steps, strict=False):
        assert loops > 0
        assert isclose(step_dt * loops, float(dt_f), abs_tol=1e-15)

    return step_dt, loop_steps


def summarize_dataclass(dataclass_instance) -> None:  # noqa: ANN001
    """Prints the fields and values of a dataclass instance."""
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)
        print(f"  {field.name}: {value}")


def load_benchmark_summary_from_h5(h5_filepath: str) -> dict[str, Any]:
    """Load benchmark results from H5 file and create a summary dictionary.

    Args:
        h5_filepath: Path to the H5 file containing benchmark results

    Returns:
        Dictionary containing summary statistics for each task and optimizer
    """
    summary = {}

    with h5py.File(h5_filepath, "r") as f:
        for task_name in f.keys():
            summary[task_name] = {}
            task_group = f[task_name]

            for optimizer_name in task_group.keys():
                opt_group = task_group[optimizer_name]

                # Collect episode data
                rewards = []
                successes = 0
                failures = 0
                lengths = []
                all_metrics = {}

                for episode_name in opt_group.keys():
                    ep_group = opt_group[episode_name]

                    # Get rewards
                    episode_rewards = ep_group["rewards"][:]
                    rewards.append(np.mean(episode_rewards))

                    # Get success/failure/length
                    successes += int(ep_group.attrs["success"])
                    failures += int(ep_group.attrs["failure"])
                    lengths.append(ep_group.attrs["length"])

                    # Get metrics if they exist
                    if "metrics" in ep_group:
                        metrics_group = ep_group["metrics"]
                        for metric_name in metrics_group.keys():
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = []
                            all_metrics[metric_name].extend(metrics_group[metric_name][:].tolist())

                # Compute reduced metrics (using task's reduce_metrics method if available)
                reduced_metrics = {}
                if all_metrics:
                    try:
                        all_tasks = get_registered_tasks()
                        if task_name in all_tasks:
                            task_cls = all_tasks[task_name][0]
                            reduced_metrics = task_cls().reduce_metrics(all_metrics)
                    except Exception:
                        # If reduce_metrics fails, just compute means
                        reduced_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

                # Store summary
                summary[task_name][optimizer_name] = {
                    "average_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "successes": successes,
                    "failures": failures,
                    "num_episodes": len(lengths),
                    "average_length": float(np.mean(lengths)),
                    "std_length": float(np.std(lengths)),
                    "reduced_metrics": reduced_metrics,
                }

    return summary


def print_benchmark_summary(summary: dict[str, Any]) -> None:
    """Print benchmark summary in a formatted way.

    Args:
        summary: Summary dictionary from load_benchmark_summary_from_h5 or computed directly
    """
    print("Summary of Benchmark Results")
    print("=" * 80)

    for task_name, optimizers in summary.items():
        print(f"Task: {task_name}")
        for optimizer_name, stats in optimizers.items():
            print(f"  Optimizer: {optimizer_name}")
            print(f"    Average reward: {stats['average_reward']:.4f}")

            # Print reduced metrics if available
            if stats.get("reduced_metrics"):
                for metric_name, metric_value in stats["reduced_metrics"].items():
                    print(f"    {metric_name}: {metric_value:.4f}")

            print(f"    Successes: {stats['successes']}/{stats['num_episodes']}")
            print(f"    Failures: {stats['failures']}/{stats['num_episodes']}")
            print(f"    Average episode length: {stats['average_length']:.4f} seconds")
            print(f"    STD of episode length: {stats['std_length']:.4f} seconds")


def load_and_print_benchmark(h5_filepath: str) -> None:
    """Load benchmark results from H5 file and print the summary.

    Args:
        h5_filepath: Path to the H5 file containing benchmark results
    """
    summary = load_benchmark_summary_from_h5(h5_filepath)
    print_benchmark_summary(summary)


# ############## #
# MAIN FUNCTIONS #
# ############## #


def load_controller(
    task_name: str,
    optimizer_name: str,
    verbose: bool = False,
    onnx_session: Any | None = None,
    target_cmd: np.ndarray | None = None,
    locomotion_only: bool = False,
) -> tuple[
    Task,
    Optimizer,
    Controller,
    TaskConfig,
    OptimizerConfig,
    ControllerConfig,
]:
    """Loads a controller associated with a task and optimizer.

    Args:
        task_name: Name of the task to load
        optimizer_name: Name of the optimizer to use
        verbose: Whether to print configuration details
        onnx_session: ONNX session for skill policy (required if optimizer_name == "skill_policy")
        target_cmd: Target command for skill policy (xyz position)
        locomotion_only: Whether to use locomotion only for skill policy

    Returns:
        Tuple of (task, optimizer, controller, task_config, optimizer_config, controller_config)
    """
    # load task
    task_dict = get_registered_tasks()
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' is not registered.")
    task_cls, task_config_cls = task_dict[task_name]
    task_config = task_config_cls()
    task = task_cls()

    # load optimizer
    optimizer_dict = get_registered_optimizers()
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Optimizer '{optimizer_name}' is not registered.")
    optimizer_cls, optimizer_config_cls = optimizer_dict[optimizer_name]
    optimizer_config = optimizer_config_cls()

    # Special handling for skill policy optimizer
    if optimizer_name == "skill_policy":
        if onnx_session is None:
            raise ValueError(
                "onnx_session must be provided when using skill_policy optimizer. "
                "Load an ONNX model with onnxruntime.InferenceSession() and pass it here."
            )
        # Set skill policy specific config
        optimizer_config.onnx_session = onnx_session
        optimizer_config.locomotion_only = locomotion_only
        if target_cmd is not None:
            optimizer_config.target_cmd = target_cmd
        # Force num_rollouts to 1 for skill policy
        optimizer_config.num_rollouts = 1
    else:
        # Standard optimizer config override
        optimizer_config.set_override(task_name)

    optimizer = optimizer_cls(optimizer_config, task.nu)

    # load controller
    controller_config_cls = ControllerConfig
    controller_config = controller_config_cls()
    controller_config.set_override(task_name)

    # For skill policy, set max_opt_iters to 1 (no optimization loop needed)
    if optimizer_name == "skill_policy":
        controller_config.max_opt_iters = 1

    controller = Controller(
        controller_config,
        task,
        task_config,
        optimizer,
        optimizer_config,
    )

    # summarize the task, optimizer, and controller configs for debugging
    if verbose:
        print("*" * 80)
        print("Loaded the following configurations...\n")
        print(f"Task ({task_name}):")
        summarize_dataclass(task_config)
        print(f"Optimizer ({optimizer_name}):")
        # Don't print onnx_session object, it's too verbose
        if optimizer_name == "skill_policy":
            print(f"  onnx_session: <InferenceSession>")
            print(f"  target_cmd: {optimizer_config.target_cmd}")
            print(f"  locomotion_only: {optimizer_config.locomotion_only}")
            print(f"  num_rollouts: {optimizer_config.num_rollouts}")
            print(f"  num_nodes: {optimizer_config.num_nodes}")
        else:
            summarize_dataclass(optimizer_config)
        print("Controller:")
        summarize_dataclass(controller_config)
        print("*" * 80)

    return task, optimizer, controller, task_config, optimizer_config, controller_config


def benchmark_single_task_and_optimizer(
    task_name: str,
    optimizer_name: str,
    num_episodes: int = 10,
    episode_length_s: float = 30.0,
    viz_dt: float = 0.02,
    min_dt: float = 0.0001,
    verbose: bool = False,
    onnx_session: Any | None = None,
    target_cmd: np.ndarray | None = None,
    locomotion_only: bool = False,
) -> list[dict[str, Any]]:
    """Benchmarks a single task and optimizer combination.

    Args:
        task_name: Name of the task to benchmark.
        optimizer_name: Name of the optimizer to use.
        num_episodes: Number of episodes to run for the benchmark.
        episode_length_s: Maximum length of each episode (in seconds).
        viz_dt: Timestep for visualization/logging (in seconds).
        min_dt: Minimum timestep as a quantum unit for synchronization between simulation and controller.
        verbose: Whether to print detailed configuration information.
        onnx_session: ONNX session for skill policy (required if optimizer_name == "skill_policy")
        target_cmd: Target command for skill policy (xyz position)
        locomotion_only: Whether to use locomotion only for skill policy

    Returns:
        benchmark_results: A list of dictionaries containing results for each episode.
    """
    # load controller
    task, optimizer, controller, task_config, optimizer_config, controller_config = load_controller(
        task_name,
        optimizer_name,
        verbose=verbose,
        onnx_session=onnx_session,
        target_cmd=target_cmd,
        locomotion_only=locomotion_only,
    )
    sim_model = task.sim_model
    ctrl_model = task.model

    # set up the synchronous loop
    sim_dt = sim_model.opt.timestep
    plan_dt = 1.0 / controller_config.control_freq
    ctrl_dt = ctrl_model.opt.timestep
    assert sim_dt <= ctrl_dt, "Simulation timestep must be less than or equal to controller timestep."
    step_dt, loop_step_counts = compute_common_step_many([sim_dt, plan_dt, ctrl_dt, viz_dt])
    loop_steps_per_sim_step = loop_step_counts[0]  # sim_dt
    loop_steps_per_plan_step = loop_step_counts[1]  # plan_dt
    loop_steps_per_ctrl_step = loop_step_counts[2]  # ctrl_dt
    loop_steps_per_viz_step = loop_step_counts[3]  # viz_dt
    assert step_dt >= min_dt, f"Common step {step_dt} is less than minimum dt {min_dt}."
    num_steps = int(episode_length_s / step_dt) + 1

    # storage for outcomes
    benchmark_results = []

    # Initialize Spot backend if needed
    spot_sim_backend = None
    is_spot_task = 'spot' in task_name and 'baseline' not in task_name

    if is_spot_task:
        spot_sim_backend = SimBackendSpot(task_to_sim_ctrl=task.task_to_sim_ctrl)

    # loop through episodes
    for i in range(num_episodes):
        np.random.seed(i)  # reset seed for each task for fairness

        # reset task and controller
        task.reset()
        controller.reset()
        curr_time = 0.0
        task.data.time = 0.0  # manually reset here in case task.reset() doesn't
        curr_state = np.concatenate([task.data.qpos, task.data.qvel])

        # simulating
        curr_episode_results = {
            "rewards": [],
            "qpos_traj": [],
            "mocap_pos_traj": [],
            "mocap_quat_traj": [],
            "success": False,
            "failure": False,
            "length": 0.0,
            "metrics": {},
        }
        did_break = False
        current_action = None  # Store current action for Spot tasks
        for step in tqdm(range(num_steps), desc=f"Episode {i + 1}/{num_episodes}", leave=False):
            curr_time = step * step_dt
            curr_state = np.concatenate([task.data.qpos, task.data.qvel])

            # advancing the simulation, controller, and planner as needed
            if step % loop_steps_per_plan_step == 0:  # planning step (updates spline)
                controller.update_action(curr_state, curr_time)
            if step % loop_steps_per_ctrl_step == 0:  # control step (updates control from spline)
                current_action = controller.action(curr_time)
                # For Spot tasks, we don't set ctrl directly - it's handled in the sim step
                if not is_spot_task:
                    task.data.ctrl[:] = current_action
            if step % loop_steps_per_viz_step == 0:  # visualization step (no-op here, but could be used for logging)
                curr_episode_results["qpos_traj"].append(np.array(task.data.qpos))
                mocap_pos = np.array(task.data.mocap_pos)  # (num_mocap, 3)
                mocap_quat = np.array(task.data.mocap_quat)  # (num_mocap, 4)
                curr_episode_results["mocap_pos_traj"].append(mocap_pos)
                curr_episode_results["mocap_quat_traj"].append(mocap_quat)
            if step % loop_steps_per_sim_step == 0:  # simulation step (updates simulation state)
                task.pre_sim_step()
                if is_spot_task and spot_sim_backend is not None and current_action is not None:
                    # Use Spot-specific simulation backend
                    spot_sim_backend.sim(sim_model, task.data, current_action)
                else:
                    # Use standard MuJoCo step
                    mj_step(sim_model, task.data)
                controller.system_metadata = task.get_sim_metadata()

                # compute the instantaneous reward in the simulation
                curr_sensor = np.asarray(task.data.sensordata)
                curr_control = np.asarray(task.data.ctrl[:])
                reward = task.reward(
                    curr_state[None, None, :],  # (B, T, dim)
                    curr_sensor[None, None, :],
                    curr_control[None, None, :],
                    task_config,
                )[0]  # returns (B,), so we index to get scalar
                curr_episode_results["rewards"].append(reward)

                # check for termination conditions and/or writing metrics
                metadata = {}
                has_success = task.success(sim_model, task.data, task_config, metadata=metadata)
                has_failure = task.failure(sim_model, task.data, task_config, metadata=metadata)
                metrics = task.compute_metrics(sim_model, task.data, task_config, metadata=metadata)

                for k, v in metrics.items():
                    if k not in curr_episode_results["metrics"]:
                        curr_episode_results["metrics"][k] = []
                    curr_episode_results["metrics"][k].append(v)

                # only perform post step AFTER measuring metrics, since that can change the state
                task.post_sim_step()

                if has_failure:  # failures take precedence over successes in rare case of tiebreaker
                    curr_episode_results["failure"] = True
                    curr_episode_results["length"] = curr_time
                    did_break = True
                    break
                elif has_success:
                    curr_episode_results["success"] = True
                    curr_episode_results["length"] = curr_time
                    did_break = True
                    break

        if not did_break:
            curr_episode_results["length"] = curr_time

        # collecting results for this episode
        curr_episode_results["rewards"] = np.asarray(curr_episode_results["rewards"])
        curr_episode_results["qpos_traj"] = np.asarray(curr_episode_results["qpos_traj"])
        if len(curr_episode_results["mocap_pos_traj"]) > 0:
            curr_episode_results["mocap_pos_traj"] = np.asarray(curr_episode_results["mocap_pos_traj"])
        else:
            curr_episode_results["mocap_pos_traj"] = np.empty((0,))
        if len(curr_episode_results["mocap_quat_traj"]) > 0:
            curr_episode_results["mocap_quat_traj"] = np.asarray(curr_episode_results["mocap_quat_traj"])
        else:
            curr_episode_results["mocap_quat_traj"] = np.empty((0,))
        benchmark_results.append(curr_episode_results)

    return benchmark_results


def summarize_benchmark_results(all_results: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    """Prints a summary of all benchmark results.

    Args:
        all_results: A dictionary with the following structure:
            {
                task_name_1: {
                    optimizer_name_1: [
                        {
                            "rewards": np.ndarray of shape (num_steps_1,),
                            "success": bool,
                            "failure": bool,
                            "length": float,
                            "metrics": dict of metric_name to list of float, each of length num_steps_1,
                        },  # results for episode 1...
                        ...
                    ],
                    optimizer_name_2: [ ... ],
                    ...
                },
                task_name_2: {
                    optimizer_name_1: [
                        {
                            "rewards": np.ndarray of shape (num_steps_2,),
                            "success": bool,
                            "failure": bool,
                            "length": float,
                            "metrics": dict of metric_name to list of float, each of length num_steps_2,
                        },  # results for episode 1...
                        ...
                    ],
                    optimizer_name_2: [ ... ],
                },
                ...
            }
    """
    all_tasks = get_registered_tasks()
    print("Summary of Benchmark Results")
    print("=" * 80)
    for task_name, task_results in all_results.items():
        print(f"Task: {task_name}")
        for optimizer_name, episode_results_for_task_opt_pair in task_results.items():
            print(f"  Optimizer: {optimizer_name}")

            all_rewards = []
            all_metrics = {}
            for _, episode_results in enumerate(episode_results_for_task_opt_pair):
                all_rewards.append(episode_results["rewards"])
                for k, v in episode_results["metrics"].items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].extend(v)

            # summarize
            task_cls = all_tasks[task_name][0]
            reduced_metrics = task_cls().reduce_metrics(all_metrics)

            all_rewards = np.concatenate(all_rewards)
            avg_reward = np.mean(all_rewards)
            print(f"    Average reward: {avg_reward:.4f}")
            for metric_name, metric_value in reduced_metrics.items():
                print(f"    {metric_name}: {metric_value:.4f}")
            num_successes = sum(1 for r in episode_results_for_task_opt_pair if r["success"])
            num_failures = sum(1 for r in episode_results_for_task_opt_pair if r["failure"])
            avg_length = np.mean([r["length"] for r in episode_results_for_task_opt_pair])
            print(f"    Successes: {num_successes}/{len(episode_results_for_task_opt_pair)}")
            print(f"    Failures: {num_failures}/{len(episode_results_for_task_opt_pair)}")
            print(f"    Average episode length: {avg_length:.4f} seconds")
            print(f"    STD of episode length: {np.std([r['length'] for r in episode_results_for_task_opt_pair]):.4f} seconds")
    print("=" * 80)


def benchmark_multiple_tasks_and_optimizers(
    task_names: list[str] | None = None,
    optimizer_names: list[str] | None = None,
    num_episodes: int = 10,
    episode_length_s: float | list[float] = 30.0,
    min_dt: float = 0.0001,
    viz_dt: float = 0.02,
    verbose: bool = False,
    onnx_session_dict: dict[str, Any] | None = None,
    target_cmd_dict: dict[str, np.ndarray] | None = None,
    locomotion_only: bool = False,
    save_results: bool = False,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Benchmarks multiple tasks and optimizers.

    Reports results for the Cartesian product of the provided task and optimizer names. If None are provided for one
    of the lists, all registered tasks or optimizers will be used.

    Args:
        task_names: List of task names to benchmark.
        optimizer_names: List of optimizer names to use.
        num_episodes: Number of episodes to run for each (task, optimizer) pair.
        episode_length_s: Maximum length of each episode (in seconds). If a list is provided, it must match the length
            of task_names.
        min_dt: Minimum timestep as a quantum unit for synchronization between simulation and controller.
        viz_dt: Timestep for visualization/logging (in seconds).
        verbose: Whether to print detailed configuration information.
        onnx_session_dict: Dictionary mapping task names to ONNX sessions (for skill_policy optimizer).
            If None and skill_policy is in optimizer_names, an error will be raised.
        target_cmd_dict: Dictionary mapping task names to target commands (for skill_policy optimizer).
        locomotion_only: Whether to use locomotion only for skill policy

    Returns:
        all_results: A dictionary mapping (task_name, optimizer_name) pairs to their benchmark results.
    """
    if task_names is None:
        task_names = list(get_registered_tasks().keys())
    if optimizer_names is None:
        optimizer_names = list(get_registered_optimizers().keys())
    if isinstance(episode_length_s, float):
        episode_length_s = [episode_length_s] * len(task_names)
    assert isinstance(episode_length_s, list), "episode_length_s must be a float or a list of floats."
    assert len(episode_length_s) == len(task_names), (
        "If episode_length_s is a list, it must match the length of task_names."
    )

    # Check if skill_policy is in optimizer_names
    # Note: We only require ONNX sessions for tasks that will actually use skill_policy
    # Other tasks can use different optimizers without needing ONNX sessions

    all_results = {}
    for i, task_name in enumerate(task_names):
        print(f"Benchmarking Task: {task_name}...")
        all_results[task_name] = {}
        for optimizer_name in optimizer_names:
            print(f"  Benchmarking Optimizer: {optimizer_name}...")

            # Get skill policy parameters if applicable
            onnx_session = None
            target_cmd = None
            if optimizer_name == "skill_policy":
                # Check if this task has an ONNX session
                if onnx_session_dict is None or task_name not in onnx_session_dict:
                    print(f"    Skipping skill_policy for {task_name} (no ONNX session provided)")
                    continue
                onnx_session = onnx_session_dict[task_name]
                if target_cmd_dict is not None and task_name in target_cmd_dict:
                    target_cmd = target_cmd_dict[task_name]

            results = benchmark_single_task_and_optimizer(
                task_name,
                optimizer_name,
                num_episodes=num_episodes,
                episode_length_s=episode_length_s[i],
                min_dt=min_dt,
                viz_dt=viz_dt,
                verbose=verbose,
                onnx_session=onnx_session,
                target_cmd=target_cmd,
                locomotion_only=locomotion_only,
            )
            all_results[task_name][optimizer_name] = results

            # Print results immediately after completion
            print(f"\n  Results for {task_name} with {optimizer_name}:")
            print(f"  {'-' * 60}")

            # Compute and display metrics
            all_tasks = get_registered_tasks()
            task_cls = all_tasks[task_name][0]

            all_rewards = []
            all_metrics = {}
            for episode_results in results:
                all_rewards.append(episode_results["rewards"])
                for k, v in episode_results["metrics"].items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].extend(v)

            # Summarize metrics
            reduced_metrics = task_cls().reduce_metrics(all_metrics)

            all_rewards = np.concatenate(all_rewards)
            avg_reward = np.mean(all_rewards)
            print(f"    Average reward: {avg_reward:.4f}")
            for metric_name, metric_value in reduced_metrics.items():
                print(f"    {metric_name}: {metric_value:.4f}")
            num_successes = sum(1 for r in results if r["success"])
            num_failures = sum(1 for r in results if r["failure"])
            avg_length = np.mean([r["length"] for r in results])
            print(f"    Successes: {num_successes}/{len(results)}")
            print(f"    Failures: {num_failures}/{len(results)}")
            print(f"    Average episode length: {avg_length:.4f} seconds")
            print(f"    STD of episode length: {np.std([r['length'] for r in results]):.4f} seconds")
            print(f"  {'-' * 60}\n")

    # Create results directory with timestamp subfolder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # save the benchmark results to files in the timestamped subfolder
    h5_filename = os.path.join(results_dir, "benchmark_results.h5")
    json_filename = os.path.join(results_dir, "benchmark_results.json")

    if save_results:
        # Save H5 file
        with h5py.File(h5_filename, "w") as f:
            # attributes
            f.attrs["viz_dt"] = viz_dt

            for task_name, task_result_dict in all_results.items():
                task_group = f.create_group(task_name)
                for optimizer_name, episodes in task_result_dict.items():
                    opt_group = task_group.create_group(optimizer_name)
                    for i, episode in enumerate(episodes):
                        ep_group = opt_group.create_group(f"episode_{i}")
                        ep_group.create_dataset("rewards", data=episode["rewards"])
                        ep_group.create_dataset("qpos_traj", data=episode["qpos_traj"])
                        ep_group.create_dataset("mocap_pos_traj", data=episode["mocap_pos_traj"])
                        ep_group.create_dataset("mocap_quat_traj", data=episode["mocap_quat_traj"])
                        ep_group.attrs["success"] = episode["success"]
                        ep_group.attrs["failure"] = episode["failure"]
                        ep_group.attrs["length"] = episode["length"]
                        metrics_group = ep_group.create_group("metrics")
                        for metric_name, metric_values in episode["metrics"].items():
                            metrics_group.create_dataset(metric_name, data=np.asarray(metric_values))

        # Create and save JSON summary
        summary = load_benchmark_summary_from_h5(h5_filename)
        with open(json_filename, "w") as f:
            json.dump(summary, f, indent=2)

    # print a summary of the results
    # summarize_benchmark_results(all_results)
    if save_results:
        print_msg = dedent(
            f"""
            Benchmark results saved to:
            - H5 file: '{h5_filename}'
            - JSON summary: '{json_filename}'

            To visualize, run:
                python scripts/visualize_benchmark.py {h5_filename}

            To load and print summary:
                from scripts.performance_benchmark import load_and_print_benchmark
                load_and_print_benchmark('{h5_filename}')
            """
        )
    else:
        print_msg = dedent(
            f"""
            Benchmark results not saved.
            """
        )
    print(print_msg)

    
    return all_results