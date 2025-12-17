# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass

import h5py
import numpy as np
import tyro
from mujoco import mj_step
from tqdm import tqdm

from judo.controller import Controller, ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks
from judo.utils.mujoco import SimBackendSpot


@dataclass
class RunMPCConfig:
    """Config class for MPC runs."""

    init_task: str = "spot_yellow_chair"
    init_optimizer: str = "cem"
    visualize: bool = True
    num_episodes: int = 5
    episode_length_s: float = 10
    viz_dt: float = 0.02
    save_results: bool = True
    output_dir: str = "run_mpc/results"
    verbose: bool = False
    # Data recording options
    record_all_data: bool = False  # If True, records all data fields
    record_qvel: bool = False
    record_xpos: bool = False
    record_xquat: bool = False
    record_ctrl: bool = False
    record_sensordata: bool = False
    record_mocap: bool = False
    record_traces: bool = False
    # Optimizer rollout recording options
    record_rollouts: bool = True  # Record optimizer rollout states and rewards
    record_rollout_controls: bool = False  # Record optimizer rollout controls
    record_rollout_sensors: bool = False  # Record optimizer rollout sensors


def run_single_episode(
    config: RunMPCConfig,
    task,
    task_config,
    controller,
    controller_config,
    sim_model,
    ctrl_model,
    spot_sim_backend,
    is_spot_task: bool,
    episode_idx: int = 0,
) -> dict:
    """Runs a single episode of MPC simulation.

    Args:
        config: Configuration for the MPC run
        task: Task instance
        task_config: Task configuration
        controller: Controller instance
        controller_config: Controller configuration
        sim_model: Simulation model
        ctrl_model: Controller model
        spot_sim_backend: Spot simulation backend (if applicable)
        is_spot_task: Whether this is a Spot task
        visualizer: Visualizer instance (optional)
        episode_idx: Episode index for seeding

    Returns:
        Dictionary containing all recorded data from the episode
    """
    # Set up synchronous loop timing
    sim_dt = sim_model.opt.timestep
    plan_dt = 1.0 / controller_config.control_freq
    ctrl_dt = ctrl_model.opt.timestep

    # Use sim_dt as the primary loop timestep (run physics at native frequency)
    step_dt = sim_dt
    num_steps = int(config.episode_length_s / step_dt) + 1

    # Compute how many simulation steps per each type of step
    sim_steps_per_plan_step = max(1, int(plan_dt / sim_dt))
    sim_steps_per_ctrl_step = max(1, int(ctrl_dt / sim_dt))
    sim_steps_per_viz_step = max(1, int(config.viz_dt / sim_dt))

    # Reset task and controller
    np.random.seed(episode_idx)  # Set seed for reproducibility
    task.reset()
    controller.reset()
    curr_time = 0.0
    task.data.time = 0.0
    curr_state = np.concatenate([task.data.qpos, task.data.qvel])

    # Determine what data to record based on config
    should_record = lambda field: config.record_all_data or getattr(config, f"record_{field}", False)

    # Initialize data storage - always record qpos, rewards, metrics, and termination
    episode_data = {
        "rewards": [],
        "qpos_traj": [],
        "time_traj": [],
        "success": False,
        "failure": False,
        "length": 0.0,
        "metrics": {},
    }

    # Add optional data fields
    if should_record("qvel"):
        episode_data["qvel_traj"] = []
    if should_record("xpos"):
        episode_data["xpos_traj"] = []
    if should_record("xquat"):
        episode_data["xquat_traj"] = []
    if should_record("ctrl"):
        episode_data["ctrl_traj"] = []
    if should_record("sensordata"):
        episode_data["sensordata_traj"] = []
    if should_record("mocap"):
        episode_data["mocap_pos_traj"] = []
        episode_data["mocap_quat_traj"] = []
    if should_record("traces"):
        episode_data["traces_traj"] = []

    # Add rollout data fields (optimizer samples)
    if config.record_rollouts or config.record_all_data:
        episode_data["rollout_states"] = []
        episode_data["rollout_rewards"] = []
        episode_data["control_timesteps"] = []  # Track which sim timesteps had control updates
    if config.record_rollout_controls or config.record_all_data:
        episode_data["rollout_controls"] = []
    if config.record_rollout_sensors or config.record_all_data:
        episode_data["rollout_sensors"] = []

    current_action = None

    # Main simulation loop
    for step in tqdm(range(num_steps), desc=f"Episode {episode_idx + 1}/{config.num_episodes}", leave=False):
        curr_time = step * step_dt
        curr_state = np.concatenate([task.data.qpos, task.data.qvel])

        # Update planning (action trajectory)
        if step % sim_steps_per_plan_step == 0:
            controller.update_action(curr_state, curr_time)

            # Record rollout data (optimizer samples) after planning
            if config.record_rollouts or config.record_all_data:
                episode_data["rollout_states"].append(np.array(controller.states))
                episode_data["rollout_rewards"].append(np.array(controller.rewards))
                episode_data["control_timesteps"].append(step)
            if config.record_rollout_controls or config.record_all_data:
                episode_data["rollout_controls"].append(np.array(controller.rollout_controls))
            if config.record_rollout_sensors or config.record_all_data:
                episode_data["rollout_sensors"].append(np.array(controller.sensors))

        # Update control (get action from trajectory)
        if step % sim_steps_per_ctrl_step == 0:
            current_action = controller.action(curr_time)
            if not is_spot_task:
                task.data.ctrl[:] = current_action

        # Simulation step (happens every loop iteration at sim_dt frequency)
        task.pre_sim_step()
        if is_spot_task and spot_sim_backend is not None and current_action is not None:
            spot_sim_backend.sim(sim_model, task.data, current_action)
        else:
            mj_step(sim_model, task.data)
        controller.system_metadata = task.get_sim_metadata()

        # Note: Skip single-step reward computation for now as it causes issues with
        # tasks that expect trajectory inputs (T > 1). The rollout rewards from MPC
        # are recorded separately and are the main data we need.
        # TODO: Fix spot task reward functions to handle single-timestep case
        episode_data["rewards"].append(0.0)  # Placeholder

        # Compute metrics (no early termination)
        metadata = {}
        metrics = task.compute_metrics(sim_model, task.data, task_config, metadata=metadata)

        for k, v in metrics.items():
            if k not in episode_data["metrics"]:
                episode_data["metrics"][k] = []
            episode_data["metrics"][k].append(v)

        task.post_sim_step()

        # Record data at visualization frequency (every N simulation steps)
        if step % sim_steps_per_viz_step == 0:
            episode_data["time_traj"].append(curr_time)
            episode_data["qpos_traj"].append(np.array(task.data.qpos))

            if should_record("qvel"):
                episode_data["qvel_traj"].append(np.array(task.data.qvel))
            if should_record("xpos"):
                episode_data["xpos_traj"].append(np.array(task.data.xpos))
            if should_record("xquat"):
                episode_data["xquat_traj"].append(np.array(task.data.xquat))
            if should_record("ctrl"):
                episode_data["ctrl_traj"].append(np.array(task.data.ctrl))
            if should_record("sensordata"):
                episode_data["sensordata_traj"].append(np.array(task.data.sensordata))
            if should_record("mocap"):
                episode_data["mocap_pos_traj"].append(np.array(task.data.mocap_pos))
                episode_data["mocap_quat_traj"].append(np.array(task.data.mocap_quat))
            if should_record("traces") and controller.traces is not None:
                episode_data["traces_traj"].append(np.array(controller.traces))

    # Episode always runs to completion
    episode_data["length"] = curr_time

    # Convert lists to numpy arrays
    for key in episode_data:
        if isinstance(episode_data[key], list) and key != "metrics":
            if len(episode_data[key]) > 0:
                episode_data[key] = np.asarray(episode_data[key])
            else:
                episode_data[key] = np.empty((0,))

    return episode_data


def run_mpc(config: RunMPCConfig) -> list[dict]:
    """Sets up and runs MPC controller for multiple episodes with data recording.

    Args:
        config: Configuration for the MPC run

    Returns:
        List of dictionaries containing all recorded data from each episode
    """
    # Load task
    task_dict = get_registered_tasks()
    if config.init_task not in task_dict:
        raise ValueError(f"Task '{config.init_task}' is not registered.")
    task_cls, task_config_cls = task_dict[config.init_task]
    task_config = task_config_cls()
    task = task_cls()

    # Load optimizer
    optimizer_dict = get_registered_optimizers()
    if config.init_optimizer not in optimizer_dict:
        raise ValueError(f"Optimizer '{config.init_optimizer}' is not registered.")
    optimizer_cls, optimizer_config_cls = optimizer_dict[config.init_optimizer]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override(config.init_task)
    optimizer = optimizer_cls(optimizer_config, task.nu)

    # Load controller
    controller_config = ControllerConfig()
    controller_config.set_override(config.init_task)
    controller = Controller(
        controller_config,
        task,
        task_config,
        optimizer,
        optimizer_config,
    )

    sim_model = task.sim_model
    ctrl_model = task.model

    # Initialize Spot backend if needed
    spot_sim_backend = None
    is_spot_task = 'spot' in config.init_task and 'baseline' not in config.init_task
    if is_spot_task:
        spot_sim_backend = SimBackendSpot(task_to_sim_ctrl=task.task_to_sim_ctrl)

    # Check visualization request
    if config.visualize:
        # Note: Visualization is not currently supported in this script
        # The new codebase uses ViserMjModel which requires different setup
        # For visualization, use the post-run visualizer script
        print("Warning: Live visualization is not currently supported in run_mpc.py")
        print("         You can visualize results after recording using:")
        print("         python scripts/visualize_benchmark.py <h5_file>")
        config.visualize = False

    # Run multiple episodes
    all_episodes_data = []
    for episode_idx in range(config.num_episodes):
        episode_data = run_single_episode(
            config=config,
            task=task,
            task_config=task_config,
            controller=controller,
            controller_config=controller_config,
            sim_model=sim_model,
            ctrl_model=ctrl_model,
            spot_sim_backend=spot_sim_backend,
            is_spot_task=is_spot_task,
            episode_idx=episode_idx,
        )
        all_episodes_data.append(episode_data)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Summary of {config.num_episodes} Episodes:")
    print(f"  Task: {config.init_task}")
    print(f"  Optimizer: {config.init_optimizer}")

    # Aggregate statistics
    all_rewards = [ep["rewards"] for ep in all_episodes_data]
    avg_reward = np.mean([np.mean(r) for r in all_rewards])
    num_successes = sum(1 for ep in all_episodes_data if ep["success"])
    num_failures = sum(1 for ep in all_episodes_data if ep["failure"])
    avg_length = np.mean([ep["length"] for ep in all_episodes_data])

    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Successes: {num_successes}/{config.num_episodes}")
    print(f"  Failures: {num_failures}/{config.num_episodes}")
    print(f"  Average Episode Length: {avg_length:.2f}s")

    # Aggregate and reduce metrics
    if all_episodes_data[0]["metrics"]:
        all_metrics = {}
        for ep in all_episodes_data:
            for k, v in ep["metrics"].items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].extend(v)
        reduced_metrics = task.reduce_metrics(all_metrics)
        for k, v in reduced_metrics.items():
            print(f"  {k}: {v:.4f}")

    print("=" * 80)

    # Save results if requested
    if config.save_results:
        output_dir = os.path.join(config.output_dir, config.init_task)
        os.makedirs(output_dir, exist_ok=True)
        h5_filename = os.path.join(output_dir, f"trajectories.h5")

        sim_dt = sim_model.opt.timestep
        plan_dt = 1.0 / controller_config.control_freq
        ctrl_dt = ctrl_model.opt.timestep

        with h5py.File(h5_filename, "w") as f:
            # Save global configuration as attributes
            f.attrs["task"] = config.init_task
            f.attrs["optimizer"] = config.init_optimizer
            f.attrs["num_episodes"] = config.num_episodes
            f.attrs["episode_length_s"] = config.episode_length_s
            f.attrs["viz_dt"] = config.viz_dt
            f.attrs["sim_dt"] = sim_dt
            f.attrs["plan_dt"] = plan_dt
            f.attrs["ctrl_dt"] = ctrl_dt

            # Create nested structure for visualizer compatibility: /<task>/<optimizer>/episode_<i>/
            task_group = f.create_group(config.init_task)
            optimizer_group = task_group.create_group(config.init_optimizer)

            # Save data for each episode in separate groups
            for i, episode_data in enumerate(all_episodes_data):
                ep_group = optimizer_group.create_group(f"episode_{i}")

                # Save episode data
                for key, value in episode_data.items():
                    if key == "metrics":
                        if value:
                            metrics_group = ep_group.create_group("metrics")
                            for metric_name, metric_values in value.items():
                                metrics_group.create_dataset(metric_name, data=np.asarray(metric_values))
                    elif key in ["success", "failure", "length"]:
                        ep_group.attrs[key] = value
                    elif isinstance(value, np.ndarray) and value.size > 0:
                        ep_group.create_dataset(key, data=value)

            # Also save stacked arrays in a separate group for easy programmatic access
            # Structure: /stacked/dataset[episode_idx, ...] for each field
            stacked_group = f.create_group("stacked")
            for key in all_episodes_data[0].keys():
                if key == "metrics":
                    # Handle metrics separately
                    if all_episodes_data[0]["metrics"]:
                        metrics_group = stacked_group.create_group("metrics")
                        for metric_name in all_episodes_data[0]["metrics"].keys():
                            metric_data = [ep["metrics"][metric_name] for ep in all_episodes_data]
                            metrics_group.create_dataset(metric_name, data=np.asarray(metric_data))
                elif key in ["success", "failure", "length"]:
                    # Save scalars as arrays indexed by episode
                    data = np.array([ep[key] for ep in all_episodes_data])
                    stacked_group.create_dataset(key, data=data)
                else:
                    # Stack trajectory data: shape becomes (num_episodes, ...)
                    data_list = [ep[key] for ep in all_episodes_data]
                    if len(data_list) > 0 and isinstance(data_list[0], np.ndarray) and data_list[0].size > 0:
                        stacked_group.create_dataset(key, data=np.asarray(data_list))

        print(f"\nResults saved to: {h5_filename}")
        print(f"To visualize: python scripts/visualize_benchmark.py {h5_filename}")

    return all_episodes_data


if __name__ == "__main__":
    run_mpc(tyro.cli(RunMPCConfig))
