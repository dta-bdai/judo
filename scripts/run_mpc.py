# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dacite
import h5py
import numpy as np
import torch
from tqdm import tqdm
import tyro

from judo.app.structs import MujocoState
from judo.controller import BatchedControllers as JudoBatchedController
from judo.controller import Controller as JudoController
from judo.controller import ControllerConfig
from judo.optimizers import (
    Optimizer,
    OptimizerConfig,
    get_registered_optimizers,
)
from judo.simulation.mj_simulation import MJSimulation
from judo.simulation.policy_mj_simulation import PolicyMJSimulation
from judo.tasks import (
    Task as JudoTask,
)
from judo.tasks import (
    TaskConfig as JudoTaskConfig,
)
from judo.tasks import (
    get_registered_tasks as get_registered_judo_tasks,
)
from judo.controller.spot_locomotion import BatchedSpotLocomotionController
from judo.tasks.spot.spot_constants import SPOT_LOCOMOTION_PT_PATH
from judo.utils.mjwarp_rollout_backend import MJWarpRolloutBackend
from judo.utils.timer import Timer
from judo.visualizers.visualizer import Visualizer


def decode_config(obj: dict) -> dict:
    """Custom JSON decoder to handle NumPy arrays and Path strings."""
    for key, value in obj.items():
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            obj[key] = np.array(value)
        if isinstance(value, str) and ("/" in value or "\\" in value):
            obj[key] = Path(value)
    return obj


@dataclass
class PublicMPCConfig:
    """Config class for MPC runs."""

    config_path: Path  # Path to .json containing task/controller config for MPC run.
    dataset_output_path: Path | None = None  # Path to .h5 dataset of trajectories.
    num_trajectories: int = 10  # Number of MPC runs to perform.
    num_parallel: int = 1  # Number of trajectories to run in parallel (batched MPC).
    max_num_task_steps: int = 750  # Maximum number of task steps to run per MPC run.
    chunk_size: int = 1  # Size of the memory chunk for the h5 dataset (defaults to 1 -> 1 trajectory per chunk).
    control_freq: float | None = None  # MPC control frequency in Hz. If None, use the controller config's frequency.
    store_rollouts: bool = True  # Store open loop control inputs and state trajectories.
    store_viapoints: bool = True  # Store open loop control of the best spline.
    visualize: bool = False
    locomotion_policy_path: Path = SPOT_LOCOMOTION_PT_PATH


@dataclass
class SizeData:
    """Sizes needed to calculate things in the dataset."""

    max_num_task_steps: int
    nq: int
    nv: int
    nu: int
    max_num_mpc_steps: int
    nsensordata: int
    num_nodes: int
    num_rollouts: int
    num_trajectories: int
    chunk_size: int
    num_timesteps: int
    sim_steps_per_mpc_step: int


def _make_size_data(sim: MJSimulation, controller: JudoController, config: PublicMPCConfig) -> SizeData:
    """Calculates the sizes for everything."""
    # Use task.nu for control dimensions (the optimizer/spline/rollout_controls all use task.nu)
    nu = sim.task.nu
    if config.control_freq is None:
        control_freq = controller.controller_cfg.control_freq
    else:
        control_freq = config.control_freq
    sim_steps_per_mpc_step = int(1.0 / control_freq / sim.task.dt)
    max_num_mpc_steps = np.ceil(config.max_num_task_steps / sim_steps_per_mpc_step).astype(int)
    nq = sim.task.model.nq
    nv = sim.task.model.nv
    nsensordata = sim.task.model.nsensordata
    num_nodes = controller.optimizer_cfg.num_nodes
    num_rollouts = controller.optimizer_cfg.num_rollouts
    return SizeData(
        max_num_mpc_steps=max_num_mpc_steps,
        nq=nq,
        nv=nv,
        nsensordata=nsensordata,
        num_nodes=num_nodes,
        num_rollouts=num_rollouts,
        num_trajectories=config.num_trajectories,
        chunk_size=config.chunk_size,
        num_timesteps=controller.num_timesteps,
        sim_steps_per_mpc_step=sim_steps_per_mpc_step,
        max_num_task_steps=config.max_num_task_steps,
        nu=nu,
    )


def load_configs_from_json_data(json_data: Any) -> tuple[JudoTask, Optimizer, ControllerConfig]:
    """Loads the configs from a json file."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

    available_tasks = get_registered_judo_tasks()
    available_optimizers = get_registered_optimizers()
    task_entry = available_tasks.get(json_data["task"])
    assert task_entry is not None, f"Task {json_data['task']} is not registered!"
    optimizer_entry = available_optimizers.get(json_data["optimizer"])
    assert optimizer_entry is not None, f"Optimizer {json_data['optimizer']} is not registered!"

    # Loads the task with the config given from the json file.
    task_cls, task_config_cls = task_entry
    task_config: JudoTaskConfig = dacite.from_dict(task_config_cls, json_data["task_config"])
    task: JudoTask = task_cls()
    task.config = task_config

    # Loads the optimizer with the config given from the json file.
    optimizer_cls, optimizer_config_cls = optimizer_entry
    optimizer_config: OptimizerConfig = dacite.from_dict(optimizer_config_cls, json_data["optimizer_config"])
    optimizer: Optimizer = optimizer_cls(optimizer_config, task.nu)

    # Loads the controller with the config given from the json file.
    controller_cfg: ControllerConfig = dacite.from_dict(ControllerConfig, json_data["controller_config"])

    return (
        task,
        optimizer,
        controller_cfg,
    )


def update_visualization(visualizer: Visualizer, sim_state: MujocoState, traces: np.ndarray) -> None:
    visualizer.data.xpos[:] = sim_state.xpos
    visualizer.data.xquat[:] = sim_state.xquat
    visualizer.viser_model.set_data(visualizer.data)
    sensor_rollout_size = traces.shape[1]
    num_trace_sensors = traces.shape[2]
    all_traces_rollout_size = sensor_rollout_size * num_trace_sensors
    visualizer.viser_model.set_traces(traces, all_traces_rollout_size)


@dataclass
class MPCTimers:
    """Timers for measuring MPC performance."""

    optimization: Timer
    sim_step: Timer
    reward_compute: Timer
    h5_write: Timer


def run_mpc_batch(
    sims: list[MJSimulation],
    batched_controllers: JudoBatchedController,
    config: PublicMPCConfig,
    size_data: SizeData,
    timers: MPCTimers,
    vis: Visualizer | None = None,
) -> list[dict[str, np.ndarray | int]]:
    """Runs MPC for one or more problems using batched controllers.

    This function handles both single and multi-problem cases uniformly by leveraging
    the BatchedControllers class which shares a single RolloutBackend for GPU efficiency.

    Typically for Spot tasks:
        physics_step = model.opt.timestep = 0.01 s
        task_step = task.dt = physics_step * num_physics_substeps = 0.01 * 2 = 0.02 s
        mpc_step = 1 / control_freq = 1 / 10 = 0.1 s
        tasks_steps_per_mpc_step = mpc_step / task_step  = 0.1 / 0.02 = 5

    We only record at the task step rate, not the physics step rate.
    I.e. we never record Spot states at 100Hz, only at 50Hz.
    We make this choice since downstream BC and RL applications must take decisions based on the current state
    (current observation) at the task step rate not the physics step rate.

    Args:
        sims: List of MJSimulation instances, one per parallel trajectory.
        batched_controllers: BatchedControllers instance managing all controllers.
        config: MPC configuration.
        size_data: Size information for allocating arrays.
        timers: MPCTimers instance for performance measurement.
        vis: Optional visualizer (only used when len(sims) == 1).

    Returns:
        List of result dictionaries, one per simulation.
    """

    num_parallel = len(sims)
    controllers = batched_controllers.controllers

    # Reset all simulations and controllers, sample new initial conditions + goals
    for sim, ctrl in zip(sims, controllers, strict=True):
        sim.task.reset()
        if isinstance(sim, PolicyMJSimulation):
            sim.reset_policy_state()
        ctrl.reset()
        ctrl.update_states(sim.sim_state)

    # Sync previous_actions from sims to rollout backend for hierarchical control
    previous_actions_list: list[np.ndarray | None] = []
    for sim in sims:
        if isinstance(sim, PolicyMJSimulation):
            previous_actions_list.append(sim.last_policy_output)
        else:
            previous_actions_list.append(None)
    batched_controllers.set_init_previous_actions(previous_actions_list)

    # First batched action update
    timers.optimization.tic()
    batched_controllers.update_action()
    timers.optimization.toc()

    # Allocate storage for all trajectories (padded with nans)
    qpos_trajs = np.full((num_parallel, size_data.max_num_task_steps, size_data.nq), np.nan, dtype="float64")
    qvel_trajs = np.full((num_parallel, size_data.max_num_task_steps, size_data.nv), np.nan, dtype="float64")
    control_trajs = np.full((num_parallel, size_data.max_num_task_steps, size_data.nu), np.nan, dtype="float64")
    sensor_trajs = np.full((num_parallel, size_data.max_num_task_steps, size_data.nsensordata), np.nan, dtype="float64")
    reward_trajs = np.full((num_parallel, size_data.max_num_task_steps), np.nan, dtype="float64")

    if config.store_viapoints:
        control_viapoints = np.full(
            (num_parallel, size_data.max_num_mpc_steps, size_data.num_nodes, size_data.nu), np.nan, dtype="float64"
        )

    if config.store_rollouts:
        rollout_states = np.full(
            (
                num_parallel,
                size_data.max_num_mpc_steps,
                size_data.num_rollouts,
                size_data.num_timesteps,
                size_data.nq + size_data.nv,
            ),
            np.nan,
            dtype="float64",
        )
        rollout_controls = np.full(
            (
                num_parallel,
                size_data.max_num_mpc_steps,
                size_data.num_rollouts,
                size_data.num_timesteps,
                size_data.nu,
            ),
            np.nan,
            dtype="float64",
        )
        rollout_rewards = np.full(
            (num_parallel, size_data.max_num_mpc_steps, size_data.num_rollouts),
            np.nan,
            dtype="int",
        )
        control_timesteps = np.full((num_parallel, size_data.max_num_mpc_steps), np.nan, dtype="int")

    mpc_step = 0  # counter for how many control updates have occurred
    for task_step in tqdm(range(config.max_num_task_steps), desc=f"Task steps ({num_parallel} parallel)", leave=False):
        # Write out current sim state for all simulations
        for i, (sim, ctrl) in enumerate(zip(sims, controllers, strict=True)):
            qpos_trajs[i, task_step] = sim.task.data.qpos
            qvel_trajs[i, task_step] = sim.task.data.qvel
            # Get task-space control from controller spline
            task_control = ctrl.compute(sim.task.data)
            control_trajs[i, task_step] = task_control
            sensor_trajs[i, task_step] = sim.task.data.sensordata

            timers.reward_compute.tic()
            reward_trajs[i, task_step] = sim.task.reward(
                states=np.concatenate([qpos_trajs[i, task_step], qvel_trajs[i, task_step]], axis=0)[None, None],
                sensors=sensor_trajs[i, task_step][None, None],
                controls=task_control[None, None],
            )[0]
            timers.reward_compute.toc()

        # Update control splines at MPC frequency
        if task_step % size_data.sim_steps_per_mpc_step == 0:
            # Update states for all controllers
            state_msgs = [sim.sim_state for sim in sims]
            batched_controllers.update_states(state_msgs)

            # Sync previous_actions from sims to rollout backend for hierarchical control
            previous_actions_list = []
            for sim in sims:
                if isinstance(sim, PolicyMJSimulation):
                    previous_actions_list.append(sim.last_policy_output)
                else:
                    previous_actions_list.append(None)
            batched_controllers.set_init_previous_actions(previous_actions_list)

            # Batched action update for all controllers
            timers.optimization.tic()
            batched_controllers.update_action()
            timers.optimization.toc()

            # Store rollout data and update control splines
            for i, ctrl in enumerate(controllers):
                if config.store_rollouts:
                    control_timesteps[i, mpc_step] = task_step
                    rollout_states[i, mpc_step] = ctrl.states
                    rollout_controls[i, mpc_step] = ctrl.rollout_controls
                    rollout_rewards[i, mpc_step] = ctrl.rewards

                if config.store_viapoints:
                    # Store task-space spline knots directly
                    control_viapoints[i, mpc_step] = ctrl.spline.y

            mpc_step += 1

        # Advance physics for all simulations
        # Pass task-space controls — sim.step() handles task_to_sim_ctrl internally
        timers.sim_step.tic()
        for i, sim in enumerate(sims):
            task_control = control_trajs[i, task_step]
            sim.step(task_control)
        timers.sim_step.toc()

        # Visualization support (only for single-sim case)
        if vis is not None and num_parallel == 1:
            update_visualization(vis, sims[0].sim_state, controllers[0].traces)
            time.sleep(sims[0].timestep)

    # Package results for each trajectory
    results = []
    for i in range(num_parallel):
        mpc_result: dict[str, np.ndarray | int] = {
            "task_step": config.max_num_task_steps,
            "qpos": qpos_trajs[i],
            "qvel": qvel_trajs[i],
            "control": control_trajs[i],
            "sensor": sensor_trajs[i],
            "reward": reward_trajs[i],
        }
        if config.store_rollouts:
            mpc_result["control_timesteps"] = control_timesteps[i]
            mpc_result["rollout_states"] = rollout_states[i]
            mpc_result["rollout_controls"] = rollout_controls[i]
            mpc_result["rollout_rewards"] = rollout_rewards[i]
        if config.store_viapoints:
            mpc_result["control_viapoints"] = control_viapoints[i]
        results.append(mpc_result)

    return results


def create_llc_controller(
    use_spot: bool, policy_path: str | None, device: str
) -> BatchedSpotLocomotionController | None:
    """Creates low-level locomotion controller if Spot task is used."""
    if not use_spot or policy_path is None:
        return None

    return BatchedSpotLocomotionController(
        model_path=policy_path,
        device=device,
    )


def run_mpc(config: PublicMPCConfig) -> None:
    """Runs MPC and writes closed-loop trajectories and optionally open-loop control & state trajectories to h5 file.

    Supports both single-problem and batched multi-problem modes:
    - num_parallel=1 (default): Runs trajectories sequentially, one at a time.
    - num_parallel>1: Runs multiple trajectories in parallel using BatchedControllers,
      leveraging GPU batched rollouts for efficiency.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

    # Load policy config from file (resolve relative to CWD).
    config_path = Path(config.config_path).resolve()
    with open(config_path, "r") as f:
        logging.info(f"Loading policy config from {config_path}")
        json_configs = json.load(f, object_hook=decode_config)

    task, optimizer, controller_cfg = load_configs_from_json_data(json_configs)

    # Disable fluid dynamics for mujoco_warp compatibility
    task.model.opt.density = 0

    # Visualization only supported for single trajectory mode
    vis = None
    if config.visualize:
        if config.num_parallel > 1:
            logging.warning("Visualization not supported in batched mode. Disabling visualization.")
        else:
            vis = Visualizer(init_task=json_configs["task"], init_optimizer=json_configs["optimizer"])

    # Ensure num_parallel doesn't exceed num_trajectories
    num_parallel = min(config.num_parallel, config.num_trajectories)
    if num_parallel != config.num_parallel:
        logging.info(f"Adjusted num_parallel from {config.num_parallel} to {num_parallel} (num_trajectories limit)")

    # Create simulations — use PolicyMJSimulation for Spot tasks (hierarchical control via C++)
    use_spot = task.uses_locomotion_policy
    policy_path = str(config.locomotion_policy_path) if use_spot else None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sims: list[MJSimulation] = []

    for _ in range(num_parallel):
        if use_spot:
            sim = PolicyMJSimulation(init_task=json_configs["task"])
        else:
            sim = MJSimulation(init_task=json_configs["task"])
        sim.task.config = copy.deepcopy(task.config)
        # Disable fluid dynamics for mujoco_warp compatibility
        sim.task.model.opt.density = 0
        sims.append(sim)

    # Create shared GPU rollout backend for batched execution
    # Torch-based controllers are used for GPU rollouts (not for simulation stepping)
    locomotion_controller = create_llc_controller(use_spot, policy_path, device)
    rollout_backend = MJWarpRolloutBackend(
        model=task.model,
        num_threads=optimizer.config.num_rollouts,
        num_problems=num_parallel,
        locomotion_controller=locomotion_controller,
        device=device,
    )
    batched_controllers = JudoBatchedController(controller_cfg, task, optimizer, rollout_backend=rollout_backend)
    size_data = _make_size_data(sims[0], batched_controllers.controllers[0], config)

    if config.dataset_output_path is None:
        # By default, save dataset next to config file.
        dataset_output_path = config_path.parent / "trajectories.h5"
    else:
        dataset_output_path = Path(config.dataset_output_path).resolve()

    # Create directory if it doesn't exist.
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Writing data to {dataset_output_path}")
    logging.info(f"Running {config.num_trajectories} trajectories with {num_parallel} parallel problems")

    # Initialize timers for performance measurement
    timers = MPCTimers(
        optimization=Timer("Opt", unit="ms"),
        sim_step=Timer("Sim", unit="ms"),
        reward_compute=Timer("Rew", unit="ms"),
        h5_write=Timer("h5p", unit="ms"),
    )

    with h5py.File(str(dataset_output_path), "w") as f:
        # Create datasets for all fields we're logging.
        qpos_dataset = f.create_dataset(
            "qpos",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nq),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nq),
            dtype="float64",
        )
        qvel_dataset = f.create_dataset(
            "qvel",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nv),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nv),
            dtype="float64",
        )
        control_dataset = f.create_dataset(
            "control",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nu),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nu),
            dtype="float64",
        )

        sensor_dataset = f.create_dataset(
            "sensor",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps, size_data.nsensordata),
            chunks=(size_data.chunk_size, size_data.max_num_task_steps, size_data.nsensordata),
            dtype="float64",
        )
        reward_dataset = f.create_dataset(
            "reward",
            shape=(size_data.num_trajectories, size_data.max_num_task_steps),
            dtype="float",
        )
        trajectory_length_dataset = f.create_dataset(
            "trajectory_length", shape=(size_data.num_trajectories,), chunks=(size_data.chunk_size), dtype="int"
        )

        # Create dataset for collecting rollouts, if applicable.
        if config.store_rollouts:
            rollout_states_dataset = f.create_dataset(
                "rollout_states",
                shape=(
                    size_data.num_trajectories,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nq + size_data.nv,
                ),
                chunks=(
                    size_data.chunk_size,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nq + size_data.nv,
                ),
                dtype="float64",
            )

            rollout_controls_dataset = f.create_dataset(
                "rollout_controls",
                shape=(
                    size_data.num_trajectories,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nu,
                ),
                chunks=(
                    size_data.chunk_size,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                    size_data.num_timesteps,
                    size_data.nu,
                ),
                dtype="float64",
            )
            rollout_rewards_dataset = f.create_dataset(
                "rollout_rewards",
                shape=(
                    size_data.num_trajectories,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                ),
                chunks=(
                    size_data.chunk_size,
                    size_data.max_num_mpc_steps,
                    size_data.num_rollouts,
                ),
                dtype="float64",
            )

        if config.store_viapoints:
            control_viapoints_dataset = f.create_dataset(
                "control_viapoints",
                shape=(size_data.num_trajectories, size_data.max_num_mpc_steps, size_data.num_nodes, size_data.nu),
                chunks=(size_data.chunk_size, size_data.max_num_mpc_steps, size_data.num_nodes, size_data.nu),
                dtype="float64",
            )

        # Store configuration data.
        f.attrs["config_path"] = str(config_path)
        f.attrs["num_parallel"] = num_parallel
        f.attrs["task"] = json_configs["task"]

        # Run main control loop.
        num_batches = (config.num_trajectories + num_parallel - 1) // num_parallel
        traj_idx = 0

        for _batch_idx in tqdm(range(num_batches), desc="Simulating trajectory batches...", unit="batch"):
            # Determine how many trajectories in this batch
            remaining = config.num_trajectories - traj_idx
            batch_size = min(num_parallel, remaining)

            # Run batched MPC - always use full batch for efficiency, only write needed results
            mpc_results = run_mpc_batch(sims, batched_controllers, config, size_data, timers, vis=vis)
            # For partial batch, only use first batch_size results
            mpc_results = mpc_results[:batch_size]

            # Write results to dataset
            timers.h5_write.tic()
            for result in mpc_results:
                if traj_idx >= config.num_trajectories:
                    break
                trajectory_length_dataset[traj_idx] = result["task_step"]
                qpos_dataset[traj_idx] = result["qpos"]
                qvel_dataset[traj_idx] = result["qvel"]
                control_dataset[traj_idx] = result["control"]
                sensor_dataset[traj_idx] = result["sensor"]
                reward_dataset[traj_idx] = result["reward"]
                if config.store_rollouts:
                    rollout_states_dataset[traj_idx] = result["rollout_states"]
                    rollout_controls_dataset[traj_idx] = result["rollout_controls"]
                    rollout_rewards_dataset[traj_idx] = result["rollout_rewards"]
                if config.store_viapoints:
                    control_viapoints_dataset[traj_idx] = result["control_viapoints"]
                traj_idx += 1
            timers.h5_write.toc()

    logging.info(f"Completed {config.num_trajectories} trajectories. Data saved to {dataset_output_path}")

    # Print timing statistics
    logging.info("=== Performance Statistics ===")
    timers.optimization.print_stats(logging.info)
    timers.sim_step.print_stats(logging.info)
    timers.reward_compute.print_stats(logging.info)
    timers.h5_write.print_stats(logging.info)

    # Print rollout backend timing stats
    logging.info("=== Rollout Backend Statistics ===")
    batched_controllers.print_timer_stats()


if __name__ == "__main__":
    run_mpc(tyro.cli(PublicMPCConfig))
