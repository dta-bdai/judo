# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import mujoco
import numpy as np
import tyro
from tqdm import tqdm

from judo.hardware.judo_app.io import JudoIOContext, StateBufferKeys
from judo.mujoco_extensions.policy_rollout import threaded_rollout
from judo.sampling_controller.sampling_controller import SamplingController
from judo.tasks.spot.spot_constants import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME, RL_LOCOMOTION_ACTION_LENGTH
from judo.tasks import get_registered_tasks
from judo.tasks.spot.spot_base import SpotBase
from judo.utils.json_serializer import decode_config


@dataclass
class MPCConfig:
    """Config class for MPC runs."""

    policy_config_path: Path  # Path to .json containing task/controller config for MPC run.
    dataset_output_path: Optional[Path] = None
    num_trajectories: int = 2
    max_time_seconds: float = 30.0  # Maximum simulation time in seconds
    chunk_size: int = 1
    use_config_control_freq: bool = True  # TODO(pculbert): allow the controller to spin faster than sim.
    store_rollouts: bool = True
    store_viapoints: bool = True


def load_policy_from_config_path(policy_config_path: Path) -> tuple[SpotBase, SamplingController, dict[str, Any]]:
    """Load task and controller from config path."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

    # Load policy config from file.
    with open(policy_config_path, "r") as f:
        logging.info(f"Loading policy config from {policy_config_path}")
        policy_config = json.load(f, object_hook=decode_config)

    # Load controller/task objects for this experiment.
    registered_tasks = get_registered_tasks()
    task_class, _ = registered_tasks[policy_config["task"]]

    judo_context = JudoIOContext(multiprocessing.Manager())
    task = task_class(judo_context)
    task.judo_context.reward_config_dict.update(policy_config["task_config"])
    task.judo_context.goal_config_dict.update({"goal_position": policy_config["goal_config"]["goal_position"]})
    task.reset()

    controller = SamplingController(judo_context, task)
    controller.judo_context.control_config_dict.update(policy_config["controller_config"])
    controller.reset()

    return task, controller, policy_config  # type:ignore


def is_terminated(task: SpotBase, config: dict) -> bool:
    """Defines if the current state is terminal."""
    return False


def sim_step(task: SpotBase, controller: SamplingController) -> None:
    """Spot policy physics evaluation. Evaluates policy and steps simulation forward."""
    # Read current action from spline.
    command = controller.spline(task.data.time)
    command = task.apply_selection_mask(command)
    current_spot_command = command

    current_full_command = np.copy(task.default_policy_command)
    current_full_command[task.command_mask] = current_spot_command

    current_state = np.concatenate([task.data.qpos, task.data.qvel])

    last_locomotion_action = task.judo_context.state_buffer[StateBufferKeys.last_locomotion_action]
    policy_outputs = np.broadcast_to(last_locomotion_action, (1, last_locomotion_action.shape[-1]))

    next_state, _, next_policy_output = threaded_rollout(
        controller.models,
        current_state[None],
        current_full_command[None, None],
        policy_outputs,
        1,
        task.physics_substeps,
        DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
    )

    assert len(next_state) == 1
    assert next_state[0].shape == (2, task.model.nq + task.model.nv)

    task.data.qpos[:] = next_state[0][-1][: task.model.nq]
    task.data.qvel[:] = next_state[0][-1][task.model.nq :]
    mujoco.mj_forward(task.model, task.data)

    task.judo_context.state_buffer[StateBufferKeys.last_locomotion_action] = next_policy_output[0]
    # Advance time since we don't use mj_step
    task.data.time += task.model.opt.timestep * task.physics_substeps


def num_mpc_steps(max_num_timesteps: int, control_freq: float, task_dt: float) -> tuple[int, int]:
    control_dt = 1.0 / control_freq  # seconds per frame
    num_sim_steps_per_mpc = int(control_dt / task_dt)
    num_mpc_steps = np.ceil(max_num_timesteps / num_sim_steps_per_mpc).astype(int)
    return num_sim_steps_per_mpc, num_mpc_steps


def run_single_mpc(
    task: SpotBase, controller: SamplingController, config: MPCConfig, max_num_timesteps: int
) -> dict[str, np.ndarray | int]:
    controller_config = dict(controller.judo_context.control_config_dict)
    task_config = dict(task.judo_context.reward_config_dict)

    if config.use_config_control_freq:
        sim_steps_per_control_update, max_num_mpc_steps = num_mpc_steps(
            max_num_timesteps, controller_config["control_freq"], task.dt
        )
    else:
        sim_steps_per_control_update = 1

    if config.store_rollouts or config.store_viapoints:
        max_num_mpc_steps = np.ceil(max_num_timesteps / sim_steps_per_control_update).astype(int)

    # Sample new initial condition + goal.
    task.reset()
    controller.reset()

    task.judo_context.state_buffer[StateBufferKeys.last_locomotion_action] = np.zeros(RL_LOCOMOTION_ACTION_LENGTH)

    # TODO(pculbert): think through controller reset behavior.

    # Allocate storage for full trajectory (padded with nans).
    qpos_traj = np.full((max_num_timesteps, task.model.nq), np.nan, dtype="float64")
    qvel_traj = np.full((max_num_timesteps, task.model.nv), np.nan, dtype="float64")
    control_traj = np.full((max_num_timesteps, task.nu), np.nan, dtype="float64")
    obs_traj = np.full((max_num_timesteps, task.model.nsensordata), np.nan, dtype="float64")
    reward_traj = np.full((max_num_timesteps,), np.nan, dtype="float64")

    if config.store_viapoints:
        control_viapoints = np.full(
            (max_num_mpc_steps, controller_config["num_nodes"], task.nu), np.nan, dtype="float64"
        )
    if config.store_rollouts:
        num_control_timesteps = controller.num_timesteps(controller_config["horizon"])
        num_state_timesteps = num_control_timesteps * controller.num_physics_substeps
        num_rollouts = controller_config["num_rollouts"]
        state_dim = task.model.nq + task.model.nv

        rollout_states = np.full(
            (
                max_num_mpc_steps,
                num_rollouts,
                num_state_timesteps,
                state_dim,
            ),
            np.nan,
            dtype="float64",
        )
        rollout_controls = np.full(
            (
                max_num_mpc_steps,
                num_rollouts,
                num_control_timesteps,
                task.nu,
            ),
            np.nan,
            dtype="float64",
        )
        candidate_controls = np.full(
            (
                max_num_mpc_steps,
                controller_config["num_rollouts"],
                controller_config["num_nodes"],
                task.nu,
            ),
            np.nan,
            dtype="float64",
        )
        rollout_rewards = np.full(
            (
                max_num_mpc_steps,
                controller_config["num_rollouts"],
            ),
            np.nan,
            dtype="float64",
        )
        control_timesteps = np.full(
            (max_num_mpc_steps,),
            np.nan,
            dtype="int",
        )

    if config.store_viapoints or config.store_rollouts:
        mpc_step = 0  # counter for how many control updates have occurred.

    timesteps = 0
    for t in range(max_num_timesteps):
        # Write out current sim state.
        # TODO(pculbert): possibly refactor to use cpp rollout for however many steps we're
        # simulating with the same control.
        qpos_traj[t] = task.data.qpos
        qvel_traj[t] = task.data.qvel
        control_traj[t] = controller.action(task.data.time)
        obs_traj[t] = task.data.sensordata
        reward_traj[t] = task.reward(
            np.concatenate([qpos_traj[t], qvel_traj[t]], axis=0)[None, None],
            obs_traj[t][None, None],
            control_traj[t][None, None],
        )[0]

        if t % sim_steps_per_control_update == 0:
            task.judo_context.state_buffer[StateBufferKeys.full_state] = np.concatenate(
                [task.data.qpos, task.data.qvel]
            )
            task.judo_context.state_buffer[StateBufferKeys.time] = task.data.time
            controller.update_action()
            if config.store_rollouts:
                control_timesteps[mpc_step] = t
                rollout_states[mpc_step] = controller.states
                rollout_controls[mpc_step] = controller.rollout_controls
                candidate_controls[mpc_step] = controller.candidate_controls
                rollout_rewards[mpc_step] = controller.rewards

            if config.store_viapoints:
                control_viapoints[mpc_step] = controller.spline.y

            if config.store_rollouts or config.store_viapoints:
                mpc_step += 1

        sim_step(task, controller)
        timesteps += 1

        if is_terminated(task, task_config):
            break

    mpc_result: dict[str, np.ndarray | int] = {
        "timesteps": timesteps,
        "qpos": qpos_traj,
        "qvel": qvel_traj,
        "action": control_traj,
        "sensors": obs_traj,
        "reward": reward_traj,
    }
    if config.store_rollouts:
        mpc_result["control_timesteps"] = control_timesteps
        mpc_result["rollout_states"] = rollout_states
        mpc_result["rollout_controls"] = rollout_controls
        mpc_result["candidate_controls"] = candidate_controls
        mpc_result["rollout_rewards"] = rollout_rewards
    if config.store_viapoints:
        mpc_result["control_viapoints"] = control_viapoints

    return mpc_result


def run_mpc(config: MPCConfig) -> None:
    """Runs MPC and writes closed-loop trajectories to file."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

    task, controller, policy_config = load_policy_from_config_path(config.policy_config_path)
    controller_config = dict(controller.judo_context.control_config_dict)

    # Convert max_time_seconds to max_num_timesteps based on task.dt
    max_num_timesteps = int(np.ceil(config.max_time_seconds / task.dt))

    if config.dataset_output_path is None:
        # By default, save dataset to same directory as config with task name.
        task_name = policy_config["task"]
        dataset_output_path = config.policy_config_path.parent / f"{task_name}_trajectories.h5"
    else:
        dataset_output_path = config.dataset_output_path

    if config.use_config_control_freq:
        sim_steps_per_control_update = int((1 / controller_config["control_freq"]) / task.dt)
    else:
        sim_steps_per_control_update = 1

    logging.info(f"Writing data to {dataset_output_path}")
    logging.info(f"Simulating {config.max_time_seconds}s ({max_num_timesteps} timesteps at dt={task.dt}s)")

    with h5py.File(str(dataset_output_path), "w") as f:
        # Create datasets for all fields we're logging.
        qpos_dataset = f.create_dataset(
            "qpos",
            shape=(config.num_trajectories, max_num_timesteps, task.model.nq),
            chunks=(config.chunk_size, max_num_timesteps, task.model.nq),
            dtype="float64",
        )
        qvel_dataset = f.create_dataset(
            "qvel",
            shape=(config.num_trajectories, max_num_timesteps, task.model.nv),
            chunks=(config.chunk_size, max_num_timesteps, task.model.nv),
            dtype="float64",
        )
        control_dataset = f.create_dataset(
            "controls",
            shape=(config.num_trajectories, max_num_timesteps, task.nu),
            chunks=(config.chunk_size, max_num_timesteps, task.nu),
            dtype="float64",
        )

        obs_dataset = f.create_dataset(
            "obs",
            shape=(
                config.num_trajectories,
                max_num_timesteps,
                task.model.nsensordata,
            ),
            chunks=(config.chunk_size, max_num_timesteps, task.model.nsensordata),
            dtype="float64",
        )
        reward_dataset = f.create_dataset(
            "reward",
            shape=(config.num_trajectories, max_num_timesteps),
            dtype="float",
        )
        trajectory_length_dataset = f.create_dataset(
            "trajectory_length", shape=(config.num_trajectories,), chunks=(config.chunk_size), dtype="int"
        )

        if config.store_rollouts or config.store_viapoints:
            max_num_mpc_steps = np.ceil(max_num_timesteps / sim_steps_per_control_update).astype(int)

        # Create dataset for collecting rollouts, if applicable.
        if config.store_rollouts:
            num_control_timesteps = controller.num_timesteps(controller_config["horizon"])
            num_state_timesteps = num_control_timesteps * controller.num_physics_substeps
            num_rollouts = controller_config["num_rollouts"]
            state_dim = task.model.nq + task.model.nv

            rollout_states_dataset = f.create_dataset(
                "rollout_states",
                shape=(
                    config.num_trajectories,
                    max_num_mpc_steps,
                    num_rollouts,
                    num_state_timesteps,
                    state_dim,
                ),
                chunks=(
                    config.chunk_size,
                    max_num_mpc_steps,
                    num_rollouts,
                    num_state_timesteps,
                    state_dim,
                ),
                dtype="float64",
            )
            rollout_controls_dataset = f.create_dataset(
                "rollout_controls",
                shape=(
                    config.num_trajectories,
                    max_num_mpc_steps,
                    num_rollouts,
                    num_control_timesteps,
                    task.nu,
                ),
                chunks=(
                    config.chunk_size,
                    max_num_mpc_steps,
                    num_rollouts,
                    num_control_timesteps,
                    task.nu,
                ),
                dtype="float64",
            )
            candidate_controls_dataset = f.create_dataset(
                "candidate_controls",
                shape=(
                    config.num_trajectories,
                    max_num_mpc_steps,
                    controller_config["num_rollouts"],
                    controller_config["num_nodes"],
                    task.nu,
                ),
                chunks=(
                    config.chunk_size,
                    max_num_mpc_steps,
                    controller_config["num_rollouts"],
                    controller_config["num_nodes"],
                    task.nu,
                ),
                dtype="float64",
            )
            rollout_rewards_dataset = f.create_dataset(
                "rollout_rewards",
                shape=(
                    config.num_trajectories,
                    max_num_mpc_steps,
                    controller_config["num_rollouts"],
                ),
                chunks=(
                    config.chunk_size,
                    max_num_mpc_steps,
                    controller_config["num_rollouts"],
                ),
                dtype="float64",
            )

        if config.store_viapoints:
            control_viapoints_dataset = f.create_dataset(
                "control_viapoints",
                shape=(config.num_trajectories, max_num_mpc_steps, controller_config["num_nodes"], task.nu),
                chunks=(config.chunk_size, max_num_mpc_steps, controller_config["num_nodes"], task.nu),
                dtype="float64",
            )

        # Store configuration data.
        f.attrs["config_path"] = str(config.policy_config_path)

        # Run main control loop.
        for traj_idx in tqdm(range(config.num_trajectories), desc="Simulating trajectories...", unit="trajectory"):
            mpc_result = run_single_mpc(task, controller, config, max_num_timesteps)

            trajectory_length_dataset[traj_idx] = mpc_result["timesteps"]
            qpos_dataset[traj_idx] = mpc_result["qpos"]
            qvel_dataset[traj_idx] = mpc_result["qvel"]
            control_dataset[traj_idx] = mpc_result["action"]
            obs_dataset[traj_idx] = mpc_result["sensors"]
            reward_dataset[traj_idx] = mpc_result["reward"]
            if config.store_rollouts:
                rollout_states_dataset[traj_idx] = mpc_result["rollout_states"]
                rollout_controls_dataset[traj_idx] = mpc_result["rollout_controls"]
                candidate_controls_dataset[traj_idx] = mpc_result["candidate_controls"]
                rollout_rewards_dataset[traj_idx] = mpc_result["rollout_rewards"]
            if config.store_viapoints:
                control_viapoints_dataset[traj_idx] = mpc_result["control_viapoints"]


if __name__ == "__main__":
    run_mpc(tyro.cli(MPCConfig))
