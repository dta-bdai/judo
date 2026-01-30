# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

"""Hierarchical rollout system for Spot tasks using locomotion policy."""

from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjData, MjModel
from omegaconf import DictConfig

from judo.app.structs import MujocoState

from judo.controller.mj_controllers import (
    MjBaseController,
    MjPassThroughController,
    PassThroughLocomotionController,
)
from judo.tasks.base import Task, TaskConfig
from judo.simulation.base import Simulation


class MJSimulatorLLC(Simulation):
    """Hierarchical simulator for Spot tasks with locomotion policy and PD control.

    This class implements hierarchical control where high-level commands are converted
    to low-level torques using a learned locomotion policy and PD controller.

    The control hierarchy:
    - High-level commands: Updated at MPC frequency (~10Hz)
    - Locomotion policy: Runs at ~56Hz (produces target joint positions)
    - PD controller: Runs at physics frequency (~2000Hz, converts targets to torques)
    - Physics: Steps at simulation frequency (dt=0.0005s)
    """

    def __init__(
        self,
        task_name: str = "cylinder_push",
        task_config: TaskConfig | None = None,
        task_registration_cfg: DictConfig | None = None,
        locomotion_controller: BatchedSpotLocomotionController | None = None,
        pd_controller: MjBatchedPdController | None = None,
        num_substeps: int = 2,
        device: str = "cuda:0",
    ):
        """Initialize the hierarchical rollout system.

        Args:
            task_name: Task name to initialize (ignored if task_config is provided).
            task_config: Optional task configuration. If provided, task name is extracted from it.
            task_registration_cfg: Optional task registration config.
            locomotion_controller: Locomotion policy (BatchedSpotLocomotionController or MjPassThroughController).
            pd_controller: PD controller (MjBatchedPdController or MjPassThroughController).
            num_substeps: Number of physics substeps per control step.
            device: PyTorch device for running the policy.
        """
        super().__init__(
            task_name=task_name,
            task_config=task_config,
            task_registration_cfg=task_registration_cfg,
        )

        physics_dt = self.task.model.opt.timestep
        self.num_substeps = num_substeps

        # Initialize locomotion controller
        self.locomotion_controller = locomotion_controller if locomotion_controller else PassThroughLocomotionController()

        # Calculate decimation: how many physics steps per policy evaluation
        self.policy_decimation = max(
            1, int(1.0 / (self.locomotion_controller.target_frequency * physics_dt))
        )

        # Initialize PD controller
        self.pd_controller = pd_controller if pd_controller else MjPassThroughController()

        self.device = device

        # Global step counter for policy decimation
        self.global_step_counter = 0

        # Previous actions (for BatchedSpotLocomotionController, None for MjPassThroughController)
        self.previous_actions: np.ndarray | None = None

    def step(self, command: np.ndarray | None = None) -> None:
        """Step the simulator one task step using the locomotion policy.

        Args:
            command: Command for low-level controller. If None, uses self.control(time) pattern
                from base Simulation class. E.g., for Spot locomotion: (25,) [base_vel(3), arm(7), legs(12), torso(3)].
        """
        # Do not support base Simulation interface: step() with no args uses control spline
        assert command is not None, "MJSimulatorLLC.step() requires a command input."

        # Initialize target_q to avoid unbound variable issues
        target_q = None

        # Rollout for num_substeps physics steps
        for _substep in range(self.num_substeps):
            # Update locomotion policy every policy_decimation steps
            if (
                target_q is None
                or self.global_step_counter % self.policy_decimation == 0
            ):
                # Both PassThroughLocomotionController and BatchedSpotLocomotionController return tuples
                target_q, self.previous_actions = self.locomotion_controller.compute(
                    self.task.data, command, self.previous_actions
                )

            # Compute PD torques (both MjBatchedPdController and MjPassThroughController return single value)
            torques = self.pd_controller.compute(self.task.data, target_q)

            # Apply torques to simulation
            self.task.data.ctrl[:] = torques

            # Step physics
            mujoco.mj_step(self.task.model, self.task.data)

            # Increment global step counter
            self.global_step_counter += 1

    def rollout_trajectory(self, commands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Roll out a trajectory using the locomotion policy.

        Args:
            commands: Commands for low-level controller. Shape (T, nu).
                Shape (T, 25) for Spot locomotion tasks.

        Returns:
            Tuple of (states, sensors):
                - states: State trajectory (T, nq + nv)
                - sensors: Sensor trajectory (T, nsensordata)
        """
        # Validate command format
        if commands.ndim != 2:
            raise ValueError(
                f"Commands must be 2D array (T, nu), got shape {commands.shape}"
            )

        T = commands.shape[0]
        states = np.zeros((T, self.task.model.nq + self.task.model.nv))
        sensors = np.zeros((T, self.task.model.nsensordata))

        for t in range(T):
            self.step(commands[t])
            # Extract state after stepping
            states[t] = np.concatenate(
                [self.task.data.qpos.copy(), self.task.data.qvel.copy()]
            )
            sensors[t] = self.task.data.sensordata.copy()

        return states, sensors

    def reset(self) -> None:
        """Reset internal state (previous actions and step counter)."""
        self.task.reset()
        self.previous_actions = None
        self.global_step_counter = 0

    @property
    def sim_state(self) -> MujocoState:
        """Returns the current simulation state."""
        return MujocoState(
            time=self.task.data.time,
            qpos=self.task.data.qpos,
            qvel=self.task.data.qvel,
            xpos=self.task.data.xpos,
            xquat=self.task.data.xquat,
            mocap_pos=self.task.data.mocap_pos,
            mocap_quat=self.task.data.mocap_quat,
            sim_metadata=self.task.get_sim_metadata(),
        )

    @property
    def timestep(self) -> float:
        """Returns the simulation timestep."""
        return self.task.model.opt.timestep
