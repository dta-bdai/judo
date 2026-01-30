# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

import warnings
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco import MjData, MjModel

from judo.controller.mj_controllers import (
    MjBaseController,
    MjBatchedPdController,
    MjPassThroughController,
    PassThroughLocomotionController,
)
from judo.controller.spot_constants import K_Q_P_RL, K_QD_P_RL, TORQUE_LIMITS_RL
from judo.controller.spot_locomotion import BatchedSpotLocomotionController
from judo.utils.timer import Timer

# TODO:
# do a proper rollout function in mjw_test
# is there a rollout funciton in mjw
# properly handle device

NCONMAX = 256
NJMAX = 256
DEVICE = "cuda:0"


def make_model_data_pairs(model: MjModel, num_pairs: int) -> tuple[MjModel, MjData]:
    """Create model/data pairs for mujoco threaded rollout."""
    data = MjData(model)
    return (model, data)


class RolloutBackend:
    """GPU-accelerated rollout backend using mujoco_warp.

    Supports two modes:
    - Direct control: controls directly set actuator commands (default)
    - Hierarchical control: controls are processed through locomotion policy + PD controller
    """

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        num_problems: int = 1,
        locomotion_controller: MjBaseController | None = None,
        pd_controller: MjBaseController | None = None,
        device: str = DEVICE,
    ) -> None:
        """Initialize the backend with optional hierarchical control.

        Args:
            model: MuJoCo model.
            num_threads: Number of parallel rollouts per problem.
            num_problems: Number of independent problems.
            locomotion_controller: Locomotion policy (BatchedSpotLocomotionController or MjPassThroughController).
            pd_controller: PD controller (MjBatchedPdController or MjPassThroughController).
            device: Device for GPU operations.
        """
        assert device != "cpu", "RolloutBackend requires a GPU device."

        # Setup MuJoCo backend
        self.device = device
        self.model = model
        self.data = MjData(self.model)
        self.num_threads = num_threads
        self.num_problems = num_problems
        self.num_worlds = num_threads * num_problems

        with wp.ScopedDevice(device):
            self.mjw_model = mjw.put_model(self.model)
            self.mjw_model.opt.ccd_iterations = 60
            self.mjw_data = mjw.put_data(
                self.model,
                self.data,
                nworld=self.num_worlds,
                nconmax=NCONMAX,
                njmax=NJMAX,
            )

        # Capture CUDA graph for step function
        with wp.ScopedCapture() as capture:
            mjw.step(self.mjw_model, self.mjw_data)
        self.mjw_step_graph = capture.graph

        # Initialize hierarchical control (optional)
        self.locomotion_controller = locomotion_controller if locomotion_controller else PassThroughLocomotionController()
        self.pd_controller = pd_controller if pd_controller else MjPassThroughController()

        # Calculate policy decimation
        physics_dt = model.opt.timestep
        self.policy_decimation = max(
            1, int(1.0 / (self.locomotion_controller.target_frequency * physics_dt))
        )
        self.global_step_counter = 0

        # Initial previous actions for rollouts (owned by RolloutBackend, synced from sims before each rollout)
        # Shape: (num_worlds, 12) after broadcasting, None for PassThroughLocomotionController
        self.init_previous_actions: torch.Tensor | None = None

        # Initialize timers for performance measurement
        self.timer_cpu_to_gpu = Timer("CPU->GPU", unit="ms")
        self.timer_rollout = Timer("Rollout ", unit="ms")
        self.timer_gpu_to_cpu = Timer("GPU->CPU", unit="ms")

        warnings.warn(
            "Field has no default value to reset to and no override for key. Its current value remains unchanged.",
            UserWarning,
            stacklevel=2,
        )

    def set_init_previous_actions(self, previous_actions_list: list[np.ndarray | None]) -> None:
        """Set initial previous actions from list of per-problem states (synced from sims).

        Broadcasts per-problem actions to all threads, so self.init_previous_actions is
        ready to use in rollout() without further processing.

        Args:
            previous_actions_list: List of previous_actions, one per problem. Each element is either:
                                  - None for PassThroughLocomotionController (stateless)
                                  - np.ndarray (12,) for BatchedSpotLocomotionController
                                  If all elements are None, sets self.init_previous_actions to None.
        """
        if all(pa is None for pa in previous_actions_list):
            self.init_previous_actions = None
        else:
            # Stack into (num_problems, 12) and broadcast to all threads
            pa_np = np.stack([pa for pa in previous_actions_list if pa is not None], axis=0)
            pa_broadcast = np.repeat(pa_np, self.num_threads, axis=0)  # (num_problems, 12) -> (num_worlds, 12)

            # Convert to torch
            self.init_previous_actions = torch.from_numpy(pa_broadcast).float().to(self.device)

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a GPU-accelerated rollout using mujoco_warp.

        Supports both single-problem and multi-problem modes:
        - Single problem: x0 shape (nq + nv,), controls shape (num_threads, horizon, nu)
        - Multi-problem: x0 shape (num_problems, nq + nv), controls shape (num_problems*num_threads, horizon, nu)

        Note: Previous actions must be set via set_init_previous_actions() before calling this method.

        Args:
            x0: Initial state(s) as [qpos, qvel] (no time).
                Shape (nq + nv,) for single problem, (num_problems, nq + nv) for multi-problem.
                qpos contains generalized positions (including floating base if applicable).
                qvel contains generalized velocities.
            controls: Control inputs. Shape (num_threads, horizon, nu) for single problem,
                (num_problems*num_threads, horizon, nu) for multi-problem.

        Returns:
            Tuple of (states, sensors):
            - states: shape (num_worlds, horizon, nq + nv)
            - sensors: shape (num_worlds, horizon, nsensordata)
        """
        nq = self.model.nq
        nv = self.model.nv
        nu = self.model.nu
        nsensordata = self.model.nsensordata
        horizon = controls.shape[1]

        # Handle both single-problem and multi-problem cases
        if x0.ndim == 1:
            # Single problem: x0 shape (x0_dim,) -> tile to (num_threads, x0_dim)
            x0_batched = np.tile(x0, (self.num_threads, 1))
        else:
            # Multi-problem: x0 shape (num_problems, x0_dim) -> tile each row and stack
            # Result: (num_problems * num_threads, x0_dim)
            x0_batched = np.repeat(x0, self.num_threads, axis=0)

        num_worlds = x0_batched.shape[0]

        # Prepare full states with time (use 0.0 as placeholder time)
        full_states = np.concatenate([np.zeros((num_worlds, 1)), x0_batched], axis=-1)

        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[-1] == nu
        assert controls.shape[0] == num_worlds

        # CPU -> GPU copy
        self.timer_cpu_to_gpu.tic()
        full_states_wp = wp.array(full_states, dtype=wp.float32)
        controls_wp = wp.array(controls, dtype=wp.float32)

        # Pre-allocate GPU output buffers
        out_qpos_wp = wp.zeros((num_worlds, horizon, nq), dtype=wp.float32)
        out_qvel_wp = wp.zeros((num_worlds, horizon, nv), dtype=wp.float32)
        out_sensors_wp = wp.zeros((num_worlds, horizon, nsensordata), dtype=wp.float32)

        wp.copy(self.mjw_data.time, full_states_wp[:, 0])
        wp.copy(self.mjw_data.qpos, full_states_wp[:, 1 : nq + 1])
        wp.copy(self.mjw_data.qvel, full_states_wp[:, 1 + nq : nq + nv + 1])
        wp.synchronize()
        self.timer_cpu_to_gpu.toc()

        # GPU rollout loop
        # Unified loop works for both direct and hierarchical control via passthrough controllers
        self.timer_rollout.tic()

        # Initialize state tracking
        qpos_wp = wp.zeros((num_worlds, nq), dtype=wp.float32)
        qvel_wp = wp.zeros((num_worlds, nv), dtype=wp.float32)
        target_q_torch = None

        # Initialize previous_actions for this rollout (local variable that evolves during simulation)
        # self.init_previous_actions is the initial state from real sims
        previous_actions_torch = self.init_previous_actions

        for t in range(horizon):
            # Synchronize before reading state
            wp.synchronize()

            # Convert warp arrays to torch tensors (zero-copy, cheap even for direct control)
            qpos_torch = wp.to_torch(qpos_wp)
            qvel_torch = wp.to_torch(qvel_wp)
            cmd_torch = wp.to_torch(controls_wp[:, t, :])

            # Update locomotion policy
            if (
                target_q_torch is None
                or self.global_step_counter % self.policy_decimation == 0
            ):
                # Both PassThroughLocomotionController and BatchedSpotLocomotionController return tuples
                target_q_torch, previous_actions_torch = self.locomotion_controller.compute_batch(
                    cmd_torch, qpos_torch, qvel_torch, previous_actions_torch
                )

            # Compute PD torques
            q_torch = qpos_torch[:, 7:]  # Skip base pose
            dq_torch = qvel_torch[:, 6:]  # Skip base velocity
            tau = self.pd_controller.compute_batch(target_q_torch, q_torch, dq_torch)

            # Convert torques to warp array (zero-copy)
            torques_wp = wp.from_torch(tau)

            # Set control
            wp.copy(self.mjw_data.ctrl, torques_wp)

            # Step simulation
            wp.capture_launch(self.mjw_step_graph)

            # Increment global step counter
            self.global_step_counter += 1

            # Copy results to GPU buffers
            wp.copy(qpos_wp, self.mjw_data.qpos)
            wp.copy(qvel_wp, self.mjw_data.qvel)
            wp.copy(out_qpos_wp[:, t, :], self.mjw_data.qpos)
            wp.copy(out_qvel_wp[:, t, :], self.mjw_data.qvel)
            wp.copy(out_sensors_wp[:, t, :], self.mjw_data.sensordata)

        # Single sync at the end of rollout
        wp.synchronize()
        self.timer_rollout.toc()

        # GPU -> CPU copy
        self.timer_gpu_to_cpu.tic()
        out_states = np.zeros((num_worlds, horizon, nq + nv), dtype=np.float32)
        out_states[:, :, :nq] = out_qpos_wp.numpy()
        out_states[:, :, nq : nq + nv] = out_qvel_wp.numpy()
        out_sensors = out_sensors_wp.numpy()
        self.timer_gpu_to_cpu.toc()

        return out_states, out_sensors

    def update(self, num_threads: int, num_problems: int = 1) -> None:
        """Update the backend with a new number of threads."""
        self.num_threads = num_threads
        self.num_problems = num_problems
        self.num_worlds = num_threads * num_problems

        with wp.ScopedDevice(DEVICE):
            self.mjw_model = mjw.put_model(self.model)
            self.mjw_model.opt.ccd_iterations = 60
            self.mjw_data = mjw.put_data(
                self.model,
                self.data,
                nworld=self.num_worlds,
                nconmax=NCONMAX,
                njmax=NJMAX,
            )

        # Recapture CUDA graph for step function
        with wp.ScopedCapture() as capture:
            mjw.step(self.mjw_model, self.mjw_data)
        self.mjw_step_graph = capture.graph

    def print_timer_stats(self) -> None:
        """Print timing statistics for all rollout operations."""
        self.timer_cpu_to_gpu.print_stats()
        self.timer_rollout.print_stats()
        self.timer_gpu_to_cpu.print_stats()

    def reset_timers(self) -> None:
        """Reset all timing statistics."""
        self.timer_cpu_to_gpu.reset()

    def reset(self) -> None:
        """Reset internal state (controllers and step counter)."""
        self.locomotion_controller.reset()
        self.pd_controller.reset()
        self.global_step_counter = 0
        self.timer_rollout.reset()
        self.timer_gpu_to_cpu.reset()
