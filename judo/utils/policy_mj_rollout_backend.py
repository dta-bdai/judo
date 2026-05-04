# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo rollout backend with locomotion policy support."""

from datetime import timedelta
from pathlib import Path

import numpy as np
from mujoco import MjModel

import mujoco_extensions.closed_loop_rollout as clr
from judo.tasks.spot.closed_loop_features import build_spot_closed_loop_features
from judo.tasks.spot.spot_constants import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME
from judo.utils.rollout_backend import RolloutBackend


class PolicyMJRolloutBackend(RolloutBackend):
    """Rollout backend with C++ mujoco_extensions and ONNX locomotion policy inference.

    For Spot tasks, the command format is a 25-dim vector:
    [base_vel(3), arm(7), legs(12), torso(3)]
    """

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        policy_path: str | Path,
        physics_substeps: int = 2,
    ) -> None:
        """Initialize the policy rollout backend.

        Args:
            model: MuJoCo model for the scene.
            num_threads: Number of parallel rollout threads.
            policy_path: Path to ONNX locomotion policy.
            physics_substeps: Physics steps per control step.
        """
        self.num_threads = num_threads
        self.model = model
        self.physics_substeps = physics_substeps
        self._policy_path = policy_path
        self._thread_pool: clr.ThreadPool | None = None

        self._setup_mujoco_extensions(model, policy_path, num_threads)

    def _setup_mujoco_extensions(self, model: MjModel, policy_path: str | Path, num_threads: int) -> None:
        """Setup the mujoco_extensions C++ rollout backend with ONNX policy."""
        features = build_spot_closed_loop_features(model, policy_path)
        self._systems = clr.create_systems_vector(
            model,
            features,
            str(policy_path),
            num_systems=num_threads,
            physics_substeps=self.physics_substeps,
            duration=timedelta(seconds=DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME),
        )
        self._thread_pool = clr.ThreadPool()

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Conduct parallel rollouts with policy inference.

        Args:
            x0: Initial state, shape (nq+nv,). Will be tiled to num_threads internally.
            controls: Control inputs, shape (num_threads, num_timesteps, cmd_dim).
            last_policy_output: Previous policy outputs, shape (num_threads, 12).
                Required for this backend.

        Returns:
            Tuple of:
                - states: Rolled out states, shape (num_threads, num_timesteps, nq+nv)
                - sensors: Sensor readings, shape (num_threads, num_timesteps, nsensor)
                - policy_outputs: Final policy outputs, shape (num_threads, 12).
        """
        if x0.ndim == 1:
            x0 = np.tile(x0, (self.num_threads, 1))

        if last_policy_output is None:
            raise ValueError("last_policy_output is required for PolicyMJRolloutBackend")

        x0 = np.asarray(x0, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        last_policy_output = np.asarray(last_policy_output, dtype=np.float64)

        if self._thread_pool is None:
            self._thread_pool = clr.ThreadPool()

        out_qpos, out_qvel, sensors, policy_outputs = clr.threaded_rollout(
            self._systems,
            x0[:, : self.model.nq],
            x0[:, self.model.nq :],
            controls,
            last_policy_output,
            self._thread_pool,
        )
        states = np.concatenate((out_qpos, out_qvel), axis=-1)

        return np.array(states), np.array(sensors), np.array(policy_outputs)

    def update(self, num_threads: int) -> None:
        """Update the number of threads.

        Recreates C++ systems for new thread count.

        Args:
            num_threads: New number of parallel threads.
        """
        self.num_threads = num_threads
        self._setup_mujoco_extensions(self.model, self._policy_path, num_threads)
