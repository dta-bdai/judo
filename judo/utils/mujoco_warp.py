# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

import time
import warnings
from typing import Literal

import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco import MjData, MjModel

# TODO:
# do a proper rollout function in mjw_test
# is there a rollout funciton in mjw

# NICE TO HAVE
# use the same logic as simulation backend to handle multiple rollout backends
# properly handle device

# DONE:
# how to set the state and the ctrl DONE

NCONMAX = 256
NJMAX = 256
DEVICE = "cuda:0"


def make_model_data_pairs(model: MjModel, num_pairs: int) -> tuple[MjModel, MjData]:
    """Create model/data pairs for mujoco threaded rollout."""
    data = MjData(model)
    return (model, data)


class RolloutBackend:
    """The backend for conducting multithreaded rollouts."""

    def __init__(
        self, model: MjModel, num_threads: int, num_problems: int = 1, backend: Literal["mujoco"] = "mujoco"
    ) -> None:
        """Initialize the backend with a number of threads."""
        self.backend = backend
        if self.backend == "mujoco":
            self.setup_mujoco_backend(model, num_threads, num_problems)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        warnings.warn(
            "Field has no default value to reset to and no override for key. Its current value remains unchanged.",
            UserWarning,
            stacklevel=2,
        )

    def setup_mujoco_backend(self, model: MjModel, num_threads: int, num_problems: int) -> None:
        """Setup the mujoco backend."""
        if self.backend == "mujoco":
            self.model = model
            self.data = MjData(self.model)
            self.num_threads = num_threads
            self.num_problems = num_problems
            self.num_worlds = num_threads * num_problems
            with wp.ScopedDevice(DEVICE):
                self.mjw_model = mjw.put_model(self.model)
                # Warning: opt.ccd_iterations, currently set to 35, needs to be increased.
                self.mjw_model.opt.ccd_iterations = 60
                self.mjw_data = mjw.put_data(
                    self.model, self.data, nworld=self.num_worlds, nconmax=NCONMAX, njmax=NJMAX
                )

            # capture the whole function as a CUDA graph
            with wp.ScopedCapture() as capture:
                mjw.step(self.mjw_model, self.mjw_data)
            self.mjw_step_graph = capture.graph
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def rollout(
        self,
        model_data_pairs: tuple[MjModel, MjData],
        x0: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout depending on the backend.

        Supports both single-problem and multi-problem modes:
        - Single problem: x0 shape (x0_dim,), controls shape (num_threads, horizon, nu)
        - Multi-problem: x0 shape (num_problems, x0_dim), controls shape (num_problems*num_threads, horizon, nu)

        Args:
            model_data_pairs: Tuple of (MjModel, MjData).
            x0: Initial state(s). Shape (x0_dim,) for single problem, (num_problems, x0_dim) for multi-problem.
            controls: Control inputs. Shape (num_threads, horizon, nu) for single problem,
                (num_problems*num_threads, horizon, nu) for multi-problem.

        Returns:
            Tuple of (states, sensors):
            - states: shape (num_worlds, horizon, nq + nv)
            - sensors: shape (num_worlds, horizon, nsensordata)
        """
        model, data = model_data_pairs
        nq = model.nq
        nv = model.nv
        nu = model.nu
        nsensordata = model.nsensordata
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

        # Prepare full states with time
        full_states = np.concatenate([time.time() * np.ones((num_worlds, 1)), x0_batched], axis=-1)
        full_states_wp = wp.array(full_states, dtype=wp.float32)
        controls_wp = wp.array(controls, dtype=wp.float32)

        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[-1] == nu
        assert controls.shape[0] == num_worlds

        # Pre-allocate GPU output buffers
        out_time_wp = wp.zeros((num_worlds, horizon, 1), dtype=wp.float32)
        out_qpos_wp = wp.zeros((num_worlds, horizon, nq), dtype=wp.float32)
        out_qvel_wp = wp.zeros((num_worlds, horizon, nv), dtype=wp.float32)
        out_sensors_wp = wp.zeros((num_worlds, horizon, nsensordata), dtype=wp.float32)

        # Execute rollout
        if self.backend == "mujoco":
            wp.copy(self.mjw_data.time, full_states_wp[:, 0])
            wp.copy(self.mjw_data.qpos, full_states_wp[:, 1 : nq + 1])
            wp.copy(self.mjw_data.qvel, full_states_wp[:, 1 + nq : nq + nv + 1])

            t5 = time.time()
            for t in range(horizon):
                # Set control - use wp.copy() for speed (no kernel recompilation)
                wp.copy(self.mjw_data.ctrl, controls_wp[:, t, :])

                # Step simulation
                wp.capture_launch(self.mjw_step_graph)

                # Copy results to GPU buffers (no CPU sync needed)
                wp.copy(out_time_wp[:, t, 0], self.mjw_data.time)
                wp.copy(out_qpos_wp[:, t, :], self.mjw_data.qpos)
                wp.copy(out_qvel_wp[:, t, :], self.mjw_data.qvel)
                wp.copy(out_sensors_wp[:, t, :], self.mjw_data.sensordata)

            # Single sync at the end of rollout
            wp.synchronize()
            t6 = time.time()
            print(f"Time for rollout loop ({num_worlds} worlds): {(t6 - t5) * 1e6:.3f} us")

            # Copy from GPU to CPU
            out_states = np.zeros((num_worlds, horizon, nq + nv + 1), dtype=np.float32)
            out_states[:, :, :1] = out_time_wp.numpy()
            out_states[:, :, 1 : nq + 1] = out_qpos_wp.numpy()
            out_states[:, :, nq + 1 : nq + nv + 1] = out_qvel_wp.numpy()
            out_sensors = out_sensors_wp.numpy()
            t7 = time.time()
            print(f"Time to copy results to CPU: {(t7 - t6) * 1e6:.3f} us")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        out_states = out_states[..., 1:]  # remove time from state
        return out_states, out_sensors

    def update(self, num_threads: int, num_problems: int = 1) -> None:
        """Update the backend with a new number of threads."""
        if self.backend == "mujoco":
            self.setup_mujoco_backend(self.model, num_threads, num_problems)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
