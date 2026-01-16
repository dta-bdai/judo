# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy
from typing import Literal

import numpy as np
from mujoco import MjData, MjModel
from mujoco.rollout import Rollout
import warp as wp
import mujoco_warp as mjw

# TODO: 
# do a proper rollout function in mjw_test
# is there a rollout funciton in mjw

# NICE TO HAVE
# use the same logic as simulation backend to handle multiple rollout backends
# properly handle device

# DONE:
# how to set the state and the ctrl DONE

NCONMAX = 24
NJMAX = 64
DEVICE = "cuda:0"

def make_model_data_pairs(model: MjModel, num_pairs: int) -> tuple[MjModel, MjData]:
    """Create model/data pairs for mujoco threaded rollout."""
    data = MjData(model)
    return (model, data)

class RolloutBackend:
    """The backend for conducting multithreaded rollouts."""

    def __init__(self, model: MjModel, num_threads: int, backend: Literal["mujoco"]) -> None:
        """Initialize the backend with a number of threads."""
        self.backend = backend
        if self.backend == "mujoco":
            self.setup_mujoco_backend(model, num_threads)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def setup_mujoco_backend(self, model: MjModel, num_threads: int) -> None:
        """Setup the mujoco backend."""
        if self.backend == "mujoco":
            self.model = model
            self.data = MjData(self.model)
            self.num_threads = num_threads
            with wp.ScopedDevice(DEVICE):
                self.mjw_model = mjw.put_model(self.model)
                self.mjw_data = mjw.put_data(self.model, self.data, nworld=self.num_threads, nconmax=NCONMAX, njmax=NJMAX)

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
        """Conduct a rollout depending on the backend."""

        model, data = model_data_pairs
        wp.capture_launch(self.mjw_step_graph)
        wp.synchronize()

        # getting shapes
        nq = model.nq
        nv = model.nv
        nu = model.nu
        nsensordata = model.nsensordata
        horizon = controls.shape[1]


        # the state passed into mujoco's rollout function includes the time
        # shape = (num_rollouts, num_states + 1)
        x0_batched = np.tile(x0, (self.num_threads, 1))
        full_states = np.concatenate([time.time() * np.ones((self.num_threads, 1)), x0_batched], axis=-1)
        
        full_states_wp = wp.array(full_states, dtype=wp.float32)
        controls_wp = wp.array(controls, dtype=wp.float32)

        out_states = np.zeros((self.num_threads, horizon, nq + nv + 1), dtype=np.float32)
        out_sensors = np.zeros((self.num_threads, horizon, nsensordata), dtype=np.float32)

        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[-1] == nu
        assert controls.shape[0] == full_states.shape[0]

        # rollout
        if self.backend == "mujoco":
            # _states, _out_sensors = self.rollout_func(ms, ds, full_states, controls)

            mjw.set_state(self.mjw_model, self.mjw_data, full_states_wp, mjw.State.FULLPHYSICS)
            for t in range(horizon):
                mjw.set_state(self.mjw_model, self.mjw_data, controls_wp[:, t, :], mjw.State.CTRL)
                mjw.step(self.mjw_model, self.mjw_data)
                out_states[:, t, 0] = self.mjw_data.time.numpy()
                out_states[:, t, 1:nq+1] = self.mjw_data.qpos.numpy()
                out_states[:, t, nq+1:nq+nv+1] = self.mjw_data.qvel.numpy()
                out_sensors[:, t, :] = self.mjw_data.sensordata.numpy()

        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        # out_states = np.array(_states)[..., 1:]  # remove time from state
        # out_sensors = np.array(_out_sensors)

        out_states = out_states[..., 1:]  # remove time from state
        return out_states, out_sensors

    def update(self, num_threads: int) -> None:
        """Update the backend with a new number of threads."""
        if self.backend == "mujoco":
            self.setup_mujoco_backend(self.model, num_threads)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
