# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy
from typing import Literal

import numpy as np
from mujoco import MjData, MjModel
from mujoco.rollout import Rollout
import warp as wp
import mujoco_warp as mjw

import warnings
import logging

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

    def __init__(self, model: MjModel, num_threads: int, backend: Literal["mujoco"]) -> None:
        """Initialize the backend with a number of threads."""
        self.backend = backend
        if self.backend == "mujoco":
            self.setup_mujoco_backend(model, num_threads)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        warnings.warn(
            f"Field has no default value to reset to and no override for key. "
            "Its current value remains unchanged.",
            UserWarning,
            stacklevel=2,
        )


    def setup_mujoco_backend(self, model: MjModel, num_threads: int) -> None:
        """Setup the mujoco backend."""
        if self.backend == "mujoco":
            self.model = model
            self.data = MjData(self.model)
            self.num_threads = num_threads
            with wp.ScopedDevice(DEVICE):
                self.mjw_model = mjw.put_model(self.model)
                # Warning: opt.ccd_iterations, currently set to 35, needs to be increased.
                self.mjw_model.opt.ccd_iterations = 60 
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
        
        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[-1] == nu
        assert controls.shape[0] == full_states.shape[0]
        
        # Pre-allocate GPU output buffers to avoid per-iteration CPU-GPU sync
        out_time_wp = wp.zeros((self.num_threads, horizon, 1), dtype=wp.float32)
        out_qpos_wp = wp.zeros((self.num_threads, horizon, nq), dtype=wp.float32)
        out_qvel_wp = wp.zeros((self.num_threads, horizon, nv), dtype=wp.float32)
        out_sensors_wp = wp.zeros((self.num_threads, horizon, nsensordata), dtype=wp.float32)


        # rollout
        if self.backend == "mujoco":
            # mjw.set_state(self.mjw_model, self.mjw_data, full_states_wp, mjw.State.FULLPHYSICS)
            wp.copy(self.mjw_data.time, full_states_wp[:, 1])
            wp.copy(self.mjw_data.qpos, full_states_wp[:, 1:nq+1])
            wp.copy(self.mjw_data.qvel, full_states_wp[:, 1+nq:nq+nv+1])

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
            print(f"Time for full rollout loop: {(t6 - t5)*1e6:.3f} us")
            
            # Copy from GPU to CPU only once at the end
            out_states = np.zeros((self.num_threads, horizon, nq + nv + 1), dtype=np.float32)
            out_states[:, :, :1] = out_time_wp.numpy()
            out_states[:, :, 1:nq+1] = out_qpos_wp.numpy()
            out_states[:, :, nq+1:nq+nv+1] = out_qvel_wp.numpy()
            out_sensors = out_sensors_wp.numpy()
            t7 = time.time()
            print(f"Time to copy final results to CPU: {(t7 - t6)*1e6:.3f} us")
            
            # # Check for NaNs at the end of rollout
            # if np.any(np.isnan(out_states)):
            #     nan_time = np.any(np.isnan(out_states[:, :, 0]), axis=1)
            #     nan_qpos = np.any(np.isnan(out_states[:, :, 1:nq+1]), axis=(1, 2))
            #     nan_qvel = np.any(np.isnan(out_states[:, :, nq+1:]), axis=(1, 2))
            #     print(f"NaN detected in out_states!")
            #     print(f"  Worlds with NaN in time: {np.sum(nan_time)}, indices: {np.where(nan_time)[0][:10]}")
            #     print(f"  Worlds with NaN in qpos: {np.sum(nan_qpos)}, indices: {np.where(nan_qpos)[0][:10]}")
            #     print(f"  Worlds with NaN in qvel: {np.sum(nan_qvel)}, indices: {np.where(nan_qvel)[0][:10]}")
            # if np.any(np.isnan(out_sensors)):
            #     nan_sensors = np.any(np.isnan(out_sensors), axis=(1, 2))
            #     print(f"NaN detected in out_sensors!")
            #     print(f"  Worlds with NaN: {np.sum(nan_sensors)}, indices: {np.where(nan_sensors)[0][:10]}")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        out_states = out_states[..., 1:]  # remove time from state
        return out_states, out_sensors

    def update(self, num_threads: int) -> None:
        """Update the backend with a new number of threads."""
        if self.backend == "mujoco":
            self.setup_mujoco_backend(self.model, num_threads)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
