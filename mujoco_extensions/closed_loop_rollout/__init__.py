# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Closed-loop rollout compatibility layer.

This module mirrors the Starfish closed_loop_rollout Python API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import importlib
from os import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from mujoco import MjModel


@dataclass
class Feature:
    """Compatibility placeholder for feature descriptors."""

    name: str
    size: int


class Policy:
    """Compatibility placeholder for policy metadata."""

    def __init__(self, policy_path: str | Path) -> None:
        self.policy_path = str(policy_path)


class ThreadPool:
    """Compatibility placeholder for Starfish thread pool API."""

    def __init__(self, thread_number: int | None = None) -> None:
        self.thread_number = thread_number or cpu_count() or 1


class ClosedLoopSystems(list):
    """List-like systems container with rollout metadata."""

    physics_substeps: int
    cutoff_time: float


def make_mujoco_system_from_model(model: MjModel) -> MjModel:
    """Return the model as a compatibility system handle."""

    return model


def create_systems_vector(
    model: MjModel,
    features: list[Feature],
    policy_path: str,
    num_systems: int,
    physics_substeps: int,
    duration: timedelta,
) -> ClosedLoopSystems:
    """Create rollout systems using the native closed_loop_rollout backend."""
    try:
        _pybind = importlib.import_module("mujoco_extensions.closed_loop_rollout.closed_loop_rollout_pybind")

        _create_systems = _pybind.closed_loop_rollout.create_systems_vector
    except ImportError as e:
        raise ImportError("mujoco_extensions closed_loop_rollout backend is not built. Run: pixi run build") from e

    del features  # Features are consumed by Starfish API but not by this backend.
    systems = ClosedLoopSystems(_create_systems(model, str(policy_path), num_systems))
    systems.physics_substeps = int(physics_substeps)
    systems.cutoff_time = float(duration.total_seconds())
    return systems


def threaded_rollout(
    systems: list[Any],
    qpos: np.ndarray,
    qvel: np.ndarray,
    cmds: np.ndarray,
    acts: np.ndarray,
    thread_pool: ThreadPool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run threaded rollouts and return split state arrays.

    Returns tuple: (qpos_hist, qvel_hist, sensors_hist, last_policy_output).
    """

    del thread_pool  # Backend internally manages native threads.
    try:
        _pybind = importlib.import_module("mujoco_extensions.closed_loop_rollout.closed_loop_rollout_pybind")

        _threaded_rollout = _pybind.closed_loop_rollout.threaded_rollout
    except ImportError as e:
        raise ImportError("mujoco_extensions closed_loop_rollout backend is not built. Run: pixi run build") from e

    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    cmds = np.asarray(cmds, dtype=np.float64)
    acts = np.asarray(acts, dtype=np.float64)

    state = np.concatenate((qpos, qvel), axis=-1)
    num_threads = state.shape[0]

    physics_substeps = int(getattr(systems, "physics_substeps", 2))
    cutoff_time = float(getattr(systems, "cutoff_time", np.inf))

    out_states, out_sensors, out_acts = _threaded_rollout(
        systems,
        state,
        cmds,
        acts,
        num_threads,
        physics_substeps,
        cutoff_time,
    )

    out_states = np.asarray(out_states)
    nq = qpos.shape[1]
    out_qpos = out_states[..., :nq]
    out_qvel = out_states[..., nq:]

    return out_qpos, out_qvel, np.asarray(out_sensors), np.asarray(out_acts)


__all__ = [
    "Feature",
    "Policy",
    "ThreadPool",
    "create_systems_vector",
    "make_mujoco_system_from_model",
    "threaded_rollout",
]
