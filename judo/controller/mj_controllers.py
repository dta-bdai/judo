# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Base controller interfaces for MuJoCo-based control.

Provides abstract base classes and concrete implementations for controllers
that can operate in both single-sample (MjData) and batched (torch) modes.
"""

from abc import ABC
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from mujoco import MjData

T_ret = TypeVar("T_ret", torch.Tensor, tuple[torch.Tensor, ...])
N_ret = TypeVar("N_ret", npt.NDArray, tuple[npt.NDArray, ...])


class MjBaseController(ABC, Generic[T_ret, N_ret]):
    """Abstract base class for MuJoCo controllers with numpy and torch interfaces."""

    def __init__(self) -> None:
        """Initialize the base controller."""

    def compute(
        self, data: MjData, cmd: np.ndarray, *other_inputs: np.ndarray | None
    ) -> N_ret:
        """Compute control output from MjData and command (numpy interface).

        Args:
            data: MuJoCo data object.
            cmd: Command/target as numpy array.
            *other_inputs: Variable state inputs specific to the controller.

        Returns:
            For stateless controllers: control_output (np.ndarray).
            For stateful controllers: tuple of (control_output, *state_outputs).
        """
        raise NotImplementedError

    def compute_batch(
        self,
        cmd: torch.Tensor,
        *state_inputs: torch.Tensor | None,
    ) -> T_ret:
        """Compute control output from command and state inputs (batched interface).

        Args:
            cmd: Command/target. For PD controller, this is target_q.
                 For locomotion controller, this is high-level command.
            *state_inputs: Variable state inputs specific to the controller.
                          - PD controller expects: q, dq
                          - Spot locomotion expects: qpos, qvel, previous_actions

        Returns:
            For stateless controllers: control_output (Tensor or ndarray).
            For stateful controllers: tuple of (control_output, *state_outputs).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset controller state."""

    def validate_command(self, cmd: np.ndarray) -> None:
        """Validate command format."""

    @property
    def target_frequency(self) -> float:
        """Target control frequency in Hz."""
        return float("inf")  # Hz - as fast as possible


class MjPassThroughController(MjBaseController[torch.Tensor, npt.NDArray]):
    """Stateless passthrough controller - returns command unchanged."""

    def __init__(self) -> None:
        """Initialize the passthrough controller."""
        super().__init__()

    def compute(self, data: MjData, cmd: np.ndarray, *_: Any) -> npt.NDArray:
        """Pass-through controller that returns input command unchanged (stateless)."""
        return cmd

    def compute_batch(self, cmd: torch.Tensor, *_: Any) -> torch.Tensor:
        """Pass-through controller that returns input command unchanged (stateless)."""
        return cmd


class PassThroughLocomotionController(
    MjBaseController[
        tuple[torch.Tensor, torch.Tensor], tuple[npt.NDArray, npt.NDArray]
    ]
):
    """Passthrough for locomotion interface - returns (cmd, None) to match stateful interface."""

    def __init__(self) -> None:
        """Initialize the passthrough locomotion controller."""
        super().__init__()

    def compute(
        self, data: MjData, cmd: np.ndarray, *_: Any
    ) -> tuple[npt.NDArray, None]:
        """Returns (cmd, None) to match locomotion controller interface."""
        return cmd, None

    def compute_batch(
        self, cmd: torch.Tensor, *_: Any
    ) -> tuple[torch.Tensor, None]:
        """Returns (cmd, None) to match locomotion controller interface."""
        return cmd, None


class MjBatchedPdController(MjBaseController[torch.Tensor, npt.NDArray]):
    """Batched PD controller that works with both numpy and torch.

    Implements proportional-derivative control:
        tau = kp * (target_q - q) - kd * dq
    """

    def __init__(
        self,
        kp: torch.Tensor | np.ndarray,
        kd: torch.Tensor | np.ndarray,
        limits: torch.Tensor | np.ndarray,
    ) -> None:
        """Initialize the batched PD controller.

        Args:
            kp: Proportional gains (num_joints,).
            kd: Derivative gains (num_joints,).
            limits: Torque limits for clipping (num_joints,).
        """
        super().__init__()

        self.kp = kp if torch.is_tensor(kp) else torch.from_numpy(kp)
        self.kd = kd if torch.is_tensor(kd) else torch.from_numpy(kd)
        self.limits = limits if torch.is_tensor(limits) else torch.from_numpy(limits)
        self._clamp_func = torch.clamp if torch.is_tensor(kp) else np.clip

    def compute_batch(
        self,
        cmd: torch.Tensor,
        *state_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute torques using PD control (batched interface, stateless).

        Args:
            cmd: Target joint positions (batch_size, num_joints) or (num_joints,).
            *state_inputs: Current state (q, dq[, previous_actions]).

        Returns:
            Torques tensor.
        """
        q, dq = state_inputs[0], state_inputs[1]
        tau = self.kp * (cmd - q) - self.kd * dq
        tau = self._clamp_func(tau, -self.limits, self.limits)
        return tau

    def compute(self, data: MjData, cmd: np.ndarray, *_: Any) -> npt.NDArray:
        """Compute torques using PD control (MjData interface, stateless).

        Args:
            data: MuJoCo data object containing current joint positions and velocities.
            cmd: Target joint positions (batch_size, num_joints) or (num_joints,).

        Returns:
            Torques as numpy array.
        """
        q = torch.from_numpy(data.qpos[7:])
        dq = torch.from_numpy(data.qvel[6:])
        tau = self.compute_batch(cmd, q, dq)
        return tau.numpy()
