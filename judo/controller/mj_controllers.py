# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC
from typing import TypeVar, Any, Generic

from mujoco import MjData
import numpy as np
import numpy.typing as npt
import torch

T_ret = TypeVar("T_ret", torch.Tensor, tuple[torch.Tensor, ...])
N_ret = TypeVar("N_ret", npt.NDArray, tuple[npt.NDArray, ...])


class MjBaseController(ABC, Generic[T_ret, N_ret]):
    def __init__(self):
        pass

    def compute(
        self, data: MjData, cmd: np.ndarray, *other_inputs: np.ndarray | None
    ) -> N_ret:
        """Compute control output from MjData and command (numpy interface).

        Args:
            data: MuJoCo data object.
            cmd: Command/target as numpy array.
            *other_inputs: Variable state inputs specific to the controller.
                          May include previous_actions for recurrent controllers.

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

        This is a general interface for batched computation. Each controller
        defines what state inputs it requires.

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
        pass

    def validate_command(self, cmd: np.ndarray) -> None:
        pass

    @property
    def target_frequency(self) -> float:
        return float("inf")  # Hz - as fast as possible


class MjPassThroughController(MjBaseController[torch.Tensor, npt.NDArray]):
    """Stateless passthrough controller - returns command unchanged."""

    def __init__(self):
        super().__init__()

    def compute(self, data: MjData, cmd: np.ndarray, *_: Any) -> npt.NDArray:
        """Pass-through controller that returns input command unchanged (stateless).

        Args:
            data: MuJoCo data object with current state.
            cmd: Input command.
            *other_inputs: Ignored for stateless passthrough.

        Returns:
            cmd unchanged.
        """
        return cmd

    def compute_batch(self, cmd: torch.Tensor, *_: Any) -> torch.Tensor:
        """Pass-through controller that returns input command unchanged (stateless).

        Args:
            cmd: Input command.
            *state_inputs: Ignored for stateless passthrough.

        Returns:
            cmd unchanged.
        """
        return cmd


class PassThroughLocomotionController(
    MjBaseController[
        tuple[torch.Tensor, torch.Tensor], tuple[npt.NDArray, npt.NDArray]
    ]
):
    """Passthrough for locomotion interface - returns (cmd, None) to match stateful interface."""

    def __init__(self):
        super().__init__()

    def compute(
        self, data: MjData, cmd: np.ndarray, *_: Any
    ) -> tuple[npt.NDArray, None]:
        """Returns (cmd, None) to match locomotion controller interface.

        Args:
            data: MuJoCo data object with current state.
            cmd: Input command.
            *other_inputs: Ignored for passthrough.

        Returns:
            Tuple of (cmd, None) - cmd unchanged, no state.
        """
        return cmd, None

    def compute_batch(
        self, cmd: torch.Tensor, *_: Any
    ) -> tuple[torch.Tensor, None]:
        """Returns (cmd, None) to match locomotion controller interface.

        Args:
            cmd: Input command.
            *state_inputs: Ignored for passthrough.

        Returns:
            Tuple of (cmd, None) - cmd unchanged, no state.
        """
        return cmd, None


class MjBatchedPdController(MjBaseController[torch.Tensor, npt.NDArray]):
    """Batched PD controller that works with both numpy and torch.

    This controller implements proportional-derivative control for converting
    target joint positions to torques:
        tau = kp * (target_q - q) - kd * dq
    """

    def __init__(
        self,
        kp: torch.Tensor | np.ndarray,
        kd: torch.Tensor | np.ndarray,
        limits: torch.Tensor | np.ndarray,
    ):
        """Initialize the batched PD controller.

        This function works with both torch tensors (GPU) and numpy arrays (CPU),
        depending on the kp/kd/limits input types.

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
                          q: Current joint positions (batch_size, num_joints) or (num_joints,).
                          dq: Current joint velocities (batch_size, num_joints) or (num_joints,).
                          previous_actions: Optional, ignored by PD controller.

        Returns:
            torques (Tensor or ndarray). PD controller is stateless.
        """
        # Extract state inputs (ignore previous_actions if provided)
        q, dq = state_inputs[0], state_inputs[1]

        # Compute PD control: tau = kp * (target_q - q) - kd * dq
        tau = self.kp * (cmd - q) - self.kd * dq
        # Clamp torques to limits
        tau = self._clamp_func(tau, -self.limits, self.limits)

        return tau

    def compute(self, data: MjData, cmd: np.ndarray, *_: Any) -> npt.NDArray:
        """Compute torques using PD control (MjData interface, stateless).

        This is the main public API for CPU-based control using MjData.

        Args:
            data: MuJoCo data object containing current joint positions and velocities.
            cmd: Target joint positions (batch_size, num_joints) or (num_joints,).
            *other_inputs: Ignored, PD controller is stateless.

        Returns:
            torques (ndarray). PD controller is stateless.
        """
        # Extract current joint state (skip base pose/velocity)
        q = torch.from_numpy(data.qpos[7:])
        dq = torch.from_numpy(data.qvel[6:])

        # Call batched version
        tau = self.compute_batch(cmd, q, dq)
        return tau.numpy()
