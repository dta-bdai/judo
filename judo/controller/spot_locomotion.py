# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

"""GPU-batched locomotion controller for Spot hierarchical control."""

from abc import ABC
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from mujoco import MjData

from judo.controller.spot_constants import (
    LOCOMOTION_DEFAULT_JOINTS_OFFSET,
    LOCOMOTION_DEFAULT_LEGS_OFFSET,
    MUJOCO_TO_ISAAC_INDICES_19,
    ISAAC_TO_MUJOCO_INDICES_12,
    RL_LOCOMOTION_COMMAND_LENGTH,
)
from judo.controller.mj_controllers import MjBaseController


class BatchedSpotLocomotionController(
    nn.Module,
    MjBaseController[
        tuple[torch.Tensor, torch.Tensor], tuple[npt.NDArray, npt.NDArray]
    ],
):
    """Batched GPU-accelerated locomotion controller for Spot.

    This controller implements the learned locomotion policy that converts
    high-level commands to target joint positions.

    The controller:
    - Takes high-level commands (base velocity, arm/leg positions) from MPC
    - Runs learned locomotion policy at ~56Hz
    - Produces target joint positions for PD control
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "cuda:0",
        action_scale: float = 0.2,
    ):
        """Initialize the batched locomotion controller.

        Args:
            model_path: Path to the pre-trained locomotion policy (.pt file).
            device: The torch device to load the model and run inference on.
            action_scale: Scaling factor for the action. Defaults to 0.2.
        """
        super().__init__()
        self.device = device
        self.action_scale = action_scale

        # Load locomotion policy network
        self.locomotion_actor = nn.Sequential(
            nn.Linear(in_features=84, out_features=512),
            nn.ELU(alpha=1.0),
            nn.Linear(in_features=512, out_features=256),
            nn.ELU(alpha=1.0),
            nn.Linear(in_features=256, out_features=128),
            nn.ELU(alpha=1.0),
            nn.Linear(in_features=128, out_features=12),
        )

        # Load weights
        model_path_str = str(model_path)
        state_dict = torch.load(
            model_path_str, map_location=torch.device("cpu"), weights_only=True
        )
        self.locomotion_actor.load_state_dict(state_dict)
        self.locomotion_actor.to(self.device)
        self.locomotion_actor.eval()

        # JIT compile for kernel fusion
        example_input = torch.randn(1, 84, device=self.device)
        self.locomotion_actor = torch.jit.trace(self.locomotion_actor, example_input)

        # Locomotion policy default joint offsets (for input normalization & target offset)
        default_joints_offset = np.array(
            LOCOMOTION_DEFAULT_JOINTS_OFFSET, dtype=np.float32
        )
        default_legs_offset = np.array(LOCOMOTION_DEFAULT_LEGS_OFFSET, dtype=np.float32)

        # Register buffers
        self.register_buffer(
            "joints_offset",
            torch.tensor(
                default_joints_offset, dtype=torch.float32, device=self.device
            ),
        )
        self.register_buffer(
            "legs_offset",
            torch.tensor(default_legs_offset, dtype=torch.float32, device=self.device),
        )
        self.register_buffer(
            "unit_gravity",
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device),
        )

        # Index conversion between MuJoCo and Isaac formats
        # Isaac format reorders joints for vectorized operations
        self.register_buffer(
            "mujoco_to_isaac_indices",
            torch.tensor(
                MUJOCO_TO_ISAAC_INDICES_19, dtype=torch.long, device=self.device
            ),
        )
        self.register_buffer(
            "isaac_to_mujoco_indices",
            torch.tensor(
                ISAAC_TO_MUJOCO_INDICES_12, dtype=torch.long, device=self.device
            ),
        )

    @staticmethod
    def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse of quaternion (all on GPU).

        Args:
            quat: Quaternion in scalar-first format (batch_size, 4) [w, x, y, z]
            vec: Vector to rotate (batch_size, 3)

        Returns:
            Rotated vector (batch_size, 3)
        """
        # Extract quaternion components (scalar-first format: w, x, y, z)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Compute rotation using quaternion formula for inverse rotation
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]

        # Precompute repeated terms
        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        # Rotation matrix elements for inverse rotation (transpose of forward rotation)
        result = torch.stack(
            [
                (ww + xx - yy - zz) * vx + 2 * (xy + wz) * vy + 2 * (xz - wy) * vz,
                2 * (xy - wz) * vx + (ww - xx + yy - zz) * vy + 2 * (yz + wx) * vz,
                2 * (xz + wy) * vx + 2 * (yz - wx) * vy + (ww - xx - yy + zz) * vz,
            ],
            dim=1,
        )

        return result

    @torch.no_grad()
    def _get_observation(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        previous_actions: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        """Extract batched observation from state (internal method).

        Args:
            qpos: Joint positions (batch_size, nq). First 7 are base pose (pos + quat).
            qvel: Joint velocities (batch_size, nv). First 6 are base velocity (linear + angular).
            previous_actions: Previous actions vector (batch_size, 12).
            cmd: Command vector for the base and arm (batch_size, 25).

        Returns:
            Observation tensor (batch_size, 84).
        """
        batch_size = qpos.shape[0]

        # Extract joint positions and velocities
        q = qpos[:, 7:]  # (batch_size, 19)
        dq = qvel[:, 6:]  # (batch_size, 19)

        # Extract quaternion (scalar-first format: w, x, y, z)
        quat = qpos[:, 3:7]  # (batch_size, 4)

        # Apply rotation to linear velocity (to base frame) - fully on GPU
        lin_vel = qvel[:, :3]  # (batch_size, 3)
        v_base = self.quat_rotate_inverse(quat, lin_vel)

        # Apply rotation to gravity vector - fully on GPU
        unit_gravity_expanded = self.unit_gravity.unsqueeze(0).expand(
            batch_size, -1
        )  # (batch_size, 3)
        gvec = self.quat_rotate_inverse(quat, unit_gravity_expanded)

        # Angular velocity is already in the correct frame
        omega = qvel[:, 3:6]  # (batch_size, 3)

        # Build observation (84 dims)
        obs = torch.zeros(batch_size, 84, dtype=torch.float32, device=self.device)

        # Base linear velocity
        obs[:, 0:3] = v_base
        # Base angular velocity
        obs[:, 3:6] = omega
        # Projected gravity
        obs[:, 6:9] = gvec
        # Base velocity command
        obs[:, 9:12] = cmd[:, 0:3]
        # Arm joint command
        obs[:, 12:19] = cmd[:, 3:10]
        # Leg joint command
        obs[:, 19:31] = cmd[:, 10:22]
        # Base roll pitch height
        obs[:, 31:34] = cmd[:, 22:25]

        # Joint position (converted to Isaac format)
        q_offset_isaac = self._mujoco_to_isaac_batch(q) - self.joints_offset.unsqueeze(
            0
        )
        obs[:, 34:53] = q_offset_isaac

        # Joint velocity (converted to Isaac format)
        dq_isaac = self._mujoco_to_isaac_batch(dq)
        obs[:, 53:72] = dq_isaac

        # Previous actions
        obs[:, 72:84] = previous_actions

        return obs

    @torch.no_grad()
    def _get_joint_target_from_obs(
        self,
        obs: torch.Tensor,
        cmd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer actions and return target joint positions (internal method).

        Args:
            obs: Observation input for the policy (batch_size, 84).
            cmd: Command vector for the base and arm (batch_size, 25).

        Returns:
            Tuple of (actions, target_q):
                - actions: Raw Isaac policy output (batch_size, 12) - Isaac order
                - target_q: Target joint positions (batch_size, 19) - Mujoco order
        """
        batch_size = obs.shape[0]

        # Run policy
        with torch.inference_mode():
            actions = self.locomotion_actor(obs)  # (batch_size, 12)

        # Extract commands
        arm_joint_command = cmd[:, 3:10]  # (batch_size, 7)
        leg_joint_command = cmd[:, 10:22]  # (batch_size, 12)

        # Convert actions to target leg positions
        target_leg = self._isaac_to_mujoco_batch(
            actions * self.action_scale + self.legs_offset
        )

        # Override leg commands where specified
        # FL (joints 0:3)
        fl_mask = torch.any(leg_joint_command[:, 0:3] != 0, dim=1)  # (batch_size,)
        target_leg[fl_mask, 0:3] = leg_joint_command[fl_mask, 0:3]

        # FR (joints 3:6)
        fr_mask = torch.any(leg_joint_command[:, 3:6] != 0, dim=1)
        target_leg[fr_mask, 3:6] = leg_joint_command[fr_mask, 3:6]

        # HL (joints 6:9)
        hl_mask = torch.any(leg_joint_command[:, 6:9] != 0, dim=1)
        target_leg[hl_mask, 6:9] = leg_joint_command[hl_mask, 6:9]

        # HR (joints 9:12)
        hr_mask = torch.any(leg_joint_command[:, 9:12] != 0, dim=1)
        target_leg[hr_mask, 9:12] = leg_joint_command[hr_mask, 9:12]

        # Assemble full target
        target_q = torch.zeros(batch_size, 19, dtype=torch.float32, device=self.device)
        target_q[:, 0:12] = target_leg
        target_q[:, 12:19] = arm_joint_command

        return actions, target_q

    def compute(
        self, data: MjData, cmd: np.ndarray, *other_inputs: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get target joint positions from high-level command (stateful controller).

        This is the main public API for CPU-based control using MjData.
        Converts numpy inputs to torch, calls compute_batch, converts back to numpy.

        Args:
            data: MuJoCo data object with current state.
            cmd: High-level command (25,) [base_vel(3), arm(7), legs(12), torso(3)].
            *other_inputs: Optional previous_actions (12,). If not provided, initializes to zeros.

        Returns:
            Tuple of (target_q, new_previous_actions):
            - target_q: Target joint positions (19,) as numpy array.
            - new_previous_actions: New previous actions (12,) to use in next call.
        """
        # Extract previous_actions from other_inputs
        assert len(other_inputs) == 1, "Expected previous_actions as the only other input."
        previous_actions = other_inputs[0]

        # Convert numpy to torch
        qpos_torch = torch.from_numpy(data.qpos.copy()).float().to(self.device)
        qvel_torch = torch.from_numpy(data.qvel.copy()).float().to(self.device)
        cmd_torch = torch.from_numpy(cmd).float().to(self.device)

        previous_actions_torch = None
        if previous_actions is not None:
            previous_actions_torch = (
                torch.from_numpy(previous_actions).float().to(self.device)
            )

        # Call batched version with new signature
        target_q_torch, new_previous_actions_torch = self.compute_batch(
            cmd_torch, qpos_torch, qvel_torch, previous_actions_torch
        )

        # Convert back to numpy
        return target_q_torch.cpu().numpy(), new_previous_actions_torch.cpu().numpy()

    @torch.no_grad()
    def compute_batch(
        self,
        cmd: torch.Tensor,
        *state_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor]:
        """Get target joint positions from high-level command (batched interface).

        This is the batched API for GPU-accelerated control. All inputs and outputs
        are torch tensors on the device.

        Args:
            cmd: High-level command (batch_size, 25) or (25,) [base_vel(3), arm(7), legs(12), torso(3)].
            *state_inputs: State inputs (qpos, qvel, previous_actions).
                          qpos: Joint positions (batch_size, nq) or (nq,). First 7 are base pose (pos + quat).
                          qvel: Joint velocities (batch_size, nv) or (nv,). First 6 are base velocity (linear + angular).
                          previous_actions: Previous policy actions (batch_size, 12) or (12,). If None, initializes to zeros.

        Returns:
            Tuple of (target_q, new_previous_actions):
            - target_q: Target joint positions (batch_size, 19) or (19,) as torch tensor on device.
            - new_previous_actions: New previous actions (batch_size, 12) to use in next call.
        """
        # Extract state inputs
        assert (
            len(state_inputs) == 3
        ), "Expected 3 state inputs: qpos, qvel, previous_actions."
        qpos, qvel, previous_actions = state_inputs[0], state_inputs[1], state_inputs[2]

        # Handle both single and batched inputs
        single_input = cmd.ndim == 1
        if single_input:
            assert (
                qpos.ndim == 1 and qvel.ndim == 1
            ), "If cmd is single input, qpos and qvel must also be single inputs."
            # Add batch dimension
            cmd = cmd.unsqueeze(0)
            qpos = qpos.unsqueeze(0)
            qvel = qvel.unsqueeze(0)
            if previous_actions is not None:
                previous_actions = previous_actions.unsqueeze(0)

        batch_size = cmd.shape[0]

        # Initialize previous actions if not provided
        if previous_actions is None:
            previous_actions = torch.zeros(batch_size, 12, device=self.device)

        # Handle batch dimensions - tile state if needed to match cmd batch size
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).expand(batch_size, -1)
        if qvel.ndim == 1:
            qvel = qvel.unsqueeze(0).expand(batch_size, -1)
        if previous_actions.shape[0] == 1 and batch_size > 1:
            previous_actions = previous_actions.expand(batch_size, -1)

        # Get observation
        obs = self._get_observation(qpos, qvel, previous_actions, cmd)

        # Get target joint positions and new previous actions
        new_previous_actions, target_q = self._get_joint_target_from_obs(obs, cmd)

        # Remove batch dimension if input was single
        if single_input:
            target_q = target_q.squeeze(0)
            new_previous_actions = new_previous_actions.squeeze(0)

        return target_q, new_previous_actions

    def reset(self):
        """Reset internal state (no-op since previous_actions is now external)."""
        pass

    def _mujoco_to_isaac_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Convert batched joint data from MuJoCo to Isaac format.

        Args:
            x: Input tensor (batch_size, 19)

        Returns:
            Converted tensor (batch_size, 19)
        """
        return x[:, self.mujoco_to_isaac_indices]

    def _isaac_to_mujoco_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Convert batched joint data from Isaac to MuJoCo format.

        Args:
            x: Input tensor (batch_size, 12)

        Returns:
            Converted tensor (batch_size, 12)
        """
        return x[:, self.isaac_to_mujoco_indices]

    def validate_command(self, cmd: np.ndarray) -> None:
        """Validate that command has the correct format.

        Args:
            cmd: Command array to validate.

        Raises:
            ValueError: If command format is invalid.
        """
        if cmd.ndim not in (1, 2):
            raise ValueError(f"Command must be 1D or 2D array, got shape {cmd.shape}")

        cmd_dim = cmd.shape[-1]
        if cmd_dim != RL_LOCOMOTION_COMMAND_LENGTH:
            raise ValueError(
                f"Command must have {RL_LOCOMOTION_COMMAND_LENGTH} dimensions "
                f"[base_vel(3), arm(7), legs(12), torso(3)], got {cmd_dim}"
            )

    @property
    def target_frequency(self) -> float:
        """Returns the target frequency of the low-level controller."""
        return 50.0  # Hz
