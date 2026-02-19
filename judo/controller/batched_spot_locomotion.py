# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""GPU-batched locomotion controller for Spot hierarchical control."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from judo.tasks.spot.spot_constants import (
    ISAAC_TO_MUJOCO_INDICES_12,
    LOCOMOTION_DEFAULT_JOINTS_OFFSET,
    LOCOMOTION_DEFAULT_LEGS_OFFSET,
    MUJOCO_TO_ISAAC_INDICES_19,
)


class BatchedSpotLocomotion(nn.Module):
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
    ) -> None:
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
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]

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

        q = qpos[:, 7:]  # (batch_size, 19)
        dq = qvel[:, 6:]  # (batch_size, 19)
        quat = qpos[:, 3:7]  # (batch_size, 4)

        lin_vel = qvel[:, :3]  # (batch_size, 3)
        v_base = self.quat_rotate_inverse(quat, lin_vel)

        unit_gravity_expanded = self.unit_gravity.unsqueeze(0).expand(batch_size, -1)
        gvec = self.quat_rotate_inverse(quat, unit_gravity_expanded)

        omega = qvel[:, 3:6]  # (batch_size, 3)

        # Build observation (84 dims)
        obs = torch.zeros(batch_size, 84, dtype=torch.float32, device=self.device)

        obs[:, 0:3] = v_base
        obs[:, 3:6] = omega
        obs[:, 6:9] = gvec
        obs[:, 9:12] = cmd[:, 0:3]
        obs[:, 12:19] = cmd[:, 3:10]
        obs[:, 19:31] = cmd[:, 10:22]
        obs[:, 31:34] = cmd[:, 22:25]

        q_offset_isaac = self._mujoco_to_isaac_batch(q) - self.joints_offset.unsqueeze(0)
        obs[:, 34:53] = q_offset_isaac

        dq_isaac = self._mujoco_to_isaac_batch(dq)
        obs[:, 53:72] = dq_isaac

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

        with torch.inference_mode():
            actions = self.locomotion_actor(obs)  # (batch_size, 12)

        arm_joint_command = cmd[:, 3:10]  # (batch_size, 7)
        leg_joint_command = cmd[:, 10:22]  # (batch_size, 12)

        target_leg = self._isaac_to_mujoco_batch(
            actions * self.action_scale + self.legs_offset
        )

        # Override leg commands where specified
        fl_mask = torch.any(leg_joint_command[:, 0:3] != 0, dim=1)
        target_leg[fl_mask, 0:3] = leg_joint_command[fl_mask, 0:3]

        fr_mask = torch.any(leg_joint_command[:, 3:6] != 0, dim=1)
        target_leg[fr_mask, 3:6] = leg_joint_command[fr_mask, 3:6]

        hl_mask = torch.any(leg_joint_command[:, 6:9] != 0, dim=1)
        target_leg[hl_mask, 6:9] = leg_joint_command[hl_mask, 6:9]

        hr_mask = torch.any(leg_joint_command[:, 9:12] != 0, dim=1)
        target_leg[hr_mask, 9:12] = leg_joint_command[hr_mask, 9:12]

        target_q = torch.zeros(batch_size, 19, dtype=torch.float32, device=self.device)
        target_q[:, 0:12] = target_leg
        target_q[:, 12:19] = arm_joint_command

        return actions, target_q

    @torch.no_grad()
    def compute_batch(
        self,
        cmd: torch.Tensor,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        previous_actions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get target joint positions from high-level command (batched interface).

        Args:
            cmd: High-level command (batch_size, 25) or (25,).
            qpos: Joint positions (batch_size, nq) or (nq,).
            qvel: Joint velocities (batch_size, nv) or (nv,).
            previous_actions: Previous actions (batch_size, 12) or None.

        Returns:
            Tuple of (target_q, new_previous_actions).
        """

        single_input = cmd.ndim == 1
        if single_input:
            assert qpos.ndim == 1 and qvel.ndim == 1
            cmd = cmd.unsqueeze(0)
            qpos = qpos.unsqueeze(0)
            qvel = qvel.unsqueeze(0)
            if previous_actions is not None:
                previous_actions = previous_actions.unsqueeze(0)

        batch_size = cmd.shape[0]

        if previous_actions is None:
            previous_actions = torch.zeros(batch_size, 12, device=self.device)

        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).expand(batch_size, -1)
        if qvel.ndim == 1:
            qvel = qvel.unsqueeze(0).expand(batch_size, -1)
        if previous_actions.shape[0] == 1 and batch_size > 1:
            previous_actions = previous_actions.expand(batch_size, -1)

        obs = self._get_observation(qpos, qvel, previous_actions, cmd)
        new_previous_actions, target_q = self._get_joint_target_from_obs(obs, cmd)

        if single_input:
            target_q = target_q.squeeze(0)
            new_previous_actions = new_previous_actions.squeeze(0)

        return target_q, new_previous_actions

    def reset(self) -> None:
        """Reset internal state (no-op since previous_actions is now external)."""

    def _mujoco_to_isaac_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Convert batched joint data from MuJoCo to Isaac format."""
        return x[:, self.mujoco_to_isaac_indices]

    def _isaac_to_mujoco_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Convert batched joint data from Isaac to MuJoCo format."""
        return x[:, self.isaac_to_mujoco_indices]

    @property
    def target_frequency(self) -> float:
        """Returns the target frequency of the low-level controller."""
        return 50.0  # Hz
