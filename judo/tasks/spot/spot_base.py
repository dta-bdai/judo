from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic

import mujoco
import numpy as np
from mujoco import MjData, MjModel
from judo.tasks.spot.spot_constants import (
    ARM_CMD_INDS,
    ARM_SOFT_LOWER_JOINT_LIMITS,
    ARM_SOFT_UPPER_JOINT_LIMITS,
    ARM_STOWED_POS,
    ARM_UNSTOWED_POS,
    BASE_SOFT_LIMITS,
    BASE_VEL_CMD_INDS,
    DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
    FRONT_LEG_CMD_INDS,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    JOINT_NAMES_BOSDYN,
    LEG_SOFT_LOWER_JOINT_LIMITS,
    LEG_SOFT_UPPER_JOINT_LIMITS,
    LEGS_STANDING_POS,
    STANDING_HEIGHT_CMD,
    STANDING_HEIGHT,
    TORSO_CMD_INDS,
    TORSO_LOWER,
    TORSO_UPPER,
)

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.utils.mujoco import RolloutBackend, SimBackendSpot
XML_PATH = str(MODEL_PATH / "xml/spot_components/robot.xml")


@dataclass
class GOAL_POSITIONS:
    """Goal positions of Spot."""

    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.0]))
    origin_decimal: np.ndarray = field(default_factory=lambda: np.array([0, 0.0, 0.01]))
    blue_cross: np.ndarray = field(default_factory=lambda: np.array([2.77, 0.71, 0.3]))
    black_cross: np.ndarray = field(default_factory=lambda: np.array([1.5, -1.5, 0.275]))


@dataclass
class SpotBaseConfig(TaskConfig):
    """Base config for spot tasks."""

    fall_penalty: float = 2500.0
    spot_fallen_threshold = 0.35  # Torso height where Spot is considered "fallen"
    w_goal: float = 60.0
    w_controls: float = 0.0


ConfigT = TypeVar("ConfigT", bound=SpotBaseConfig)


class SpotBase(Task[ConfigT], Generic[ConfigT]):
    """Flexible base task for Spot locomotion/skills.

    Controls are a compact vector mapped to the 25-dim policy command:
    - Base only:                        [base_vel(3)]
    - Base + Arm:                       [base_vel(3), arm_cmd(7)]
    - Base + Legs:                      [base_vel(3), front_leg_cmd(6), leg_selection(1)]
    - Base + Arm + Legs:                [base_vel(3), arm_cmd(7), front_leg_cmd(6), leg_selection(1)]
    - With Torso:                       [..., torso_cmd(3)]  (appended to any of the above)

    The mapping to the 25-dim policy command is done in task_to_sim_ctrl.
    """

    name: str = "spot_base"
    config_t: type[SpotBaseConfig] = SpotBaseConfig
    default_backend = "mujoco_spot"  # Use Spot-specific backend

    def __init__(
        self,
        model_path: str = XML_PATH,
        use_arm: bool = True,
        use_gripper: bool = False,
        use_legs: bool = False,
        use_torso: bool = False,
        config: SpotBaseConfig | None = None,
    ) -> None:
        super().__init__(model_path=model_path, config=config)
        self.use_arm = use_arm
        self.use_gripper = use_gripper
        self.use_legs = use_legs
        self.use_torso = use_torso

        # Selection indices
        self.leg_selection_index = None
        self.gripper_selection_index = None

        self.set_command_values()

        # Use ONNX-based rollout backend
        self.RolloutBackend = RolloutBackend
        self.SimBackend = SimBackendSpot

        self.default_policy_command = np.array(
            [0, 0, 0] + list(ARM_STOWED_POS) + [0] * 12 + [0, 0, STANDING_HEIGHT_CMD]
        )

    @property
    def nu(self) -> int:
        """Number of controls for this task."""
        return len(self.default_command)

    @property
    def ctrlrange(self) -> np.ndarray:
        """Control bounds for the task."""
        BASE_LOWER = -BASE_SOFT_LIMITS
        BASE_UPPER = BASE_SOFT_LIMITS
        GRIPPER_LOWER = GRIPPER_OPEN_POS if self.use_gripper else GRIPPER_CLOSED_POS
        GRIPPER_UPPER = GRIPPER_CLOSED_POS
        ARM_LOWER = np.concatenate((ARM_SOFT_LOWER_JOINT_LIMITS[:-1], [GRIPPER_LOWER]))
        ARM_UPPER = np.concatenate((ARM_SOFT_UPPER_JOINT_LIMITS[:-1], [GRIPPER_UPPER]))
        LEGS_LOWER = LEG_SOFT_LOWER_JOINT_LIMITS[0:6]
        LEGS_UPPER = LEG_SOFT_UPPER_JOINT_LIMITS[0:6]
        LEG_SELECTION_LOWER = -np.ones(1)
        LEG_SELECTION_UPPER = np.ones(1)
        GRIPPER_SELECTION_LOWER = -np.ones(1)
        GRIPPER_SELECTION_UPPER = np.ones(1)
        TORSO_BOUNDS_LOWER = TORSO_LOWER
        TORSO_BOUNDS_UPPER = TORSO_UPPER

        if not self.use_arm and not self.use_legs:  # Base
            lower_components = [BASE_LOWER]
            upper_components = [BASE_UPPER]
        elif self.use_arm and not self.use_legs:  # Base and arm
            lower_components = [BASE_LOWER, ARM_LOWER]
            upper_components = [BASE_UPPER, ARM_UPPER]
            if self.use_gripper:
                lower_components.append(GRIPPER_SELECTION_LOWER)
                upper_components.append(GRIPPER_SELECTION_UPPER)
        elif not self.use_arm and self.use_legs:  # Base and legs
            lower_components = [BASE_LOWER, LEGS_LOWER, LEG_SELECTION_LOWER]
            upper_components = [BASE_UPPER, LEGS_UPPER, LEG_SELECTION_UPPER]
        elif self.use_arm and self.use_legs:  # Base, arm, and legs
            lower_components = [BASE_LOWER, ARM_LOWER]
            upper_components = [BASE_UPPER, ARM_UPPER]
            if self.use_gripper:
                lower_components.append(GRIPPER_SELECTION_LOWER)
                upper_components.append(GRIPPER_SELECTION_UPPER)
            lower_components.extend([LEGS_LOWER, LEG_SELECTION_LOWER])
            upper_components.extend([LEGS_UPPER, LEG_SELECTION_UPPER])
        else:
            raise ValueError("Invalid combination of use_arm and use_legs")

        # Add torso if enabled
        if self.use_torso:
            lower_components.append(TORSO_BOUNDS_LOWER)
            upper_components.append(TORSO_BOUNDS_UPPER)

        lower_bound = np.concatenate(lower_components)
        upper_bound = np.concatenate(upper_components)

        return np.stack([lower_bound, upper_bound], axis=-1)

    def set_command_values(self) -> None:
        """Update default_command and command_mask."""
        # Reset selection indices
        self.leg_selection_index = None
        self.gripper_selection_index = None

        if not self.use_arm and not self.use_legs:  # Base
            # Base velocity
            command_values = [0, 0, 0]
            command_mask = BASE_VEL_CMD_INDS
        elif self.use_arm and not self.use_legs:  # Base and arm
            # Base velocity, arm joint angles
            command_values = [0, 0, 0, *ARM_UNSTOWED_POS]
            command_mask = BASE_VEL_CMD_INDS + ARM_CMD_INDS
            if self.use_gripper:
                command_values.append(0.0)  # gripper selection
                self.gripper_selection_index = len(command_values) - 1
        elif not self.use_arm and self.use_legs:  # Base and legs
            # Base velocity, leg joint angles, leg selection
            command_values = [0, 0, 0, *LEGS_STANDING_POS[0:6], 0]
            command_mask = BASE_VEL_CMD_INDS + FRONT_LEG_CMD_INDS
            self.leg_selection_index = len(command_values) - 1
        elif self.use_arm and self.use_legs:  # Base, arm, and legs
            # Base velocity, arm joint angles, leg joint angles, leg selection
            command_values = [0, 0, 0, *ARM_UNSTOWED_POS]
            command_mask = BASE_VEL_CMD_INDS + ARM_CMD_INDS + FRONT_LEG_CMD_INDS
            if self.use_gripper:
                command_values.append(0.0)  # gripper selection
                self.gripper_selection_index = len(command_values) - 1
            command_values.extend([*LEGS_STANDING_POS[0:6], 0])
            self.leg_selection_index = len(command_values) - 1

        # Add torso if enabled
        if self.use_torso:
            command_values.extend([0, 0, STANDING_HEIGHT])  # roll, pitch, height
            command_mask = command_mask + TORSO_CMD_INDS

        self.default_command = np.array(command_values)
        self.command_mask = np.array(command_mask)

    def apply_selection_mask(self, controls: np.ndarray) -> np.ndarray:
        """Activate or deactivate leg and gripper commands depending on the selections.

        leg selection:
        -1.0 to -0.5: manipulation with left leg
        -0.5 to +0.5: no leg manipulation
        +0.5 to +1.0: manipulation with right leg

        gripper selection:
        -1.0 to 0.0: gripper closed
        0.0 to +1.0: gripper open

        Returns:
            controls_full: controls after applying selection mask and removing selection indexes.
                The provided 'controls' is returned unchanged if no selection indexes are used.
        """
        if not self.leg_selection_index and not self.gripper_selection_index:
            return controls

        added_dim = False
        if controls.ndim == 1:  # Expand 1D control vector
            controls = np.expand_dims(controls, axis=0)
            added_dim = True

        controls = controls.copy()

        # Collect selection indices to remove
        selection_indices = []
        if self.gripper_selection_index is not None:
            selection_indices.append(self.gripper_selection_index)
        if self.leg_selection_index is not None:
            selection_indices.append(self.leg_selection_index)

        # Apply gripper mask
        if self.use_arm and self.use_gripper and self.gripper_selection_index is not None:
            selection = controls[..., self.gripper_selection_index]
            mask_gripper = selection < 0.0
            # Gripper is at index 9 in ARM_CMD (last element of 7 arm joints)
            # In the controls array, it's at position 3 (base) + 6 (arm joints) = 9
            controls[mask_gripper, 9] = GRIPPER_CLOSED_POS

        # Apply leg mask
        if self.use_legs and self.leg_selection_index is not None:
            selection = controls[..., self.leg_selection_index]
            mask_fl = selection < -0.5
            mask_fr = selection > 0.5
            mask_neither = ~(mask_fl | mask_fr)

            # Determine where leg commands start
            leg_start_idx = 3  # after base
            if self.use_arm:
                leg_start_idx += 7  # arm commands
                if self.use_gripper:
                    leg_start_idx += 1  # gripper selection (will be removed, but affects indexing now)

            # Last 6 entries before leg selection are leg commands (FL: 3, FR: 3)
            controls[mask_fl, leg_start_idx + 3 : leg_start_idx + 6] = 0.0  # Zero FR
            controls[mask_fr, leg_start_idx : leg_start_idx + 3] = 0.0  # Zero FL
            controls[mask_neither, leg_start_idx : leg_start_idx + 6] = 0.0  # Zero both

        # Remove selection indices
        controls_full = np.delete(controls, selection_indices, axis=-1)

        if added_dim:  # Squeeze 1D control vector
            controls_full = controls_full.squeeze(axis=0)

        return controls_full

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Map compact controls (..., nu) to 25-dim policy command expected by C++ rollout.

        Layout of 25-dim policy command:
        [0:3]  torso_vel_cmd
        [3:10] arm_cmd
        [10:22] leg_cmd (4 legs x 3) used for override
        [22:25] torso_pos_cmd
        """
        controls = np.asarray(controls)
        added_dim = False
        if controls.ndim == 1:
            controls = controls[None]
            added_dim = True

        # Apply selection mask first - this handles gripper and leg selection logic
        # and removes the selection indices from the control vector
        controls = self.apply_selection_mask(controls)

        T = controls.shape[1] if controls.ndim == 3 else 1
        if controls.ndim == 2:
            # assume (..., nu) at sim timestep grid
            controls = controls[:, None, :]
            T = 1

        out = np.zeros((controls.shape[0], controls.shape[1], 25), dtype=controls.dtype)

        # After apply_selection_mask, selection indices are removed
        base_end = 3
        arm_end = base_end + (7 if self.use_arm else 0)
        legs_end = arm_end + (6 if self.use_legs else 0)  # No +1 for selection, already removed
        torso_end = legs_end + (3 if self.use_torso else 0)

        # Base velocity
        out[..., 0:3] = controls[..., 0:base_end]

        # Default torso height (may be overridden below if use_torso)
        out[..., 24] = STANDING_HEIGHT

        # Arm commands
        if self.use_arm:
            out[..., 3:10] = controls[..., base_end:arm_end]

        # Leg override commands (front legs only)
        # Note: leg masking is already applied in apply_selection_mask
        if self.use_legs:
            leg_block = controls[..., arm_end:legs_end]  # shape (..., 6) - no selection index
            fl_cmd = leg_block[..., 0:3]
            fr_cmd = leg_block[..., 3:6]

            # place into policy command leg slots [10:22]
            # groups: FL(10:13), FR(13:16), HL(16:19), HR(19:22)
            out[..., 10:13] = fl_cmd
            out[..., 13:16] = fr_cmd

        # Torso commands (roll, pitch, height)
        if self.use_torso:
            out[..., 22:25] = controls[..., legs_end:torso_end]

        if added_dim:
            out = out.squeeze(axis=0)
        if T == 1 and out.ndim == 3:
            out = out[:, 0, :]
        # out[..., :] = self.default_policy_command
        return out

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Simple standing reward used as a default for base class.

        Penalizes falling and large control magnitudes.
        """
        rewards = np.zeros(states.shape[0])
        return rewards

    @property
    def reset_arm_pos(self) -> np.ndarray:
        """Reset position of the arm"""
        return ARM_UNSTOWED_POS if self.use_arm else ARM_STOWED_POS

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""

        return np.array(
            [
                *np.random.randn(2),
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
            ]
        )

    def reset(self) -> None:
        self.data.qpos = self.reset_pose
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)


