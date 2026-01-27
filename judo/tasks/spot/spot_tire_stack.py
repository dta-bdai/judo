# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjModel, MjData
from judo.utils.indexing import get_pos_indices, get_sensor_indices, get_vel_indices
from judo.utils.fields import np_1d_field
from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
)
from judo.tasks.spot.spot_base import GOAL_POSITIONS, SpotBase, SpotBaseConfig

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_two_tires.xml")

TIRE_RADIUS = 0.339
TIRE_HALF_WIDTH = 0.175
TIRE_HEIGHT = 0.67
USE_LEGS = False
USE_GRIPPER = True
USE_TORSO = True

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([-2.0, -2.0])
DEFAULT_BOTTOM_TIRE_POS = np.array([-2.0, 1.0])
DEFAULT_TOP_TIRE_POS = np.array([-2.0, 0.0])


@dataclass
class SpotTireStackConfig(SpotBaseConfig):
    """Config for the spot tire stacking task."""

    w_fence: float = 1000.0
    w_tire_stack_xy: float = 100.0  # X,Y alignment of second tire over first
    w_tire_stack_z: float = 200.0  # Z position (height) - emphasized
    w_bottom_tire_velocity: float = 50.0
    w_gripper_proximity: float = 10.0
    w_torso_proximity: float = 10.0
    w_tire_orientation: float = 20.0  # Reward for proper tire orientation
    w_gripper_height_penalty: float = 1000.0  # Strong penalty for gripper going too high
    w_controls: float = 1.0
    orientation_reward_distance_threshold: float = 2.0  # Only apply orientation reward when tires are closer than this
    stack_xy_tolerance: float = 0.1  # X,Y alignment tolerance for success
    stack_z_tolerance: float = 0.1  # Z position tolerance for success


class SpotTireStack(SpotBase[SpotTireStackConfig]):
    """Task getting Spot to stack one tire on top of another tire."""

    name: str = "spot_tire_stack"
    config_t: type[SpotTireStackConfig] = SpotTireStackConfig

    def __init__(self, model_path: str = XML_PATH, config: SpotTireStackConfig | None = None) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS, use_gripper=USE_GRIPPER, use_torso=USE_TORSO, config=config)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.bottom_tire_pose_idx = get_pos_indices(self.model, ["tire_rubber_joint"])  # flat tire
        self.top_tire_pose_idx = get_pos_indices(self.model, ["tire_rubber_2_joint"])  # standing tire

        # Velocity indices
        self.bottom_tire_vel_idx = get_vel_indices(self.model, ["tire_rubber_joint"])
        self.top_tire_vel_idx = get_vel_indices(self.model, ["tire_rubber_2_joint"])

        # Sensor indices
        self.gripper_pos_idx = get_sensor_indices(self.model, "trace_fngr_site")
        self.top_tire_y_axis_idx = get_sensor_indices(self.model, "top_tire_y_axis")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot tire stacking task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]

        #######
        ### Get positions and orientations
        #######

        # Tire positions (include initial state)
        p_W_bottom_tire = qpos[..., self.bottom_tire_pose_idx[0:3]]  # bottom tire (flat)
        p_W_top_tire = qpos[..., self.top_tire_pose_idx[0:3]]  # top tire (standing)

        # Gripper position (excludes initial state from sensors)
        p_WG = sensors[..., self.gripper_pos_idx]

        # Top tire y-axis (excludes initial state from sensors)
        top_tire_y_axis = sensors[..., self.top_tire_y_axis_idx]

        # Skip initial state from qpos to match sensor dimensions
        p_W_bottom_tire_no_init = p_W_bottom_tire[..., 1:, :]
        p_W_top_tire_no_init = p_W_top_tire[..., 1:, :]
        body_pos_no_init = body_pos[..., 1:, :]

        #######
        ### Primary stacking rewards
        #######

        # 1. X,Y alignment: top tire should be directly above bottom tire
        p_bottom_to_top_xy = p_W_top_tire_no_init[..., :2] - p_W_bottom_tire_no_init[..., :2]
        xy_distance = np.linalg.norm(p_bottom_to_top_xy, axis=-1)
        tire_stack_xy_reward = -self.config.w_tire_stack_xy * xy_distance.mean(axis=-1)

        # 2. Z position: top tire height for proper stacking
        desired_top_tire_z = p_W_bottom_tire_no_init[..., 2] + TIRE_HALF_WIDTH * 2
        z_error = np.abs(p_W_top_tire_no_init[..., 2] - desired_top_tire_z)
        tire_stack_z_reward = -self.config.w_tire_stack_z * z_error.mean(axis=-1)

        # 3. Tire orientation: top tire should align with bottom tire when close
        tire_distance = np.linalg.norm(p_W_top_tire_no_init - p_W_bottom_tire_no_init, axis=-1)
        orientation_active = tire_distance < self.config.orientation_reward_distance_threshold

        # Vector from top tire to bottom tire (desired direction)
        top_to_bottom_vec = p_W_bottom_tire_no_init - p_W_top_tire_no_init
        top_to_bottom_norm = np.maximum(np.linalg.norm(top_to_bottom_vec, axis=-1, keepdims=True), 1e-6)
        top_to_bottom_unit = top_to_bottom_vec / top_to_bottom_norm

        # Alignment measured by dot product
        dot_product = np.sum(top_tire_y_axis * top_to_bottom_unit, axis=-1)
        orientation_error = 1.0 - dot_product

        # Apply reward only when tires are close enough
        orientation_active_any = orientation_active.any(axis=-1)
        tire_orientation_reward = np.where(
            orientation_active_any, -self.config.w_tire_orientation * orientation_error.mean(axis=-1), 0.0
        )

        #######
        ### Keep bottom tire stationary
        #######

        # Bottom tire should have minimal linear and angular velocity
        bottom_tire_lin_vel = qvel[..., self.bottom_tire_vel_idx[0:3]]
        bottom_tire_ang_vel = qvel[..., self.bottom_tire_vel_idx[3:6]]

        bottom_tire_lin_speed = np.linalg.norm(bottom_tire_lin_vel, axis=-1)
        bottom_tire_ang_speed = np.linalg.norm(bottom_tire_ang_vel, axis=-1)

        bottom_tire_velocity_reward = -self.config.w_bottom_tire_velocity * (
            bottom_tire_lin_speed.mean(axis=-1) + bottom_tire_ang_speed.mean(axis=-1)
        )

        #######
        ### Shaping rewards
        #######

        # Helper function to apply quaternion to vector
        def apply_quat_to_vec(quat, vec):
            """Apply quaternion rotation to a vector."""
            # quat format: [w, x, y, z]
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

            # Vector as pure quaternion
            v_quat = np.stack([np.zeros_like(vec[..., 0]), vec[..., 0], vec[..., 1], vec[..., 2]], axis=-1)

            # Conjugate of quat
            quat_conj = np.stack([w, -x, -y, -z], axis=-1)

            # Quaternion multiplication: quat * v_quat * quat_conj
            # First: quat * v_quat
            t_w = -x * v_quat[..., 1] - y * v_quat[..., 2] - z * v_quat[..., 3]
            t_x = w * v_quat[..., 1] + y * v_quat[..., 3] - z * v_quat[..., 2]
            t_y = w * v_quat[..., 2] + z * v_quat[..., 1] - x * v_quat[..., 3]
            t_z = w * v_quat[..., 3] + x * v_quat[..., 2] - y * v_quat[..., 1]

            # Second: temp * quat_conj
            result_x = t_w * (-x) + t_x * w + t_y * (-z) - t_z * (-y)
            result_y = t_w * (-y) + t_y * w + t_z * (-x) - t_x * (-z)
            result_z = t_w * (-z) + t_z * w + t_x * (-y) - t_y * (-x)

            return np.stack([result_x, result_y, result_z], axis=-1)

        # TORSO positioning
        p_bottom_to_top = p_W_top_tire_no_init - p_W_bottom_tire_no_init
        p_bottom_to_top_norm = np.maximum(np.linalg.norm(p_bottom_to_top, axis=-1, keepdims=True), 1e-6)
        u_stacking_direction = p_bottom_to_top / p_bottom_to_top_norm

        # Dynamic rotation based on tire distance with smooth transition
        max_distance = 3.0 * TIRE_RADIUS
        min_distance = 1.0 * TIRE_RADIUS

        # Normalize distance
        normalized_distance = np.clip((tire_distance - min_distance) / (max_distance - min_distance), 0, 1)

        # Calculate dynamic angle
        angle_degrees = -90.0 * (1.0 - normalized_distance) - 25.0
        angle_radians = np.deg2rad(angle_degrees)

        # Create rotation quaternion around z-axis
        cos_half_angle = np.cos(angle_radians / 2)
        sin_half_angle = np.sin(angle_radians / 2)

        # Quaternion: [w, x, y, z] for rotation around z-axis
        rotation_quat = np.stack(
            [cos_half_angle, np.zeros_like(cos_half_angle), np.zeros_like(cos_half_angle), sin_half_angle], axis=-1
        )

        u_stacking_direction_rotated = apply_quat_to_vec(rotation_quat, u_stacking_direction)

        # Position torso away from top tire
        distance_from_tire = 0.8
        p_WB_desired = p_W_top_tire_no_init + distance_from_tire * u_stacking_direction_rotated
        p_WB_desired[..., 2] = STANDING_HEIGHT

        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(
            body_pos_no_init - p_WB_desired, axis=-1
        ).mean(axis=-1)

        # GRIPPER positioning
        # Desired point: top of the top tire, invariant to rotation around tire's Y-axis
        z_world = np.zeros_like(p_W_top_tire_no_init)
        z_world[..., 2] = 1.0
        Ty_W = top_tire_y_axis
        z_proj_perp = z_world - np.sum(z_world * Ty_W, axis=-1, keepdims=True) * Ty_W
        z_proj_perp_norm = np.maximum(np.linalg.norm(z_proj_perp, axis=-1, keepdims=True), 1e-6)
        u_top = z_proj_perp / z_proj_perp_norm
        p_W_gripper_des = p_W_top_tire_no_init + TIRE_RADIUS * u_top

        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            p_WG - p_W_gripper_des, axis=-1
        ).mean(axis=-1)

        # Gripper height penalty (prevent gripper from going too high)
        max_tire_height = 2 * TIRE_RADIUS
        gripper_height = p_WG[..., 2]
        gripper_height_excess = np.maximum(0, gripper_height - max_tire_height)
        gripper_height_penalty = -self.config.w_gripper_height_penalty * gripper_height_excess.mean(axis=-1)

        #######
        ### Penalties
        #######

        # Fence reward
        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        # Check if spot has fallen
        spot_fallen_reward = -self.config.fall_penalty * (
            body_height <= self.config.spot_fallen_threshold
        ).any(axis=-1)

        # Control penalty
        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        assert tire_stack_xy_reward.shape == (batch_size,)
        assert tire_stack_z_reward.shape == (batch_size,)
        assert tire_orientation_reward.shape == (batch_size,)
        assert bottom_tire_velocity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_height_penalty.shape == (batch_size,)
        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            tire_stack_xy_reward
            + tire_stack_z_reward
            + tire_orientation_reward
            + bottom_tire_velocity_reward
            + gripper_proximity_reward
            + torso_proximity_reward
            + gripper_height_penalty
            + spot_fence_reward
            + spot_fallen_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and two tires (hardcoded positions)."""
        # Robot pose
        robot_pose = np.array([*DEFAULT_SPOT_POS, STANDING_HEIGHT, 1.0, 0.0, 0.0, 0.0])

        # First tire pose - lying flat (90 degree roll)
        tire_1_pose = np.array([
            *DEFAULT_BOTTOM_TIRE_POS,
            TIRE_HALF_WIDTH,
            1 / np.sqrt(2),  # w
            1 / np.sqrt(2),  # x (90 degree roll)
            0,               # y
            0                # z
        ])

        # Second tire pose - upright (no rotation)
        tire_2_pose = np.array([
            *DEFAULT_TOP_TIRE_POS,
            TIRE_HEIGHT / 2,
            1.0,  # w
            0.0,  # x
            0.0,  # y
            0.0   # z
        ])

        return np.array([
            *robot_pose,
            *LEGS_STANDING_POS,
            *self.reset_arm_pos,
            *tire_1_pose,
            *tire_2_pose,
        ])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the tires are successfully stacked."""
        # Get tire positions
        bottom_tire_pos = data.qpos[self.bottom_tire_pose_idx[0:3]]
        top_tire_pos = data.qpos[self.top_tire_pose_idx[0:3]]

        # Check X,Y alignment
        xy_distance = np.linalg.norm(top_tire_pos[:2] - bottom_tire_pos[:2])
        xy_aligned = xy_distance <= self.config.stack_xy_tolerance

        # Check Z position
        desired_top_tire_z = bottom_tire_pos[2] + TIRE_HALF_WIDTH * 2
        z_error = np.abs(top_tire_pos[2] - desired_top_tire_z)
        z_correct = z_error <= self.config.stack_z_tolerance

        return bool(xy_aligned and z_correct)
