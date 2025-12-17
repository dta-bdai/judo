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

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_barrier.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
Y_AXIS = np.array([1.0, 0.0, 0.0])
X_AXIS = np.array([0.0, -1.0, 0.0])  # flipped for hardware

# annulus object position sampling
RADIUS_MIN = 1.5
RADIUS_MAX = 3.0
USE_LEGS = False
USE_GRIPPER = True
USE_TORSO = True

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
DEFAULT_OBJECT_POS = np.array([2.0, 0.0])


@dataclass
class SpotBarrierDragConfig(SpotBaseConfig):
    """Config for the spot barrier dragging task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([-0.0, 0, 0.0], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, -1.0],
        maxs=[5.0, 5.0, 2.0],
        steps=[0.1, 0.1, 0.1],
        vis_name="barrier_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[-1.0, 1.3, 0.0],
    )
    w_fence: float = 1000.0
    w_goal: float = 500.0
    w_orientation: float = 50.0  # Reward for keeping barrier aligned with global x, y, z axes
    w_heading_alignment: float = 500.0  # Reward for barrier facing towards goal
    object_fallen_threshold: float = 0.5
    orientation_threshold: float = 0.7
    w_torso_proximity: float = 50.0
    torso_proximity_threshold: float = 1.0
    w_object_fallen: float = 2500.0
    w_gripper_to_grasp_proximity: float = 250.0
    w_gripper_orientation: float = 100.0
    w_approach_site_proximity: float = 100.0
    w_controls: float = 2.0
    w_object_velocity: float = 10.0
    # Resistance-based grasp detection parameters
    resistance_threshold: float = 0.15
    fully_closed_tolerance: float = 0.05
    w_false_grasp_penalty: float = 1000.0
    w_grasp_quality: float = 100.0
    position_tolerance: float = 0.2


class SpotBarrierDrag(SpotBase[SpotBarrierDragConfig]):
    """Task getting Spot to drag an upright crowd barrier to a goal location."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS, use_gripper=USE_GRIPPER, use_torso=USE_TORSO)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["crowd_barrier_joint"])
        self.object_vel_idx = get_vel_indices(self.model, ["crowd_barrier_joint"])
        self.object_x_axis_idx = get_sensor_indices(self.model, "object_x_axis")
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")
        self.object_z_axis_idx = get_sensor_indices(self.model, "object_z_axis")

        # Grasp point proximity sensors (left and right mid-height points for drag task)
        self.gripper_to_grasp_left_idx = get_sensor_indices(self.model, "sensor_gripper_to_grasp_drag_left")
        self.gripper_to_grasp_right_idx = get_sensor_indices(self.model, "sensor_gripper_to_grasp_drag_right")

        # Gripper orientation sensors
        self.gripper_x_axis_idx = get_sensor_indices(self.model, "sensor_gripper_x_axis")
        self.gripper_y_axis_idx = get_sensor_indices(self.model, "sensor_gripper_y_axis")
        self.gripper_z_axis_idx = get_sensor_indices(self.model, "sensor_gripper_z_axis")

        # Gripper position sensor
        self.gripper_pos_idx = get_sensor_indices(self.model, "trace_fngr_site")

        # Torso to approach site proximity sensors (left and right)
        self.torso_to_approach_left_idx = get_sensor_indices(self.model, "sensor_torso_to_approach_left")
        self.torso_to_approach_right_idx = get_sensor_indices(self.model, "sensor_torso_to_approach_right")

        # Get gripper joint position index for closure reward
        self.gripper_joint_idx = get_pos_indices(self.model, "arm_f1x")[0]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotBarrierDragConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot crowd barrier dragging task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]
        object_linear_velocity = qvel[..., self.object_vel_idx[0:3]]

        object_z_axis = sensors[..., self.object_z_axis_idx]
        object_x_axis = sensors[..., self.object_x_axis_idx]
        object_y_axis = sensors[..., self.object_y_axis_idx]

        # Grasp point proximity vectors
        gripper_to_grasp_left = sensors[..., self.gripper_to_grasp_left_idx]
        gripper_to_grasp_right = sensors[..., self.gripper_to_grasp_right_idx]

        # Gripper orientation axes
        gripper_x_axis = sensors[..., self.gripper_x_axis_idx]
        gripper_y_axis = sensors[..., self.gripper_y_axis_idx]

        # Gripper position
        gripper_pos = sensors[..., self.gripper_pos_idx]

        # Torso to approach site vectors
        torso_to_approach_left = sensors[..., self.torso_to_approach_left_idx]
        torso_to_approach_right = sensors[..., self.torso_to_approach_right_idx]

        # === Resistance-Based Grasp Detection ===
        gripper_joint_pos = qpos[..., 1:, self.gripper_joint_idx]  # Skip initial state
        gripper_joint_cmd = controls[..., 9]  # ARM_CMD_INDS[6] maps to controls index 9

        # Position error: actual - commanded
        position_error = gripper_joint_pos - gripper_joint_cmd

        # Grasp detection logic
        has_resistance = position_error > config.resistance_threshold
        is_fully_closed = np.abs(gripper_joint_pos - GRIPPER_CLOSED_POS) < config.fully_closed_tolerance
        is_not_empty = ~is_fully_closed
        is_grasping = has_resistance & is_not_empty
        is_false_grasp = is_fully_closed & (gripper_joint_cmd < -0.5)

        # Fence reward
        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        # Check if spot has fallen
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)

        # Check if object has fallen
        object_fallen_reward = -config.w_object_fallen * (
            object_pos[..., 2] <= config.object_fallen_threshold
        ).any(axis=-1)

        # Goal reward - distance to goal position
        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        # Orientation reward - keep barrier aligned with global x, y, z axes
        x_alignment = np.sum(object_x_axis * X_AXIS, axis=-1)
        y_alignment = np.sum(object_y_axis * Y_AXIS, axis=-1)
        z_alignment = np.sum(object_z_axis * Z_AXIS, axis=-1)
        orientation_error = (1.0 - x_alignment) + (1.0 - y_alignment) + (1.0 - z_alignment)
        object_orientation_reward = -config.w_orientation * orientation_error.mean(axis=-1)

        # Heading alignment reward - keep barrier's x-axis pointing towards goal
        # Note: object_pos from qpos includes initial state, but object_x_axis from sensors doesn't
        barrier_to_goal = np.array(config.goal_position)[None, None, :2] - object_pos[..., 1:, :2]  # Skip initial state to match sensors
        barrier_to_goal_norm = barrier_to_goal / (np.linalg.norm(barrier_to_goal, axis=-1, keepdims=True) + 1e-8)

        object_x_horizontal = object_x_axis[..., :2]  # Already excludes initial state (from sensors)
        object_x_horizontal_norm = object_x_horizontal / (np.linalg.norm(object_x_horizontal, axis=-1, keepdims=True) + 1e-8)

        heading_alignment = np.sum(object_x_horizontal_norm * barrier_to_goal_norm, axis=-1)
        heading_alignment_abs = np.abs(heading_alignment)
        heading_error = 1.0 - heading_alignment_abs
        heading_alignment_reward = -config.w_heading_alignment * heading_error.mean(axis=-1)

        # Approach site proximity reward
        distance_to_left_approach = np.linalg.norm(torso_to_approach_left, axis=-1)
        distance_to_right_approach = np.linalg.norm(torso_to_approach_right, axis=-1)
        approach_distances = np.stack([distance_to_left_approach, distance_to_right_approach], axis=-1)
        min_approach_distance = np.min(approach_distances, axis=-1)
        approach_site_proximity_reward = -config.w_approach_site_proximity * min_approach_distance.mean(-1)

        # Torso proximity reward
        torso_proximity_reward = config.w_torso_proximity * np.minimum(
            config.torso_proximity_threshold, np.linalg.norm(body_pos - object_pos, axis=-1)
        ).mean(-1)

        # Gripper to grasp point proximity
        distance_to_left = np.linalg.norm(gripper_to_grasp_left, axis=-1)
        distance_to_right = np.linalg.norm(gripper_to_grasp_right, axis=-1)
        grasp_distances = np.stack([distance_to_left, distance_to_right], axis=-1)
        min_grasp_distance = np.min(grasp_distances, axis=-1)
        gripper_to_grasp_proximity_reward = -config.w_gripper_to_grasp_proximity * min_grasp_distance.mean(-1)

        # Gripper orientation reward - adaptive based on position
        # Note: gripper_pos from sensors excludes initial state, object_pos from qpos includes it
        object_to_gripper = gripper_pos - object_pos[..., 1:, :]  # Skip initial state to match sensors

        gripper_x_axis_norm = gripper_x_axis / (np.linalg.norm(gripper_x_axis, axis=-1, keepdims=True) + 1e-8)
        object_y_axis_norm = object_y_axis / (np.linalg.norm(object_y_axis, axis=-1, keepdims=True) + 1e-8)

        # Determine if gripper is in front (+Y) or behind (-Y) the object
        projection_on_y = np.sum(object_to_gripper * object_y_axis_norm, axis=-1, keepdims=True)

        # Gripper should point toward the object
        target_y_axis = np.where(projection_on_y > 0, -object_y_axis_norm, object_y_axis_norm)

        x_dot_product = np.sum(gripper_x_axis_norm * target_y_axis, axis=-1)
        x_alignment_error = 1.0 - x_dot_product

        # Gripper Y-axis aligned with object Z-axis (upward)
        gripper_y_axis_norm = gripper_y_axis / (np.linalg.norm(gripper_y_axis, axis=-1, keepdims=True) + 1e-8)
        object_z_axis_norm = object_z_axis / (np.linalg.norm(object_z_axis, axis=-1, keepdims=True) + 1e-8)
        y_dot_product = np.sum(gripper_y_axis_norm * object_z_axis_norm, axis=-1)
        y_alignment_error = 1.0 - y_dot_product

        total_orientation_error = x_alignment_error + y_alignment_error
        gripper_orientation_reward = -config.w_gripper_orientation * total_orientation_error.mean(axis=-1)

        # Object velocity penalty
        object_linear_velocity_reward = -config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        # Control penalty
        controls_reward = -config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        # Grasp-based rewards
        false_grasp_penalty = -config.w_false_grasp_penalty * is_false_grasp.any(axis=-1)
        grasp_quality = np.clip(position_error / 0.5, 0, 1)
        grasp_quality_reward = config.w_grasp_quality * (is_grasping * grasp_quality).mean(axis=-1)

        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert object_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert object_orientation_reward.shape == (batch_size,)
        assert heading_alignment_reward.shape == (batch_size,)
        assert approach_site_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_to_grasp_proximity_reward.shape == (batch_size,)
        assert gripper_orientation_reward.shape == (batch_size,)
        assert object_linear_velocity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)
        assert false_grasp_penalty.shape == (batch_size,)
        assert grasp_quality_reward.shape == (batch_size,)

        return (
            spot_fence_reward
            + spot_fallen_reward
            + object_fallen_reward
            + goal_reward
            + object_orientation_reward
            + heading_alignment_reward
            + gripper_to_grasp_proximity_reward
            + gripper_orientation_reward
            + false_grasp_penalty
            + grasp_quality_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object - barrier starts upright."""
        # Barrier starts upright (no rotation)
        reset_object_pose = np.array([*DEFAULT_OBJECT_POS, 0.035, 1, 0, 0, 0])

        return np.array(
            [
                *DEFAULT_SPOT_POS,
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *reset_object_pose,
            ]
        )

    def success(self, model: MjModel, data: MjData, config: SpotBarrierDragConfig, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the barrier has been successfully dragged to the goal position."""
        object_pos = data.qpos[self.object_pose_idx[0:3]]
        goal_position = np.array(config.goal_position)

        # Check position tolerance
        distance_to_goal = np.linalg.norm(object_pos - goal_position)
        position_success = distance_to_goal <= config.position_tolerance

        return bool(position_success)
