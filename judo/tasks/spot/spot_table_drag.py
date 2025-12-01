# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mujoco import MjModel, MjData

from judo.utils.indexing import get_pos_indices, get_sensor_indices, get_vel_indices
from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.tasks.spot.spot_base import GOAL_POSITIONS, SpotBase, SpotBaseConfig

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_table_short.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
TABLE_HEIGHT = 0.51  # Height of the table surface from the ground
TABLE_WIDTH = 0.8  # Width of the table
USE_LEGS = False

@dataclass
class SpotTableDragConfig(SpotBaseConfig):
    """Config for the spot table drag task."""

    goal_position: np.ndarray = field(default_factory=lambda: GOAL_POSITIONS().origin)

    # Parameters
    robot_proximity_radius: float = 1.0

    # Reward weights
    w_goal: float = 500.0
    w_torso_proximity: float = 200.0
    w_gripper_proximity: float = 250.0
    w_gripper_orientation: float = 1000.0
    w_table_orientation: float = 100.0

    # Success criteria
    position_tolerance: float = 0.2
    orientation_tolerance: float = 0.1


class SpotTableDrag(SpotBase[SpotTableDragConfig]):
    """Task getting Spot to drag a table to a desired goal location."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS, use_arm=True, use_gripper=True)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["table_short_joint"])

        # Sensor indices
        self.gripper_x_axis_idx = get_sensor_indices(self.model, "sensor_gripper_x_axis")
        self.gripper_y_axis_idx = get_sensor_indices(self.model, "sensor_gripper_y_axis")
        self.table_x_axis_idx = get_sensor_indices(self.model, "sensor_table_x_axis")
        self.table_y_axis_idx = get_sensor_indices(self.model, "sensor_table_y_axis")
        self.table_z_axis_idx = get_sensor_indices(self.model, "sensor_table_z_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")


    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotTableDragConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot table dragging task."""
        batch_size = states.shape[0]
        reward = np.zeros((batch_size,))

        qpos = states[..., : self.model.nq]

        W_p_torso = qpos[..., self.body_pose_idx[:3]]
        W_p_table = qpos[..., self.object_pose_idx[:3]]
        W_p_gripper = sensors[..., self.gripper_pos_idx]
        gripper_to_object = sensors[..., self.end_effector_to_object_idx]
        gripper_x_axis = sensors[..., self.gripper_x_axis_idx]
        gripper_y_axis = sensors[..., self.gripper_y_axis_idx]
        table_x_axis = sensors[..., self.table_x_axis_idx]
        table_y_axis = sensors[..., self.table_y_axis_idx]
        table_z_axis = sensors[..., self.table_z_axis_idx]

        # Calculate gripper tip
        W_p_gripper_tip = W_p_gripper + 0.1 * gripper_x_axis

        # Calculate contact points on table edges (4 edges at table height) - relative to table center
        # These are vectors from table center to contact points
        contact_offsets = []
        for i, j in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            contact_offsets.append(
                table_x_axis * i * (TABLE_WIDTH / 2)
                + table_y_axis * j * (TABLE_WIDTH / 2)
                + table_z_axis * TABLE_HEIGHT
            )
        contact_offsets = np.stack(contact_offsets, axis=-2)  # Shape: (batch_size, horizon, 4, 3)

        # Unused variable to avoid issues
        _ = gripper_y_axis  # Not used in this implementation but kept for potential future use

        #######
        ### Objective rewards
        #######
        # Goal reward - only care about XY position
        reward += -config.w_goal * np.linalg.norm(
            W_p_table[..., :2] - np.array(config.goal_position[:2])[None, None], axis=-1
        ).mean(-1)

        # Torso proximity reward - encourage robot to stay close to table
        torso_t_table = W_p_table[..., :2] - W_p_torso[..., :2]
        torso_t_table_norm = (
            config.robot_proximity_radius
            - np.clip(np.linalg.norm(torso_t_table, axis=-1), 0, config.robot_proximity_radius)
        ) / config.robot_proximity_radius
        reward += -config.w_torso_proximity * torso_t_table_norm.mean(axis=-1)

        # Gripper proximity to contact points reward
        # gripper_to_object is vector from object center to gripper
        # contact_offsets are vectors from object center to contact points
        # So gripper to contact = gripper_to_object - contact_offset
        gripper_to_contacts = gripper_to_object[..., None, :] - contact_offsets
        # Add gripper tip extension (0.1m along gripper x-axis)
        gripper_to_contacts = gripper_to_contacts - 0.1 * gripper_x_axis[..., None, :]

        gripper_to_contacts_dist = np.linalg.norm(gripper_to_contacts, axis=-1)
        # Find closest contact point index for each batch
        contact_index = gripper_to_contacts_dist.mean(axis=-2).argmin(axis=-1)
        row_indices = np.arange(batch_size)
        gripper_t_contact_dist = gripper_to_contacts_dist[row_indices, :, contact_index].mean(axis=-1)
        reward += -config.w_gripper_proximity * gripper_t_contact_dist

        # Gripper orientation alignment reward
        # Align gripper to push toward table center from closest contact point
        # contact_direction points from contact point to table center+height
        closest_contact_offset = contact_offsets[row_indices, :, contact_index, :]
        contact_direction = table_z_axis * TABLE_HEIGHT - closest_contact_offset
        contact_direction_norm = contact_direction / (np.linalg.norm(contact_direction, axis=-1, keepdims=True) + 1e-8)
        gripper_x_axis_norm = gripper_x_axis / (np.linalg.norm(gripper_x_axis, axis=-1, keepdims=True) + 1e-8)
        dot_product = np.sum(gripper_x_axis_norm * contact_direction_norm, axis=-1)
        # Clip for numerical stability
        dot_product = np.clip(dot_product, -1.0, 1.0)
        reward += -config.w_gripper_orientation * (1.0 - np.abs(dot_product)).mean(axis=-1)

        # Table orientation reward - keep table upright
        reward += -config.w_table_orientation * (1.0 - np.maximum(table_z_axis[..., 2].mean(axis=-1), 0.0))

        #######
        ### Penalize bad behavior
        #######
        # Check if any state in the rollout has spot fallen
        body_height = qpos[..., self.body_pose_idx[2]]
        reward += -config.fall_penalty * (body_height <= config.spot_fallen_threshold).any(axis=-1)

        # Compute a penalty to prefer small commands
        reward += -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        return reward


    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        standing_pose = np.array([0, 0, STANDING_HEIGHT])
        robot_radius = 1.0

        object_xyz = (np.random.rand(3) * 2.0 - 1.0) * 3.0
        while np.linalg.norm(object_xyz[:2]) < robot_radius:
            object_xyz = (np.random.rand(3) * 2.0 - 1.0) * 3.0
        object_xyz[2] = 0.52  # Table base height

        # Randomize orientation: x axis in XY plane, z axis up
        theta = np.random.uniform(0, 2 * np.pi)  # rotation about Z axis
        # Quaternion for rotation about Z axis by theta: (w, x, y, z)
        object_orientation = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

        return np.array(
            [*standing_pose, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_xyz, *object_orientation]
        )

    def success(self, model: MjModel, data: MjData, config: SpotTableDragConfig, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the table is upright and at the goal position."""
        # Get object z-axis sensor data for orientation check
        object_z_axis = data.sensordata[self.table_z_axis_idx]

        # Check orientation tolerance (object should be upright, z-axis aligned with world z-axis)
        orientation_alignment = np.dot(object_z_axis, Z_AXIS)
        orientation_success = orientation_alignment >= (1.0 - config.orientation_tolerance)

        # Get object position
        object_pos = data.qpos[self.object_pose_idx[0:3]]

        # Check if object is within position tolerance of goal (XY only)
        position_error = np.linalg.norm(object_pos[:2] - config.goal_position[:2])
        position_success = position_error <= config.position_tolerance

        return bool(orientation_success and position_success)
