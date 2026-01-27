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

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_tire.xml")

TIRE_RADIUS = 0.339
TIRE_WIDTH = 0.175
USE_LEGS = True
USE_GRIPPER = True
USE_TORSO = True

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([0.0, 0.0])
DEFAULT_TIRE_POS = np.array([2.0, 0.0])


@dataclass
class SpotTireUprightConfig(SpotBaseConfig):
    """Config for the spot tire uprighting task."""

    w_fence: float = 1000.0
    w_tire_orientation: float = 200.0
    orientation_error_smoothing_width: float = 1.0
    w_gripper_proximity: float = 10.0
    w_foot_proximity: float = 5.0
    w_torso_proximity: float = 5.0
    gripper_too_inside_tire_penalty: float = 150.0
    gripper_not_above_tire_penalty: float = 100.0
    w_controls: float = 2.0


class SpotTireUpright(SpotBase[SpotTireUprightConfig]):
    """Task getting Spot to get a tire upright."""

    name: str = "spot_tire_upright"
    config_t: type[SpotTireUprightConfig] = SpotTireUprightConfig

    def __init__(self, model_path: str = XML_PATH, config: SpotTireUprightConfig | None = None) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS, use_gripper=USE_GRIPPER, use_torso=USE_TORSO, config=config)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["tire_rubber_joint"])
        self.object_vel_idx = get_vel_indices(self.model, ["tire_rubber_joint"])

        # Object orientation sensor (y-axis for tire uprighting)
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")

        # Gripper position sensor
        self.gripper_pos_idx = get_sensor_indices(self.model, "trace_fngr_site")

        # Foot position sensors
        self.fl_pos_idx = get_sensor_indices(self.model, "fl_pos")
        self.fr_pos_idx = get_sensor_indices(self.model, "fr_pos")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot tire uprighting task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]

        object_y_axis = sensors[..., self.object_y_axis_idx]

        #######
        ### Proximity rewards
        #######

        # Note: Sensors exclude initial state, so we need to skip initial state from qpos
        W_p_tire = object_pos[..., 1:, :]  # Skip initial state to match sensors
        W_p_torso = body_pos[..., 1:, :]  # Skip initial state to match sensors
        W_p_tire_torso = W_p_torso - W_p_tire
        W_u_tire_torso = W_p_tire_torso / (np.linalg.norm(W_p_tire_torso, axis=-1, keepdims=True) + 1e-8)

        # Compute the desired gripper position
        W_p_gripper_des = W_p_tire + (TIRE_RADIUS - 0.05) * W_u_tire_torso
        W_p_gripper_des[..., 2] = TIRE_WIDTH + 0.1

        # Compute the distance from gripper to desired gripper position
        W_p_gripper = sensors[..., self.gripper_pos_idx]
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            W_p_gripper - W_p_gripper_des, axis=-1
        ).mean(axis=-1)

        # Helper function to rotate vector by yaw
        def apply_yaw_rotation(vec, yaw_angle):
            """Apply yaw rotation to a vector."""
            cos_yaw = np.cos(yaw_angle)
            sin_yaw = np.sin(yaw_angle)
            # Rotation around z-axis
            rotated_x = vec[..., 0] * cos_yaw - vec[..., 1] * sin_yaw
            rotated_y = vec[..., 0] * sin_yaw + vec[..., 1] * cos_yaw
            result = np.stack([rotated_x, rotated_y, vec[..., 2]], axis=-1)
            return result

        # Compute the desired right foot position
        W_u_tire_torso_rotated = apply_yaw_rotation(W_u_tire_torso, np.pi / 4)
        W_p_right_foot_des = W_p_tire + TIRE_RADIUS * W_u_tire_torso_rotated
        W_p_right_foot_des[..., 2] = 0.1

        # Compute the desired left foot position
        W_u_tire_torso_rotated = apply_yaw_rotation(W_u_tire_torso, -np.pi / 4)
        W_p_left_foot_des = W_p_tire + TIRE_RADIUS * W_u_tire_torso_rotated
        W_p_left_foot_des[..., 2] = 0.1

        # Compute the distance from right foot to desired foot position
        W_p_foot = sensors[..., self.fr_pos_idx]
        right_foot_proximity_reward = -self.config.w_foot_proximity * np.linalg.norm(
            W_p_foot - W_p_right_foot_des, axis=-1
        ).mean(axis=-1)

        # Compute the distance from left foot to desired foot position
        W_p_foot = sensors[..., self.fl_pos_idx]
        left_foot_proximity_reward = -self.config.w_foot_proximity * np.linalg.norm(
            W_p_foot - W_p_left_foot_des, axis=-1
        ).mean(axis=-1)

        # Use the maximum reward of both feet
        foot_proximity_reward = np.maximum(right_foot_proximity_reward, left_foot_proximity_reward)

        # Compute distance from torso to desired torso position
        W_p_torso_des = W_p_tire + 0.75 * W_u_tire_torso
        W_p_torso_des[..., 2] = STANDING_HEIGHT  # Enforce standing height

        # Compute the distance from torso to desired torso position
        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(
            W_p_torso - W_p_torso_des, axis=-1
        ).mean(axis=-1)

        #######
        ### Goal rewards
        #######

        # Compute orientation reward (tire y-axis should be horizontal, so z-component should be 0)
        orientation_error = np.abs(object_y_axis[..., 2])  # 0 to 1
        orientation_error_smooth = np.exp(
            orientation_error / self.config.orientation_error_smoothing_width
        )  # 1 to e
        orientation_reward = -self.config.w_tire_orientation * orientation_error_smooth.mean(axis=-1)

        #######
        ### Penalize bad behavior
        #######

        # Don't put the gripper too close to tire center
        gripper_distance_from_tire = np.linalg.norm(W_p_gripper - W_p_tire, axis=-1)
        gripper_inside_tire_reward = -self.config.gripper_too_inside_tire_penalty * (
            gripper_distance_from_tire < (TIRE_RADIUS * 0.5)
        ).mean(axis=-1)

        # Don't put the gripper below the tire if too far from the tire center
        gripper_height = W_p_gripper[..., 2]
        gripper_not_above_tire = gripper_height < 2 * TIRE_WIDTH + 0.05
        gripper_too_far_from_tire = gripper_distance_from_tire > TIRE_RADIUS
        gripper_not_above_tire_reward = -self.config.gripper_not_above_tire_penalty * (
            np.logical_and(gripper_not_above_tire, gripper_too_far_from_tire)
        ).mean(axis=-1)

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

        assert orientation_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert foot_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_inside_tire_reward.shape == (batch_size,)
        assert gripper_not_above_tire_reward.shape == (batch_size,)
        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            orientation_reward
            + gripper_proximity_reward
            + foot_proximity_reward
            + torso_proximity_reward
            + gripper_inside_tire_reward
            + gripper_not_above_tire_reward
            + spot_fence_reward
            + spot_fallen_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and tire - tire starts lying flat."""
        for _ in range(100):
            # Place tire at random x and y
            tire_x = np.random.uniform(-2, 2)
            tire_y = np.random.uniform(-2, 2)

            # z is picked so that tire is lying flat
            tire_z = TIRE_WIDTH

            # orientation is picked first by applying +/-90 degree roll
            if np.random.random() < 0.5:
                # 90 degree roll: quat = [cos(45), sin(45), 0, 0]
                tire_quat = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])
            else:
                # -90 degree roll: quat = [cos(45), -sin(45), 0, 0]
                tire_quat = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0])

            # then apply random yaw
            random_yaw = np.random.uniform(0, 2 * np.pi)
            yaw_quat = np.array([np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)])

            # Quaternion multiplication: yaw_quat * tire_quat
            w1, x1, y1, z1 = yaw_quat
            w2, x2, y2, z2 = tire_quat
            tire_quat_final = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

            # Place robot at random x and y
            robot_x = np.random.uniform(-2, 2)
            robot_y = np.random.uniform(-2, 2)
            robot_z = STANDING_HEIGHT

            # Robot has random yaw
            random_yaw_robot = np.random.uniform(0, 2 * np.pi)
            robot_quat = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])

            # Check that robot and tire are sufficiently far apart
            robot_pos = np.array([robot_x, robot_y, robot_z])
            tire_pos = np.array([tire_x, tire_y, tire_z])
            if np.linalg.norm(robot_pos - tire_pos) > 2:
                return np.array([
                    *robot_pos,
                    *robot_quat,
                    *LEGS_STANDING_POS,
                    *self.reset_arm_pos,
                    *tire_pos,
                    *tire_quat_final,
                ])

        # Ran out of attempts, use default reset pose
        tire_pose = np.array([*DEFAULT_TIRE_POS, TIRE_WIDTH, np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        return np.array([
            *DEFAULT_SPOT_POS,
            STANDING_HEIGHT,
            1,
            0,
            0,
            0,
            *LEGS_STANDING_POS,
            *self.reset_arm_pos,
            *tire_pose,
        ])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the tire is upright (y-axis horizontal)."""
        # Get tire y-axis sensor data for orientation check
        object_y_axis = data.sensordata[self.object_y_axis_idx]

        # Check if y-axis is horizontal (z-component should be close to 0)
        orientation_error = np.abs(object_y_axis[2])
        orientation_success = orientation_error <= 0.1  # Small tolerance

        return bool(orientation_success)
