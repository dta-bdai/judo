# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjModel, MjData

from judo.utils.indexing import get_pos_indices, get_sensor_indices
from judo.utils.math_utils import apply_quat_to_vec, quat_mul
from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_tire.xml")

TIRE_RADIUS: float = 0.339
TIRE_WIDTH: float = 0.175
USE_LEGS = True

DEFAULT_TORSO_POSITION = np.array([-1.75, 0, STANDING_HEIGHT])


@dataclass
class SpotTireUprightConfig(SpotBaseConfig):
    """Config for the spot tire uprighting task."""

    orientation_error_smoothing_width: float = 1.0
    w_tire_orientation: float = 200.0
    w_gripper_proximity: float = 10.0
    w_foot_proximity: float = 5.0
    w_torso_proximity: float = 5.0
    gripper_too_inside_tire_penalty: float = 150.0
    gripper_not_above_tire_penalty: float = 100.0


class SpotTireUpright(SpotBase):
    """Task getting Spot to upright a tire lying flat on the ground."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["tire_rubber_joint"])
        self.tire_y_axis_idx = get_sensor_indices(self.model, "tire_y_axis")
        self.gripper_pos_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")
        self.fl_pos_idx = get_sensor_indices(self.model, "fl_pos")
        self.fr_pos_idx = get_sensor_indices(self.model, "fr_pos")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotTireUprightConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot tire uprighting task."""
        batch_size = states.shape[0]

        # (batch, horizon, size)
        # or (batch, horizon) if scalar
        qpos = states[..., : self.model.nq]

        #######
        ### Proximity rewards
        #######

        # Compute unit vector pointing from tire to torso
        W_p_tire = qpos[..., self.object_pose_idx[:3]]
        W_p_torso = qpos[..., self.body_pose_idx[:3]]
        W_p_tire_torso = W_p_torso - W_p_tire
        W_u_tire_torso = W_p_tire_torso / np.linalg.norm(W_p_tire_torso, axis=-1, keepdims=True)

        # Compute the desired gripper position
        # W_p_gripper_des = W_p_tire + (TIRE_RADIUS - 0.05) * W_u_tire_torso
        # W_p_gripper_des[..., 2] = TIRE_WIDTH + 0.1

        # # Compute the distance from gripper to desired gripper position
        # W_p_gripper = sensors[..., self.gripper_pos_idx]
        # gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
        #     W_p_gripper - W_p_gripper_des, axis=-1
        # ).mean(axis=-1)

        # Compute the desired right foot position
        pi_over_4_yaw_quat = np.array([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
        W_u_tire_torso_rotated = apply_quat_to_vec(quat=pi_over_4_yaw_quat, vec=W_u_tire_torso)
        W_p_right_foot_des = W_p_tire + TIRE_RADIUS * W_u_tire_torso_rotated
        W_p_right_foot_des[..., 2] = 0.1

        # Compute the desired left foot position
        pi_over_4_yaw_quat = np.array([np.cos(np.pi / 8), 0, 0, np.sin(-np.pi / 8)])
        W_u_tire_torso_rotated = apply_quat_to_vec(quat=pi_over_4_yaw_quat, vec=W_u_tire_torso)
        W_p_left_foot_des = W_p_tire + TIRE_RADIUS * W_u_tire_torso_rotated
        W_p_left_foot_des[..., 2] = 0.1

        # Compute the distance from right foot to desired foot position
        # W_p_foot = sensors[..., self.fr_pos_idx]
        # right_foot_proximity_reward = -config.w_foot_proximity * np.linalg.norm(
        #     W_p_foot - W_p_right_foot_des, axis=-1
        # ).mean(axis=-1)

        # Compute the distance from left foot to desired foot position
        # W_p_foot = sensors[..., self.fl_pos_idx]
        # left_foot_proximity_reward = -config.w_foot_proximity * np.linalg.norm(
        #     W_p_foot - W_p_left_foot_des, axis=-1
        # ).mean(axis=-1)

        # # Use the maximum reward of both feet
        # foot_proximity_reward = np.maximum(right_foot_proximity_reward, left_foot_proximity_reward)

        # Compute distance from torso to desired torso position
        W_p_torso_des = W_p_tire + 0.75 * W_u_tire_torso
        W_p_torso_des[..., 2] = STANDING_HEIGHT  # Enforce standing height

        # Compute the distance from torso to desired torso position
        torso_proximity_reward = -config.w_torso_proximity * np.linalg.norm(
            W_p_torso - W_p_torso_des, axis=-1
        ).mean(axis=-1)

        #######
        ### Goal rewards
        #######

        # Compute orientation reward
        tire_y_axis = sensors[..., self.tire_y_axis_idx]
        orientation_error = np.abs(tire_y_axis[..., 2])  # 0 to 1
        orientation_error_smooth = np.exp(
            orientation_error / config.orientation_error_smoothing_width
        )  # 1 to e
        orientation_reward = -config.w_tire_orientation * orientation_error_smooth.mean(axis=-1)

        #######
        ### Penalize bad behavior
        #######

        # Don't put the gripper too close to tire center
        # gripper_distance_from_tire = np.linalg.norm(W_p_gripper - W_p_tire, axis=-1)
        # gripper_inside_tire_reward = -config.gripper_too_inside_tire_penalty * (
        #     gripper_distance_from_tire < (TIRE_RADIUS * 0.5)
        # ).mean(axis=-1)

        # # Don't put the gripper below the tire if too far from the tire center
        # gripper_height = W_p_gripper[..., 2]
        # gripper_not_above_tire = gripper_height < 2 * TIRE_WIDTH + 0.05
        # gripper_too_far_from_tire = gripper_distance_from_tire > TIRE_RADIUS
        # gripper_not_above_tire_reward = -config.gripper_not_above_tire_penalty * (
        #     np.logical_and(gripper_not_above_tire, gripper_too_far_from_tire)
        # ).mean(axis=-1)

        # Check if any state in the rollout has spot fallen
        body_height = qpos[..., self.body_pose_idx[2]]
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)

        # Compute a penalty to prefer small commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert orientation_reward.shape == (batch_size,)
        # assert gripper_proximity_reward.shape == (batch_size,)
        # assert foot_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)

        # assert gripper_inside_tire_reward.shape == (batch_size,)
        # assert gripper_not_above_tire_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        reward = (
            orientation_reward
            # + gripper_proximity_reward
            # + foot_proximity_reward
            + torso_proximity_reward
            # + gripper_inside_tire_reward
            # + gripper_not_above_tire_reward
            + spot_fallen_reward
            + controls_reward
        )
        return reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        for _ in range(100):

            #######
            ### Generate random tire pose
            #######

            # Place tire at random x and y
            tire_pose = np.zeros(7)
            tire_pose[0] = np.random.uniform(-2, 2)
            tire_pose[1] = np.random.uniform(-2, 2)

            # z is picked so that tire is lying flat
            tire_pose[2] = TIRE_WIDTH

            # orientation is picked first by applying +/-90 degree roll
            if np.random.random() < 0.5:
                tire_pose[3:] = [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]
            else:
                tire_pose[3:] = [1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0]
            # then apply random yaw
            random_yaw = np.random.uniform(0, 2 * np.pi)
            random_yaw_quat = np.array([np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)])
            tire_pose[3:] = quat_mul(random_yaw_quat, tire_pose[3:])

            #######
            ### Generate random robot pose
            #######

            # Place robot at random x and y
            robot_pose = np.zeros(7)
            robot_pose[0] = np.random.uniform(-2, 2)
            robot_pose[1] = np.random.uniform(-2, 2)
            robot_pose[2] = STANDING_HEIGHT

            # Robot has random yaw
            random_yaw_robot = np.random.uniform(0, 2 * np.pi)
            robot_pose[3:] = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])

            # Check that robot and tire are sufficiently far apart
            if np.linalg.norm(robot_pose[:3] - tire_pose[:3]) > 2:
                return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_pose])

        # Ran out of attempts, use default reset pose
        tire_pose = np.zeros(7)
        tire_pose[2] = TIRE_WIDTH
        tire_pose[3:] = [np.sin(np.pi / 4), np.cos(np.pi / 4), 0, 0]  # Apply 90 degree roll
        return np.array([*DEFAULT_TORSO_POSITION, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_pose])
