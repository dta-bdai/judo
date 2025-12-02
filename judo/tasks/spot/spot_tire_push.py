# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjModel, MjData

from judo.utils.indexing import get_pos_indices, get_sensor_indices
from judo.utils.math_utils import apply_quat_to_vec, quat_mul
from judo import MODEL_PATH
from judo.utils.fields import np_1d_field
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
Z_AXIS = np.array([0.0, 0.0, 1.0])

@dataclass
class SpotTirePushConfig(SpotBaseConfig):
    """Config for the spot tire pushing task."""
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, TIRE_RADIUS], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.0],
        steps=[0.1, 0.1, 0.05],
        vis_name="box_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, TIRE_RADIUS],
    )

    w_tire_orientation: float = 100.0
    w_gripper_proximity: float = 4.0
    w_torso_proximity: float = 0.1
    orientation_threshold: float = 0.5

class SpotTirePush(SpotBase):
    """Task getting Spot to push a tire to a goal location."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["tire_rubber_joint"])
        self.tire_y_axis_idx = get_sensor_indices(self.model, "tire_y_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")
        self.fl_pos_idx = get_sensor_indices(self.model, "fl_pos")
        self.fr_pos_idx = get_sensor_indices(self.model, "fr_pos")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotTirePushConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot tire pushing task."""
        batch_size = states.shape[0]

        # (batch, horizon, size)
        # or (batch, horizon) if scalar
        qpos = states[..., : self.model.nq]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]
        tire_y_axis = sensors[..., self.tire_y_axis_idx]

        # Compute unit vector pointing from tire to torso
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx]   
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object, axis=-1
        ).mean(axis=-1)

        tire_orientation_reward = -config.w_tire_orientation * np.abs(
            np.dot(tire_y_axis, Z_AXIS) > config.orientation_threshold
        ).sum(axis=-1)

        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        torso_proximity_reward = config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        # Check if any state in the rollout has spot fallen
        body_height = qpos[..., self.body_pose_idx[2]]
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)

        # Compute a penalty to prefer small commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert tire_orientation_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        reward = (
            + spot_fallen_reward
            + goal_reward
            + tire_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
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
            # tire_pose[3:] = quat_mul(random_yaw_quat, tire_pose[3:])
            tire_pose[3:] = np.array([1, 0, 0, 0])

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
