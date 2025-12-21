# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjModel, MjData

from judo.utils.indexing import get_pos_indices, get_sensor_indices, get_vel_indices
from judo.utils.math_utils import apply_quat_to_vec, quat_mul
from judo import MODEL_PATH
from judo.utils.fields import np_1d_field
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_traffic_cone.xml")

USE_LEGS = False
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0
DEFAULT_CONE_HEIGHT = 0.0

DEFAULT_TORSO_POSITION = np.array([-1.75, 0, STANDING_HEIGHT])
Z_AXIS = np.array([0.0, 0.0, 1.0])
# Success condition tolerances
POSITION_TOLERANCE = 0.2
VELOCITY_TOLERANCE = 0.05

@dataclass
class SpotConePushConfig(SpotBaseConfig):
    """Config for the spot cone pushing task."""
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, DEFAULT_CONE_HEIGHT], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.0],
        steps=[0.1, 0.1, 0.05],
        vis_name="box_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, DEFAULT_CONE_HEIGHT],
    )

    w_object_orientation: float = 100.0
    w_gripper_proximity: float = 4.0
    w_torso_proximity: float = 0.1
    orientation_threshold: float = 0.7
    w_object_velocity: float = 20.0

class SpotConePush(SpotBase):
    """Task getting Spot to push a cone to a goal location."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["traffic_cone_joint"])
        self.object_vel_idx = get_vel_indices(self.model, ["traffic_cone_joint"])
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotConePushConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot cone pushing task."""
        batch_size = states.shape[0]

        # (batch, horizon, size)
        # or (batch, horizon) if scalar
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]
        object_y_axis = sensors[..., self.object_y_axis_idx]
        object_linear_velocity = qvel[..., self.object_vel_idx[0:3]]

        # Compute unit vector pointing from tire to torso
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx]   
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object, axis=-1
        ).mean(axis=-1)

        object_orientation_reward = -config.w_object_orientation * np.abs(
            np.dot(object_y_axis, Z_AXIS) > config.orientation_threshold
        ).sum(axis=-1)

        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        torso_proximity_reward = config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        object_linear_velocity_penalty = -config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )
        # Check if any state in the rollout has spot fallen
        body_height = qpos[..., self.body_pose_idx[2]]
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)

        # Compute a penalty to prefer small commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert object_orientation_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_penalty.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        reward = (
            + spot_fallen_reward
            + goal_reward
            + object_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + object_linear_velocity_penalty
            + controls_reward
        )
        return reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""

        # Sample object position in annulus
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + 0.1 * np.random.randn(2)

        object_pose = np.array([*object_pos, DEFAULT_CONE_HEIGHT, 1, 0, 0, 0])

        # Place robot at random x and y
        robot_pose_xy = np.random.uniform(-0.5, 0.5, 2)
        random_yaw_robot = np.random.uniform(0, 2 * np.pi)
        robot_pose_orientation = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])
        robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(self, model: MjModel, data: MjData, config: SpotConePushConfig, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the cone is in the goal position."""
        object_pos = data.qpos[..., self.object_pose_idx[0:3]]
        object_vel = data.qvel[..., self.object_vel_idx[0:3]]
        goal_pos = np.array(config.goal_position)
        position_check = np.linalg.norm(object_pos - goal_pos, axis=-1, ord=np.inf) < POSITION_TOLERANCE
        velocity_check = np.linalg.norm(object_vel, axis=-1) < VELOCITY_TOLERANCE
        return position_check and velocity_check

    def failure(self, model: MjModel, data: MjData, config: SpotConePushConfig, metadata: dict[str, Any] | None = None) -> bool:
        """Check if Spot has fallen."""
        body_height = data.qpos[..., self.body_pose_idx[2]]
        return body_height <= config.spot_fallen_threshold