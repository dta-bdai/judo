# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mujoco import MjModel, MjData

from judo.utils.indexing import get_pos_indices, get_sensor_indices
from judo.utils.fields import np_1d_field
from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.tasks.spot.spot_base import GOAL_POSITIONS, SpotBase, SpotBaseConfig

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_box.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
RESET_OBJECT_POSE = np.array([3, 0, 0.275, 1, 0, 0, 0])
# annulus object position sampling
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0
USE_LEGS = False

@dataclass
class SpotBoxPushConfig(SpotBaseConfig):
    """Config for the spot box pushing task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, 0.254], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.0],
        steps=[0.1, 0.1, 0.05],
        vis_name="box_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, 0.254],
    )
    w_orientation: float = 100.0
    w_torso_proximity: float = 0.1
    w_gripper_proximity: float = 4.0
    orientation_threshold: float = 0.5


class SpotBoxPush(SpotBase):
    """Task getting Spot to push a box to a desired goal location."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["box_joint"])
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")


    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotBoxPushConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot box pushing task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]

        object_y_axis = sensors[..., self.object_y_axis_idx]
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (body_height <= config.spot_fallen_threshold).any(axis=-1)

        # Compute l2 distance from object pos. to goal.
        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        box_orientation_reward = -config.w_orientation * np.abs(
            np.dot(object_y_axis, Z_AXIS) > config.orientation_threshold
        ).sum(axis=-1)

        # Compute l2 distance from torso pos. to object pos.
        torso_proximity_reward = config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        # Compute l2 distance from torso pos. to object pos.
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object,
            axis=-1,
        ).mean(-1)

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert box_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            spot_fallen_reward
            + goal_reward
            + box_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object.

        Ensures robot and box are at least 0.5m apart.
        """
        MIN_DISTANCE = 0.5  # Minimum distance between robot and box
        max_attempts = 100

        # Initialize with defaults
        base_xy = np.zeros(2)
        object_pos = np.zeros(2)

        for _ in range(max_attempts):
            # Sample robot base position
            base_xy = np.random.randn(2)

            # Sample object position in annulus
            radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
            theta = 2 * np.pi * np.random.rand()
            object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + 0.1 * np.random.randn(2)

            # Check distance
            distance = np.linalg.norm(base_xy - object_pos[:2])
            if distance >= MIN_DISTANCE:
                break

        reset_object_pose = np.array([*object_pos, 0.254, 1, 0, 0, 0])

        return np.array(
            [
                *base_xy,
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

    def success(self, model: MjModel, data: MjData, config: SpotBoxPushConfig, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the box is in the goal position."""
        object_pos = data.qpos[..., self.object_pose_idx[0:3]]
        goal_pos = np.array(config.goal_position)
        return np.linalg.norm(object_pos - goal_pos, axis=-1, ord=np.inf) < 0.5