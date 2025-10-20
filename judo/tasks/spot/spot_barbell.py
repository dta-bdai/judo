# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from judo.utils.indexing import get_pos_indices, get_sensor_indices
from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.tasks.spot.spot_base import GOAL_POSITIONS, SpotBase, SpotBaseConfig

XML_PATH = str(MODEL_PATH / "xml/spot_components/spot_barbell.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])

# annulus object position sampling
RADIUS_MIN = 1.0
RADIUS_MAX = 3.0
USE_LEGS = False

@dataclass
class SpotBarbellConfig(SpotBaseConfig):
    """Config for the spot barbell lift task."""

    goal_position: np.ndarray = field(default_factory=lambda: GOAL_POSITIONS().origin + np.array([0, 0, 1.0]))

    # Parameters
    robot_proximity_radius: float = 0.75
    grasp_distance_threshold: float = 0.15  # Distance threshold to consider object "graspable"
    grasp_height_threshold: float = 0.3  # Height threshold for considering barbell as "lifted"

    # Reward weights
    w_goal: float = 50.0
    w_goal_height: float = 200.0
    w_torso_proximity: float = 50.0
    w_gripper_proximity: float = 250.0
    w_gripper_barbell_orientation: float = 50.0
    w_barbell_orientation: float = 100.0
    w_gripper_close: float = 100.0  # Reward for closing gripper when grasping


class SpotBarbell(SpotBase[SpotBarbellConfig]):
    """Task getting Spot to lift and move a barbell to a desired goal location."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS, use_arm=True, use_gripper=True)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["barbell_joint"])
        self.gripper_joint_idx = get_pos_indices(self.model, "arm_f1x")

        # Sensor indices
        self.gripper_y_axis_idx = get_sensor_indices(self.model, "sensor_gripper_y_axis")
        self.barbell_x_axis_idx = get_sensor_indices(self.model, "sensor_barbell_x_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")


    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotBarbellConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot barbell lifting task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]

        W_p_torso = qpos[..., self.body_pose_idx[:3]]
        W_p_barbell = qpos[..., self.object_pose_idx[:3]]
        body_height = qpos[..., self.body_pose_idx[2]]
        gripper_joint_pos = qpos[..., self.gripper_joint_idx[0]]

        gripper_to_object = sensors[..., self.end_effector_to_object_idx]
        gripper_y_axis = sensors[..., self.gripper_y_axis_idx]
        barbell_x_axis = sensors[..., self.barbell_x_axis_idx]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (body_height <= config.spot_fallen_threshold).any(axis=-1)

        # Goal reward - penalize distance from barbell to goal
        goal_reward = -config.w_goal * np.linalg.norm(
            W_p_barbell - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        # Goal height reward - particularly important for lifting
        goal_height_error = np.abs(W_p_barbell[..., 2] - config.goal_position[2]).mean(axis=-1)
        goal_height_reward = -config.w_goal_height * goal_height_error

        # Torso proximity reward - encourage robot to stay close to barbell
        torso_t_barbell = W_p_barbell[..., :2] - W_p_torso[..., :2]
        torso_t_barbell_norm = (
            config.robot_proximity_radius
            - np.clip(np.linalg.norm(torso_t_barbell, axis=-1), 0, config.robot_proximity_radius)
        ) / config.robot_proximity_radius
        torso_proximity_reward = -config.w_torso_proximity * torso_t_barbell_norm.mean(axis=-1)

        # Gripper proximity reward - encourage gripper to be close to barbell
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            gripper_to_object, axis=-1
        ).mean(axis=-1)

        # Barbell orientation reward - encourage barbell to be horizontal (x-axis in XY plane)
        # We want the x-axis to be horizontal, so penalize z-component
        orientation_error = 1.0 - np.linalg.norm(barbell_x_axis[..., :2], axis=-1)
        barbell_orientation_reward = -config.w_barbell_orientation * orientation_error.mean(axis=-1)

        # Gripper-barbell orientation alignment reward
        # Compute angle between gripper_y_axis and barbell_x_axis
        gripper_y_axis_norm = gripper_y_axis / (np.linalg.norm(gripper_y_axis, axis=-1, keepdims=True) + 1e-8)
        barbell_x_axis_norm = barbell_x_axis / (np.linalg.norm(barbell_x_axis, axis=-1, keepdims=True) + 1e-8)
        dot_product = np.sum(gripper_y_axis_norm * barbell_x_axis_norm, axis=-1)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip for numerical stability
        gripper_barbell_orientation_reward = -config.w_gripper_barbell_orientation * (1.0 - np.abs(dot_product)).mean(axis=-1)

        # Gripper closing reward when grasping
        # Define "grasping" as: gripper is close to object AND object is lifted above threshold
        # Note: sensors have one fewer timestep than qpos, so we need to align dimensions
        gripper_distance = np.linalg.norm(gripper_to_object, axis=-1)
        n_sensor_timesteps = gripper_distance.shape[-1]
        barbell_height = W_p_barbell[..., :n_sensor_timesteps, 2]
        gripper_joint_pos_aligned = gripper_joint_pos[..., :n_sensor_timesteps]

        is_grasping = (gripper_distance < config.grasp_distance_threshold) & (barbell_height > config.grasp_height_threshold)
        # Normalize gripper position: 0 = closed, -1.54 = fully open
        gripper_openness = (gripper_joint_pos_aligned - 0.0) / (-1.54 - 0.0)
        gripper_closeness = 1.0 - gripper_openness
        gripper_close_reward = -config.w_gripper_close * ((1.0 - gripper_closeness) * is_grasping).mean(axis=-1)

        # Compute a penalty to prefer small commands
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert goal_height_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert barbell_orientation_reward.shape == (batch_size,)
        assert gripper_barbell_orientation_reward.shape == (batch_size,)
        assert gripper_close_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            spot_fallen_reward
            + goal_reward
            + goal_height_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + barbell_orientation_reward
            + gripper_barbell_orientation_reward
            + gripper_close_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        standing_pose = np.array([0, 0, STANDING_HEIGHT])
        robot_radius = 1.0

        object_xyz = (np.random.rand(3) * 2.0 - 1.0) * 3.0
        while np.linalg.norm(object_xyz[:2]) < robot_radius:
            object_xyz = (np.random.rand(3) * 2.0 - 1.0) * 3.0
        object_xyz[2] = 0.1  # On the ground

        # Randomize orientation: x axis in XY plane, z axis up
        theta = np.random.uniform(0, 2 * np.pi)  # rotation about Z axis
        # Quaternion for rotation about Z axis by theta: (w, x, y, z)
        object_orientation = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

        return np.array(
            [*standing_pose, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_xyz, *object_orientation]
        )
