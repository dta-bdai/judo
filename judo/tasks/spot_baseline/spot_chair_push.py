"""Spot chair manipulation task built on the locomotion baseline."""

from dataclasses import dataclass
from typing import Any

from mujoco import MjModel, MjData
import mujoco
import numpy as np

from judo.utils.fields import np_1d_field

from judo import MODEL_PATH
from judo.tasks.spot_baseline.spot_base import GOAL_POSITIONS
from judo.tasks.spot_baseline.spot_constants import STANDING_HEIGHT, LEGS_STANDING_POS
from judo.tasks.spot_baseline.spot_locomotion import (
    SpotLocomotion,
    SpotLocomotionConfig,
)
from judo.utils.indexing import get_pos_indices, get_sensor_indices, get_vel_indices

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_yellow_chair.xml")
Z_AXIS = np.array([0.0, 0.0, 1.0])
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0

DEFAULT_CHAIR_HEIGHT = 0.0

# Success condition tolerances
POSITION_TOLERANCE = 0.2
VELOCITY_TOLERANCE = 0.05


@dataclass
class SpotChairPushBaselineConfig(SpotLocomotionConfig):
    """Config for the Spot chair manipulation task built on locomotion."""

    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, STANDING_HEIGHT], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.5],
        steps=[0.1, 0.1, 0.05],
        vis_name="robot_goal_unused",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, STANDING_HEIGHT],
    )
    w_position: float = 0.0

    chair_goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, DEFAULT_CHAIR_HEIGHT], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.0],
        steps=[0.1, 0.1, 0.05],
        vis_name="box_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, DEFAULT_CHAIR_HEIGHT],
    )

    w_orientation: float = 15.0
    w_torso_proximity: float = 0.1
    w_gripper_proximity: float = 4.0
    w_object_velocity: float = 20.0
    orientation_threshold: float = 0.7


class SpotChairPushBaseline(SpotLocomotion):
    """Spot locomotion task augmented with chair manipulation rewards."""

    name: str = "spot_chair_push_baseline"
    config_t: type[SpotChairPushBaselineConfig] = SpotChairPushBaselineConfig

    def __init__(
        self,
        model_path: str = XML_PATH,
        sim_model_path: str | None = None,
        config: SpotChairPushBaselineConfig | None = None,
    ) -> None:
        super().__init__(model_path=model_path, sim_model_path=sim_model_path, config=config)

        self.object_pose_idx = get_pos_indices(self.model, ["yellow_chair_joint"])
        self.object_vel_idx = get_vel_indices(self.model, ["yellow_chair_joint"])
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Compose locomotion rewards with chair manipulation objectives."""
        locomotion_reward = super().reward(states, sensors, controls, system_metadata)

        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]
        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]
        object_linear_velocity = qvel[..., self.object_vel_idx[0:3]]

        object_y_axis = sensors[..., self.object_y_axis_idx]
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx]

        fallen = (body_height <= self.config.spot_fallen_threshold).any(axis=1)
        fall_penalty = -self.config.fall_penalty * fallen.astype(np.float64)

        chair_goal = np.asarray(self.config.chair_goal_position, dtype=np.float64)
        object_goal_error = np.linalg.norm(object_pos - chair_goal, axis=-1)
        goal_reward = -self.config.w_goal * np.mean(object_goal_error, axis=1)

        alignment = np.sum(object_y_axis * Z_AXIS, axis=-1)
        orientation_violation = (np.abs(alignment) < self.config.orientation_threshold).astype(np.float64)
        orientation_reward = -self.config.w_orientation * np.mean(orientation_violation, axis=1)

        torso_distance = np.linalg.norm(body_pos - object_pos, axis=-1)
        torso_proximity = self.config.w_torso_proximity * np.mean(torso_distance, axis=1)

        gripper_distance = np.linalg.norm(end_effector_to_object, axis=-1)
        gripper_proximity = -self.config.w_gripper_proximity * np.mean(gripper_distance, axis=1)

        object_linear_velocity_penalty = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        return (
            locomotion_reward
            + fall_penalty
            + goal_reward
            + orientation_reward
            + torso_proximity
            + gripper_proximity
            + object_linear_velocity_penalty
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""

        # Sample object position in annulus
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + 0.1 * np.random.randn(2)

        object_pose = np.array([*object_pos, DEFAULT_CHAIR_HEIGHT, 1, 0, 0, 0])

        # Place robot at random x and y
        robot_pose_xy = np.random.uniform(-0.5, 0.5, 2)
        random_yaw_robot = np.random.uniform(0, 2 * np.pi)
        robot_pose_orientation = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])
        robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the chair is in the goal position."""
        object_pos = data.qpos[..., self.object_pose_idx[0:3]]
        object_vel = data.qvel[..., self.object_vel_idx[0:3]]
        goal_pos = np.array(self.config.chair_goal_position)
        position_check = np.linalg.norm(object_pos - goal_pos, axis=-1, ord=np.inf) < POSITION_TOLERANCE
        velocity_check = np.linalg.norm(object_vel, axis=-1) < VELOCITY_TOLERANCE
        return position_check and velocity_check

    def failure(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if Spot has fallen."""
        body_height = data.qpos[..., self.body_pose_idx[2]]
        return body_height <= self.config.spot_fallen_threshold
