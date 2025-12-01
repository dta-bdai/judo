"""Simplified Spot base task using standard MuJoCo rollouts."""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import mujoco
import numpy as np

from judo.tasks import Task, TaskConfig

from judo import MODEL_PATH
from judo.tasks.spot_baseline.spot_constants import (
    ARM_STOWED_POS,
    ARM_UNSTOWED_POS,
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "spot/robot.xml")


@dataclass
class GOAL_POSITIONS:
    """Goal positions of Spot."""

    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.0]))
    origin_decimal: np.ndarray = field(default_factory=lambda: np.array([0, 0.0, 0.01]))
    blue_cross: np.ndarray = field(default_factory=lambda: np.array([2.77, 0.71, 0.3]))
    black_cross: np.ndarray = field(default_factory=lambda: np.array([1.5, -1.5, 0.275]))


@dataclass
class SpotBaseConfig(TaskConfig):
    """Base config for spot tasks."""

    fall_penalty: float = 2500.0
    spot_fallen_threshold: float = 0.35  # Torso height where Spot is considered "fallen"
    w_goal: float = 60.0
    w_controls: float = 0.0


ConfigT = TypeVar("ConfigT", bound=SpotBaseConfig)


class SpotBase(Task[ConfigT], Generic[ConfigT]):
    """Simplified base task for Spot using standard MuJoCo rollouts.

    This version uses the default MuJoCo rollout backend and does not override
    control mappings (nu, ctrlrange, task_to_sim_ctrl). Controls map directly
    to MuJoCo actuators defined in the XML.
    """

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Default reward function - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement reward()")

    @property
    def reset_arm_pos(self) -> np.ndarray:
        """Default arm pose for resets."""
        return ARM_UNSTOWED_POS

    @property
    def reset_pose(self) -> np.ndarray:
        """Standing pose with small random XY offset."""
        pose = np.zeros_like(self.data.qpos)
        pose[0:2] = 0.05 * np.random.randn(2)
        pose[2] = STANDING_HEIGHT
        pose[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        pose[7 : 7 + len(LEGS_STANDING_POS)] = LEGS_STANDING_POS
        pose[7 + len(LEGS_STANDING_POS) : 7 + len(LEGS_STANDING_POS) + len(self.reset_arm_pos)] = self.reset_arm_pos
        return pose

    def reset(self) -> None:
        """Reset the task state."""
        pose = self.reset_pose
        self.data.qpos[:] = pose[: self.model.nq]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
