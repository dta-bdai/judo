"""Spot locomotion base task."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import mujoco

from judo import MODEL_PATH
from judo.utils.indexing import get_pos_indices
from judo.tasks.spot_baseline.spot_base import (
    GOAL_POSITIONS,
    SpotBase,
    SpotBaseConfig,
)
from judo.tasks.spot_baseline.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
    ARM_UNSTOWED_POS,
)
from judo.utils.fields import np_1d_field
from judo.gui import slider  

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_locomotion.xml")


@slider("goal_x", -5.0, 5.0, 0.1)
@slider("goal_y", -5.0, 5.0, 0.1)
@slider("goal_z", 0.1, 1.0, 0.05)
@slider("cadence_hz", 0.5, 4.0, 0.05)
@slider("duty_ratio", 0.2, 0.9, 0.01)
@slider("step_amplitude", 0.0, 0.15, 0.005)
@slider("height_goal", 0.3, 0.8, 0.01)
@slider("w_upright", 0.0, 500.0, 5.0)
@slider("w_height", 0.0, 200.0, 2.0)
@slider("w_position", 0.0, 200.0, 2.0)
@slider("w_gait", 0.0, 50.0, 1.0)
@slider("w_balance", 0.0, 100.0, 1.0)
@slider("w_posture", 0.0, 50.0, 1.0)
@slider("w_effort", 0.0, 0.01, 0.0001)
@slider("w_arm_posture", 0.0, 50.0, 1.0)
@dataclass
class SpotLocomotionConfig(SpotBaseConfig):
    """Config for Spot locomotion tasks, mirroring A1 MJPC quadruped structure."""

    # Goal position with GUI sliders and 3D visualization
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, STANDING_HEIGHT]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.5],
        steps=[0.1, 0.1, 0.05],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[2.0, 0.0, 0.55],
    )

    # Spot-specific trot timing (≈0.6s period) with longer stance phase for stability
    cadence_hz: float = 1.6  # 0.6s cycle measured from Spot choreographer gaits
    duty_ratio: float = 0.6  # keep feet on ground longer than A1 trot
    step_amplitude: float = 0.035  # tuned to Spot leg length and clearance

    # Physical targets
    height_goal: float = STANDING_HEIGHT

    # Residual weights (mirrors A1 task_flat user sensors)
    w_upright: float = 10.0
    w_height: float = 60.0
    w_position: float = 40.0
    w_gait: float = 10.0
    w_balance: float = 50.0
    w_posture: float = 5.0
    w_effort: float = 2.0e-3
    w_arm_posture: float = 10.0  # Keep arm in unstowed position


class SpotLocomotion(SpotBase[SpotLocomotionConfig]):
    """Base class for Spot locomotion tasks."""

    name: str = "spot_locomotion"
    config_t: type[SpotLocomotionConfig] = SpotLocomotionConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        super().__init__(model_path=model_path)

        # Get indices for base position
        self.body_pose_idx = get_pos_indices(self.model, "base")

        # Get indices for leg joints (assuming 12 leg joints: fl, fr, hl, hr each with 3 DOF)
        # Adjust these based on your actual joint names in the XML
        self.leg_start_idx = 7  # After base (3 pos + 4 quat)
        self.num_leg_joints = 12

        # Get indices for arm joints (7 DOF arm including gripper)
        self.arm_start_idx = self.leg_start_idx + self.num_leg_joints
        self.num_arm_joints = 7

        self.ground_height = self.model.geom("ground").size[2]
        self.foot_radius = self.model.geom("rear_left_foot_collision").size[0]

        # Get foot site IDs for forward kinematics
        self.foot_names: tuple[str, ...] = ("fl", "fr", "hl", "hr")
        self.foot_site_ids = {
            'fl': self.model.site("site_front_left").id,
            'fr': self.model.site("site_front_right").id,
            'hl': self.model.site("site_rear_left").id,
            'hr': self.model.site("site_rear_right").id,
        }
        self._foot_site_id_list = [self.foot_site_ids[name] for name in self.foot_names]

        # Foot order and phases mirror MJPC quadruped trot (FL/HR, FR/HL pairs)
        self._foot_phase = {
            'fl': 0.0,
            'hr': 0.0,
            'fr': 0.5,
            'hl': 0.5,
        }

        # Use Spot-specific standing pose rather than the A1 keyframe to avoid
        # driving the legs toward the wrong geometry.
        self._joint_targets = LEGS_STANDING_POS.copy()

        # Cache arm target for arm posture reward (stowed position)
        self._arm_targets = np.array(ARM_UNSTOWED_POS)

        # Phase velocity and fall time calculations (matching A1)
        # Use default config values for initialization
        default_config = SpotLocomotionConfig()
        self._dt = self.model.opt.timestep
        self._phase_velocity = 2.0 * np.pi * default_config.cadence_hz
        self._fall_time = np.sqrt(2.0 * default_config.height_goal / 9.81)
        self._time_offset = 0

        # Build sensor blocks dictionary
        self._sensor_blocks = self._build_sensor_blocks()

    # --------------------------------------------------------------------- #
    # Reward helpers (matching A1 structure)
    # --------------------------------------------------------------------- #
    def _build_sensor_blocks(self) -> dict[str, tuple[int, int]]:
        """Build dictionary mapping sensor names to (offset, dim) tuples."""
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for sensor_id in range(self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
            )
            dim = self.model.sensor_dim[sensor_id]
            if name:
                offsets[name] = (cursor, dim)
            cursor += dim
        return offsets

    def _sensor_block(self, sensors: np.ndarray, name: str) -> np.ndarray:
        """Extract a named sensor block from the sensor array."""
        start, dim = self._sensor_blocks[name]
        return sensors[..., start : start + dim]

    def _phase_signal_a1(self, horizon: int) -> np.ndarray:
        """Generate phase signal for gait (matching A1 implementation)."""
        indices = self._time_offset + np.arange(horizon)
        absolute_time = indices.astype(np.float64) * self._dt
        return self._phase_velocity * absolute_time

    def _step_height_a1(
        self,
        phase_signal: np.ndarray,
        footphase: float,
        config: SpotLocomotionConfig,
    ) -> np.ndarray:
        """Compute step height schedule (matching A1 implementation)."""
        angle = (phase_signal + np.pi - footphase) % (2.0 * np.pi) - np.pi
        duty = config.duty_ratio
        if duty >= 1.0:
            return np.zeros_like(angle)
        scaled = angle * (0.5 / (1.0 - duty))
        clipped = np.clip(scaled, -np.pi / 2.0, np.pi / 2.0)
        value = np.cos(clipped)
        value[np.abs(value) < 1e-6] = 0.0
        return config.step_amplitude * value

    def _foot_schedule_a1(self, phase_signal: np.ndarray, config: SpotLocomotionConfig) -> np.ndarray:
        """Generate foot height schedule for all feet (matching A1 implementation)."""
        schedule = np.zeros((phase_signal.shape[0], len(self.foot_names)))
        for idx, name in enumerate(self.foot_names):
            schedule[:, idx] = self._step_height_a1(
                phase_signal,
                2.0 * np.pi * self._foot_phase[name],
                config,
            )
        return schedule

    def _batch_foot_positions(self, qpos: np.ndarray) -> np.ndarray:
        """Compute foot positions for a batch of qpos sequences."""
        batch_size, horizon, _ = qpos.shape
        positions = np.zeros((batch_size, horizon, len(self.foot_names), 3))
        for b in range(batch_size):
            for t in range(horizon):
                self.data.qpos[:] = qpos[b, t]
                mujoco.mj_forward(self.model, self.data)
                for idx, site_id in enumerate(self._foot_site_id_list):
                    positions[b, t, idx] = self.data.site_xpos[site_id]
        return positions

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotLocomotionConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Python re-implementation of the MJPC QuadrupedFlat trot reward (matching A1)."""
        batch_size, horizon, _ = states.shape
        qpos = states[..., : self.model.nq]

        # Extract sensor data (matching A1 structure)
        torso_pos = self._sensor_block(sensors, "base_position")
        head_pos = self._sensor_block(sensors, "base_position")  # Use base as "head" for Spot
        torso_up_axis = self._sensor_block(sensors, "base_xaxis")  # Spot uses Z as up
        com_pos = self._sensor_block(sensors, "base_subtreecom")
        com_vel = self._sensor_block(sensors, "base_subtreelinvel")

        # Collect foot positions from sensors
        foot_blocks = [
            self._sensor_block(sensors, f"{name}_pos") for name in self.foot_names
        ]
        foot_pos = np.stack(foot_blocks, axis=2)

        avg_feet = foot_pos.mean(axis=2)
        avg_foot_height = avg_feet[..., 2]
        avg_foot_xy = avg_feet[..., :2]

        # Upright cost (matching A1)
        upright_error = torso_up_axis - np.array([0.0, 0.0, 1.0])
        upright_cost = -config.w_upright * np.mean(
            np.sum(upright_error**2, axis=-1),
            axis=1,
        )

        # Height cost (matching A1)
        height_error = (torso_pos[..., 2] - avg_foot_height) - config.height_goal
        height_cost = -config.w_height * np.mean(height_error**2, axis=1)

        # Position cost (matching A1)
        goal = np.asarray(config.goal_position, dtype=np.float64)
        pos_error = head_pos - goal
        position_cost = -config.w_position * np.mean(
            np.sum(pos_error**2, axis=-1),
            axis=1,
        )

        # Gait cost (matching A1)
        phase = self._phase_signal_a1(horizon)
        schedule = self._foot_schedule_a1(phase, config)[None, :, :]
        gait_mask = (schedule > 1e-6).astype(np.float64)
        foot_heights = foot_pos[..., 2]
        desired_height = self.ground_height + self.foot_radius + schedule
        gait_error = (foot_heights - desired_height) * gait_mask
        gait_cost = -config.w_gait * np.mean(gait_error**2, axis=(1, 2))

        # Balance cost (matching A1)
        capture_point = com_pos[..., :2] + com_vel[..., :2] * self._fall_time
        balance_error = capture_point - avg_foot_xy
        balance_cost = -config.w_balance * np.mean(
            np.sum(balance_error**2, axis=-1),
            axis=1,
        )

        # Posture cost (matching A1)
        posture_target = self._joint_targets[None, None, :]
        posture_error = qpos[..., self.leg_start_idx:self.leg_start_idx + self.num_leg_joints] - posture_target
        posture_cost = -config.w_posture * np.mean(
            np.sum(posture_error**2, axis=-1),
            axis=1,
        )

        # Arm posture cost (keep arm in unstowed position)
        arm_target = self._arm_targets[None, None, :]
        arm_error = qpos[..., self.arm_start_idx:self.arm_start_idx + self.num_arm_joints] - arm_target
        arm_posture_cost = -config.w_arm_posture * np.mean(
            np.sum(arm_error**2, axis=-1),
            axis=1,
        )

        # Effort cost (matching A1)
        effort_cost = -config.w_effort * np.mean(
            np.sum(controls**2, axis=-1),
            axis=1,
        )

        total_reward = (
            upright_cost
            + height_cost
            + position_cost
            + gait_cost
            + balance_cost
            + posture_cost
            + arm_posture_cost
            + effort_cost
        )
        return total_reward

    def optimizer_warm_start(self) -> np.ndarray:
        """Provide warm start controls keeping the arm unstowed."""
        return np.concatenate([LEGS_STANDING_POS, self._arm_targets])

    def get_sim_metadata(self) -> dict[str, Any]:
        """Return metadata for visualization.

        This passes the goal position to the visualizer for rendering traces.
        """
        # Use default config for visualization metadata
        default_config = SpotLocomotionConfig()
        return {
            "goal_pos": default_config.goal_position,
        }

    def pre_rollout(self, curr_state: np.ndarray, config: SpotLocomotionConfig) -> None:
        """Advance the internal gait clock before each MPC rollout (matching A1)."""
        self._time_offset += 1

    @property
    def reset_pose(self) -> np.ndarray:
        """Standing pose with small base perturbations."""
        pose = np.zeros_like(self.data.qpos)
        pose[0:2] = np.random.randn(2)
        pose[2] = STANDING_HEIGHT
        pose[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        pose[self.leg_start_idx : self.leg_start_idx + self.num_leg_joints] = LEGS_STANDING_POS
        pose[self.arm_start_idx : self.arm_start_idx + self.num_arm_joints] = self._arm_targets
        return pose
