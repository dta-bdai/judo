# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from judo.optimizers.base import Optimizer, OptimizerConfig
from judo.utils.indexing import get_pos_indices, get_vel_indices
import onnxruntime


@dataclass
class SkillPolicyConfig(OptimizerConfig):
    """Configuration for skill policy optimizer.

    This optimizer uses a pre-trained ONNX network to directly generate control actions,
    rather than sampling and optimizing trajectories.
    """

    onnx_model_path: str = "/home/jzhang/Documents/judo-private/scripts/skill_policies/best_skill_policy (1).onnx"

    onnx_session = onnxruntime.InferenceSession(
            str(onnx_model_path),
            providers=["CPUExecutionProvider"]
        )

    # Policy parameters
    policy_frequency: float = 20.0  # Hz - frequency at which policy runs
    locomotion_only: bool = False  # Whether to only use locomotion (no arm commands)

    # Target command - position of target object in world frame
    target_cmd: np.ndarray | None = None  # Shape (3,) for xyz position

    # Previous actions for history (initialized automatically)
    skill_action_length: int = 10  # Length of skill action vector

    # Override num_rollouts to 1 since we don't do sampling
    num_rollouts: int = 0


class SkillPolicyOptimizer(Optimizer[SkillPolicyConfig]):
    """Skill policy optimizer using pre-trained ONNX network.

    Unlike traditional optimizers that sample and optimize trajectories,
    this optimizer uses a neural network policy to directly predict
    control actions based on the current state.

    The policy is treated as a "degenerate optimizer" that:
    1. Takes a single "sample" (the current state)
    2. Runs the network to get predicted actions
    3. Returns those actions as the "nominal" trajectory
    """

    def __init__(self, config: SkillPolicyConfig, nu: int) -> None:
        """Initialize skill policy optimizer.

        Args:
            config: Configuration for skill policy
            nu: Number of control dimensions (should match policy output)
        """
        super().__init__(config, nu)

        if config.onnx_session is None:
            raise ValueError("onnx_session must be provided in SkillPolicyConfig")

        self.session = config.onnx_session
        self.policy_frequency = config.policy_frequency
        self.locomotion_only = config.locomotion_only
        self.target_cmd = config.target_cmd if config.target_cmd is not None else np.zeros(3)

        # Initialize previous actions for policy history
        self.previous_skill_action = np.zeros(config.skill_action_length, dtype=np.float32)
        self.previous_locomotion_action = np.zeros(12, dtype=np.float32)
        self.previous_arm_target_pos = np.zeros(7, dtype=np.float32)  # ARM_UNSTOWED_POS

        # Cache for current state (updated in pre_optimization)
        self.current_state: np.ndarray | None = None
        self.model = None  # Set externally if needed for state indexing

    def set_current_state(self, state: np.ndarray, model: Any = None) -> None:
        """Set the current state for policy inference.

        Args:
            state: Current state vector (qpos, qvel concatenated)
            model: MuJoCo model (optional, for dynamic state indexing)
        """
        self.current_state = state
        if model is not None:
            self.model = model

    def convert_state_to_onnx_input(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """Convert MuJoCo state to ONNX input format.

        Uses dynamic state indexing via utils.indexing to properly extract state components,
        matching the reference implementation from dexterity codebase.

        Args:
            state: State vector (qpos, qvel concatenated)

        Returns:
            Dictionary of ONNX inputs with batch dimension added
        """
        from judo.tasks.spot.spot_constants import mujoco_to_isaac
        from judo.utils.indexing import get_pos_indices, get_vel_indices

        if self.model is None:
            raise ValueError("Model must be set before converting state to ONNX input")

        nq = self.model.nq
        nv = self.model.nv
        qpos = state[:nq]
        qvel = state[nq:]

        # Extract base state using dynamic indexing
        base_pos_idx = get_pos_indices(self.model, "base")
        base_vel_idx = get_vel_indices(self.model, "base")

        base_pos = qpos[base_pos_idx[0:3]]
        base_quat = qpos[base_pos_idx[3:7]]  # w, x, y, z
        base_lin_vel = qvel[base_vel_idx[0:3]]
        base_ang_vel = qvel[base_vel_idx[3:6]]

        # Extract joint states
        # Get all joint names (legs + arm)
        from judo.tasks.spot.spot_constants import JOINT_NAMES_BOSDYN
        try:
            joint_pos_idx = get_pos_indices(self.model, JOINT_NAMES_BOSDYN)
            joint_vel_idx = get_vel_indices(self.model, JOINT_NAMES_BOSDYN)
            joint_pos = qpos[joint_pos_idx]
            joint_vel = qvel[joint_vel_idx]
        except ValueError:
            # Fallback for tasks without all joints (e.g., spot_box without legs)
            # Use hardcoded indices
            joint_pos = qpos[7:26] if nq >= 26 else qpos[7:min(nq, 26)]
            joint_vel = qvel[6:25] if nv >= 25 else qvel[6:min(nv, 25)]
            # Pad to 19 joints if needed
            if len(joint_pos) < 19:
                joint_pos = np.pad(joint_pos, (0, 19 - len(joint_pos)))
            if len(joint_vel) < 19:
                joint_vel = np.pad(joint_vel, (0, 19 - len(joint_vel)))

        # Extract object state (first object)
        # Try to find box_joint, object_joint, or similar
        obj_joint_names = ["box_joint", "object_joint", "object", "cube_joint"]
        obj_pos = np.zeros(3)
        obj_quat = np.array([1.0, 0.0, 0.0, 0.0])
        obj_lin_vel = np.zeros(3)
        obj_ang_vel = np.zeros(3)

        for obj_name in obj_joint_names:
            try:
                obj_pos_idx = get_pos_indices(self.model, obj_name)
                obj_vel_idx = get_vel_indices(self.model, obj_name)
                obj_pos = qpos[obj_pos_idx[0:3]]
                obj_quat = qpos[obj_pos_idx[3:7]]
                obj_lin_vel = qvel[obj_vel_idx[0:3]]
                obj_ang_vel = qvel[obj_vel_idx[3:6]]
                break
            except (ValueError, IndexError):
                continue

        # Convert angular velocities from body frame to world frame
        def ang_vel_world(ang_vel_b: np.ndarray, w_q_b: np.ndarray) -> np.ndarray:
            """Convert angular velocity from local frame to world frame."""
            rot = R.from_quat(w_q_b, scalar_first=True)
            return rot.apply(ang_vel_b)

        base_ang_vel_w = ang_vel_world(base_ang_vel, base_quat)
        obj_ang_vel_w = ang_vel_world(obj_ang_vel, obj_quat)

        # Convert joint positions from MuJoCo to Isaac ordering
        joint_pos_isaac = mujoco_to_isaac(joint_pos)
        joint_vel_isaac = mujoco_to_isaac(joint_vel)

        # Build target command (support tracking second object in future)
        target_cmd_w = np.concatenate([self.target_cmd, [1.0, 0.0, 0.0, 0.0]])

        # Build ONNX input dict (matching reference implementation)
        onnx_inputs = {
            "robot_pos_w": base_pos,
            "robot_quat_w": base_quat,
            "robot_lin_vel_w": base_lin_vel,
            "robot_ang_vel_w": base_ang_vel_w,
            "joint_pos": joint_pos_isaac,
            "joint_vel": joint_vel_isaac,
            "object_pos_w": obj_pos,
            "object_quat_w": obj_quat,
            "object_lin_vel_w": obj_lin_vel,
            "object_ang_vel_w": obj_ang_vel_w,
            "target_cmd_w": target_cmd_w,
            "previous_skill_action": self.previous_skill_action.copy(),
            "previous_arm_target_pos": self.previous_arm_target_pos.copy(),
            "previous_locomotion_action": self.previous_locomotion_action.copy(),
            "locomotion_only": np.array([self.locomotion_only], dtype=bool),
        }

        # Convert to float32 and add batch dimension
        onnx_inputs = {k: v.astype(np.float32)[np.newaxis, :] for k, v in onnx_inputs.items()}

        return onnx_inputs

    def run_policy(self, state: np.ndarray) -> np.ndarray:
        """Run the ONNX policy network to get control actions.

        Args:
            state: Current state vector

        Returns:
            Control action vector of shape (nu,)
        """
        # Convert state to ONNX input format
        onnx_inputs = self.convert_state_to_onnx_input(state)

        # Run ONNX inference
        output = self.session.run(None, onnx_inputs)

        # Unpack outputs (remove batch dimension)
        leg_pos_target, arm_target_pos, locomotion_actions, skill_actions = [v[0] for v in output]

        # Update previous actions for next step
        self.previous_skill_action = skill_actions
        self.previous_locomotion_action = locomotion_actions
        if not self.locomotion_only:
            self.previous_arm_target_pos = arm_target_pos

        # Convert from Isaac to MuJoCo ordering
        from judo.tasks.spot.spot_constants import isaac_to_mujoco, ARM_UNSTOWED_POS
        leg_pos_mujoco = isaac_to_mujoco(leg_pos_target)

        # Build control vector - always 19 dims (12 legs + 7 arm)
        # This matches the reference implementation in rollout_skill_policy
        target_q = np.zeros(19)
        target_q[0:12] = leg_pos_mujoco

        if self.locomotion_only:
            # Use default unstowed position for arm
            target_q[12:19] = ARM_UNSTOWED_POS
        else:
            target_q[12:19] = arm_target_pos

        # Validate output dimension matches task
        if self.nu != 19:
            raise ValueError(
                f"Skill policy outputs 19 joint positions but task expects nu={self.nu}. "
                f"Skill policy should only be used with tasks that have nu=19 "
                f"(e.g., spot_baseline tasks, not spot tasks with custom control mapping)."
            )

        return target_q

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Sample control knots - for skill policy, we just return current policy output.

        This method is required by the Optimizer interface but doesn't do traditional
        sampling. Instead, it runs the policy network once to get the predicted actions.

        Args:
            nominal_knots: Ignored (policy doesn't use nominal trajectory)

        Returns:
            Single "sample" from the policy, shape (1, num_nodes, nu)
        """
        if self.current_state is None:
            raise ValueError("Must call set_current_state() before sampling")

        # Run policy to get control action
        control_action = self.run_policy(self.current_state)

        # Expand to match expected shape: (num_rollouts=1, num_nodes, nu)
        # We repeat the same action across all nodes since policy outputs a single action
        sampled_knots = np.tile(control_action, (1, self.num_nodes, 1))

        return sampled_knots

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update nominal knots - for skill policy, just return the policy output.

        Since we only have one "sample" (the policy output), we just return it.
        Rewards are ignored since the policy is pre-trained.

        Args:
            sampled_knots: The policy output, shape (1, num_nodes, nu)
            rewards: Ignored

        Returns:
            The policy output as the nominal trajectory, shape (num_nodes, nu)
        """
        return sampled_knots[0]

    def pre_optimization(self, old_times: np.ndarray, new_times: np.ndarray) -> None:
        """Called before optimization to update internal state.

        For skill policy, this is where we could update policy parameters if needed.
        Currently just a pass-through.
        """
        pass

    def stop_cond(self) -> bool:
        """Skill policy always stops after one iteration (no optimization loop).

        Returns:
            True to stop after first iteration
        """
        # Since we only need one policy evaluation, stop after first iteration
        return False  # Let the controller handle max_opt_iters=1
