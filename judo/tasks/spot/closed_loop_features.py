# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Feature setup helpers for Spot closed-loop rollout."""

from __future__ import annotations

from pathlib import Path

from mujoco import MjModel

import mujoco_extensions.closed_loop_rollout as clr
import mujoco_extensions.closed_loop_rollout.features as features

from judo.tasks.spot.spot_constants import ARM_JOINT_NAMES, LEG_JOINT_NAMES_BOSDYN


def build_spot_closed_loop_features(model: MjModel, policy_path: str | Path) -> list[clr.Feature]:
    """Build Spot locomotion/manipulation features matching Starfish wiring."""

    policy = clr.Policy(str(policy_path))
    mujoco_system = clr.make_mujoco_system_from_model(model)

    full_joint_names = [f"spot/{joint_name}" for joint_name in [*LEG_JOINT_NAMES_BOSDYN, *ARM_JOINT_NAMES]]
    leg_joint_names = [f"spot/{joint_name}" for joint_name in LEG_JOINT_NAMES_BOSDYN]
    arm_joint_names = [f"spot/{joint_name}" for joint_name in ARM_JOINT_NAMES]

    return [
        features.make_local_velocity_feature("spot/torso", mujoco_system),
        features.make_free_joint_angular_velocity_feature("spot/base", mujoco_system),
        features.make_local_gravity_feature("spot/torso", mujoco_system),
        features.make_command_feature(25),
        features.make_joint_position_feature(full_joint_names, mujoco_system),
        features.make_joint_velocity_feature(full_joint_names, mujoco_system),
        features.make_policy_output_feature(policy),
        features.make_joint_control_feature(leg_joint_names, mujoco_system),
        features.make_skip_command_feature(3),
        features.make_direct_joint_control_feature(arm_joint_names, mujoco_system),
        features.make_spot_leg_command_override_feature(mujoco_system),
    ]
