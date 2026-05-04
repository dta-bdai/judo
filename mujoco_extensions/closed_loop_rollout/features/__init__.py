# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Feature factory compatibility layer for closed_loop_rollout."""

from __future__ import annotations

from collections.abc import Sequence

from mujoco_extensions.closed_loop_rollout import Feature, Policy


def make_local_velocity_feature(body_name: str, mujoco_system: object) -> Feature:
    del body_name, mujoco_system
    return Feature("local_velocity", 3)


def make_free_joint_angular_velocity_feature(joint_name: str, mujoco_system: object) -> Feature:
    del joint_name, mujoco_system
    return Feature("free_joint_angular_velocity", 3)


def make_local_gravity_feature(body_name: str, mujoco_system: object) -> Feature:
    del body_name, mujoco_system
    return Feature("local_gravity", 3)


def make_command_feature(size: int) -> Feature:
    return Feature("command", int(size))


def make_joint_position_feature(joint_names: Sequence[str], mujoco_system: object) -> Feature:
    del mujoco_system
    return Feature("joint_position", len(tuple(joint_names)))


def make_joint_velocity_feature(joint_names: Sequence[str], mujoco_system: object) -> Feature:
    del mujoco_system
    return Feature("joint_velocity", len(tuple(joint_names)))


def make_policy_output_feature(policy: Policy) -> Feature:
    del policy
    return Feature("policy_output", 12)


def make_joint_control_feature(joint_names: Sequence[str], mujoco_system: object) -> Feature:
    del mujoco_system
    return Feature("joint_control", len(tuple(joint_names)))


def make_direct_joint_control_feature(joint_names: Sequence[str], mujoco_system: object) -> Feature:
    del mujoco_system
    return Feature("direct_joint_control", len(tuple(joint_names)))


def make_skip_command_feature(size: int) -> Feature:
    return Feature("skip_command", int(size))


def make_spot_leg_command_override_feature(mujoco_system: object) -> Feature:
    del mujoco_system
    return Feature("spot_leg_command_override", 12)
