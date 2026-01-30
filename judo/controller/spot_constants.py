# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

"""Constants for Spot locomotion controller.

This file contains index conversion mappings and other constants needed for the
hierarchical locomotion controller. These constants are shared between the judo
and starfish projects to ensure consistency.
"""

import numpy as np

# Index conversion between Isaac Gym and MuJoCo joint orderings
# Isaac ordering groups joints by type: arm_sh0, then all hip x, then arm_sh1, then all hip y, etc.
# MuJoCo ordering is per-leg: FL (hx, hy, kn), FR (hx, hy, kn), HL, HR, then arm

ISAAC_TO_MUJOCO_INDICES_12 = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
ISAAC_TO_MUJOCO_INDICES_19 = [1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18]
MUJOCO_TO_ISAAC_INDICES_12 = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
MUJOCO_TO_ISAAC_INDICES_19 = [12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18]


def isaac_to_mujoco(var_isaac: np.ndarray) -> np.ndarray:
    """Reorders an array from Isaac Gym to MuJoCo format.

    Args:
        var_isaac: Input array in Isaac Gym format (12 or 19 elements).

    Returns:
        Array reordered for MuJoCo.

    Raises:
        ValueError: If array size is not 12 or 19.
    """
    if len(var_isaac) == 12:
        isaac_to_mujoco_list = ISAAC_TO_MUJOCO_INDICES_12
    elif len(var_isaac) == 19:
        isaac_to_mujoco_list = ISAAC_TO_MUJOCO_INDICES_19
    else:
        raise ValueError(
            f"The vector must of size 12 (legs only) or 19 (arm + legs), size of given vector {var_isaac.shape}"
        )
    var_mujoco = var_isaac[isaac_to_mujoco_list]
    return var_mujoco


def mujoco_to_isaac(var_mujoco: np.ndarray) -> np.ndarray:
    """Reorders an array from MuJoCo to Isaac Gym format.

    Args:
        var_mujoco: Input array in MuJoCo format (12 or 19 elements).

    Returns:
        Array reordered for Isaac Gym.

    Raises:
        ValueError: If array size is not 12 or 19.
    """
    if len(var_mujoco) == 12:
        mujoco_to_isaac_list = MUJOCO_TO_ISAAC_INDICES_12
    elif len(var_mujoco) == 19:
        mujoco_to_isaac_list = MUJOCO_TO_ISAAC_INDICES_19
    else:
        raise ValueError(
            f"The vector must of size 12 (legs only) or 19 (arm + legs), size of given vector {var_mujoco.shape}"
        )
    var_isaac = var_mujoco[mujoco_to_isaac_list]
    return var_isaac


# Locomotion policy default joint offsets from RL training
# Used for input normalization & joint target offset
LOCOMOTION_DEFAULT_JOINTS_OFFSET = (
    0.0,
    0.12,
    -0.12,
    0.12,
    -0.12,
    -0.9,
    0.5,
    0.5,
    0.5,
    0.5,
    1.8,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    0.0,
    -0.9,
    0.0,
    -1.54,
)

LOCOMOTION_DEFAULT_LEGS_OFFSET = (0.12, -0.12, 0.12, -0.12, 0.5, 0.5, 0.5, 0.5, -1.0, -1.0, -1.0, -1.0)

# PD gains used during RL locomotion training
LEG_K_Q_P_RL = [60, 60, 60.0] * 4
LEG_K_QD_P_RL = [1.5, 1.5, 1.5] * 4
ARM_K_Q_P_RL = [120.0, 120.0, 120.0, 100.0, 100.0, 100.0, 16.0]
ARM_K_QD_P_RL = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.32]
K_Q_P_RL = LEG_K_Q_P_RL + ARM_K_Q_P_RL
K_QD_P_RL = LEG_K_QD_P_RL + ARM_K_QD_P_RL

# Torque limits used during RL locomotion training
LEG_TORQUE_LIMITS_RL = [45.0, 45.0, 60.0] * 4
ARM_TORQUE_LIMITS_RL = [90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32]
TORQUE_LIMITS_RL = LEG_TORQUE_LIMITS_RL + ARM_TORQUE_LIMITS_RL

# Command format constants
# Command structure: [base_vel(3), arm(7), legs(12), torso(3)] = 25 dimensions
RL_LOCOMOTION_COMMAND_LENGTH = 25
BASE_VEL_INDICES = slice(0, 3)
ARM_COMMAND_INDICES = slice(3, 10)
LEG_COMMAND_INDICES = slice(10, 22)
TORSO_COMMAND_INDICES = slice(22, 25)


def validate_command_format(cmd: np.ndarray) -> None:
    """Validate that command has the correct format.

    Args:
        cmd: Command array to validate.

    Raises:
        ValueError: If command format is invalid.
    """
    if cmd.ndim not in (1, 2):
        raise ValueError(f"Command must be 1D or 2D array, got shape {cmd.shape}")

    cmd_dim = cmd.shape[-1]
    if cmd_dim != RL_LOCOMOTION_COMMAND_LENGTH:
        raise ValueError(
            f"Command must have {RL_LOCOMOTION_COMMAND_LENGTH} dimensions "
            f"[base_vel(3), arm(7), legs(12), torso(3)], got {cmd_dim}"
        )
