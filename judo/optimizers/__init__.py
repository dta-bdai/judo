# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Type

from judo.optimizers.base import Optimizer, OptimizerConfig
from judo.optimizers.cem import CrossEntropyMethod, CrossEntropyMethodConfig
from judo.optimizers.mppi import MPPI, MPPIConfig
from judo.optimizers.cmaes import CMAES, CMAESConfig
from judo.optimizers.overrides import (
    set_default_caltech_leap_cube_overrides,
    set_default_cartpole_overrides,
    set_default_cylinder_push_overrides,
    set_default_fr3_pick_overrides,
    set_default_leap_cube_down_overrides,
    set_default_leap_cube_overrides,
    set_default_spot_yellow_chair_ramp_overrides,
    set_default_spot_barbell_overrides,
    set_default_spot_table_drag_overrides,
    set_default_spot_traffic_cone_overrides,
    set_default_spot_box_overrides,
    set_default_spot_locomotion_overrides,
    set_default_spot_box_push_baseline_overrides,
    set_default_spot_chair_push_baseline_overrides,
    set_default_spot_tire_push_baseline_overrides,
    set_default_spot_rack_push_baseline_overrides,
    set_default_spot_cone_push_baseline_overrides,
)
from judo.optimizers.ps import PredictiveSampling, PredictiveSamplingConfig
from judo.optimizers.skill_policy import SkillPolicyOptimizer, SkillPolicyConfig

set_default_caltech_leap_cube_overrides()
set_default_cartpole_overrides()
set_default_cylinder_push_overrides()
set_default_fr3_pick_overrides()
set_default_leap_cube_overrides()
set_default_leap_cube_down_overrides()
set_default_spot_yellow_chair_ramp_overrides()
set_default_spot_barbell_overrides()
set_default_spot_table_drag_overrides()
set_default_spot_traffic_cone_overrides()
set_default_spot_box_overrides()
set_default_spot_locomotion_overrides()
set_default_spot_box_push_baseline_overrides()
set_default_spot_chair_push_baseline_overrides()
set_default_spot_tire_push_baseline_overrides()
set_default_spot_rack_push_baseline_overrides()
set_default_spot_cone_push_baseline_overrides()

_registered_optimizers: dict[str, tuple[Type[Optimizer], Type[OptimizerConfig]]] = {
    "cem": (CrossEntropyMethod, CrossEntropyMethodConfig),
    "mppi": (MPPI, MPPIConfig),
    "ps": (PredictiveSampling, PredictiveSamplingConfig),
    "cmaes": (CMAES, CMAESConfig),
    "skill_policy": (SkillPolicyOptimizer, SkillPolicyConfig),
}


def get_registered_optimizers() -> dict[str, tuple[Type[Optimizer], Type[OptimizerConfig]]]:
    """Get the registered optimizer."""
    return _registered_optimizers


def register_optimizer(
    name: str,
    controller_type: Type[Optimizer],
    controller_config_type: Type[OptimizerConfig],
) -> None:
    """Register a new optimizer."""
    _registered_optimizers[name] = (controller_type, controller_config_type)


__all__ = [
    "get_registered_optimizers",
    "register_optimizer",
    "CrossEntropyMethod",
    "CrossEntropyMethodConfig",
    "MPPI",
    "MPPIConfig",
    "CMAES",
    "CMAESConfig",
    "Optimizer",
    "OptimizerConfig",
    "PredictiveSampling",
    "PredictiveSamplingConfig",
    "SkillPolicyOptimizer",
    "SkillPolicyConfig",
]
