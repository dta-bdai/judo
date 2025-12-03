# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_box_push import SpotBoxPush, SpotBoxPushConfig
from judo.tasks.spot.spot_crowd_barrier import SpotCrowdBarrier, SpotCrowdBarrierConfig
from judo.tasks.spot.spot_yellow_chair import SpotYellowChair, SpotYellowChairConfig
from judo.tasks.spot.spot_yellow_chair_ramp import SpotYellowChairRamp, SpotYellowChairRampConfig
from judo.tasks.spot.spot_traffic_cone import SpotTrafficCone, SpotTrafficConeConfig
from judo.tasks.spot.spot_barbell import SpotBarbell, SpotBarbellConfig
from judo.tasks.spot.spot_table_drag import SpotTableDrag, SpotTableDragConfig
from judo.tasks.spot.spot_tire_push import SpotTirePush, SpotTirePushConfig
from judo.tasks.spot.spot_chair_push import SpotChairPush, SpotChairPushConfig
from judo.tasks.spot.spot_rack_push import SpotRackPush, SpotRackPushConfig
from judo.tasks.spot.spot_cone_push import SpotConePush, SpotConePushConfig

from .spot_baseline import (
    SpotBoxPushBaseline,
    SpotBoxPushBaselineConfig,
    SpotTirePushBaseline,
    SpotTirePushBaselineConfig,
    SpotChairPushBaseline,
    SpotChairPushBaselineConfig,
    SpotRackPushBaseline,
    SpotRackPushBaselineConfig,
    SpotConePushBaseline,
    SpotConePushBaselineConfig,
)

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    "spot_base": (SpotBase, SpotBaseConfig),
    "spot_box_push": (SpotBoxPush, SpotBoxPushConfig),
    "spot_yellow_chair": (SpotYellowChair, SpotYellowChairConfig),
    "spot_yellow_chair_ramp": (SpotYellowChairRamp, SpotYellowChairRampConfig),
    "spot_traffic_cone": (SpotTrafficCone, SpotTrafficConeConfig),
    "spot_crowd_barrier": (SpotCrowdBarrier, SpotCrowdBarrierConfig),
    "spot_barbell": (SpotBarbell, SpotBarbellConfig),
    "spot_table_drag": (SpotTableDrag, SpotTableDragConfig),
    "spot_tire_push": (SpotTirePush, SpotTirePushConfig),
    "spot_chair_push": (SpotChairPush, SpotChairPushConfig),
    "spot_rack_push": (SpotRackPush, SpotRackPushConfig),
    "spot_cone_push": (SpotConePush, SpotConePushConfig),
    "cylinder_push": (CylinderPush, CylinderPushConfig),
    "cartpole": (Cartpole, CartpoleConfig),
    "fr3_pick": (FR3Pick, FR3PickConfig),
    "leap_cube": (LeapCube, LeapCubeConfig),
    "leap_cube_down": (LeapCubeDown, LeapCubeDownConfig),
    "caltech_leap_cube": (CaltechLeapCube, CaltechLeapCubeConfig),
    # "spot_locomotion": (SpotLocomotion, SpotLocomotionConfig),
    "spot_box_push_baseline": (SpotBoxPushBaseline, SpotBoxPushBaselineConfig),
    "spot_tire_push_baseline": (SpotTirePushBaseline, SpotTirePushBaselineConfig),
    "spot_chair_push_baseline": (SpotChairPushBaseline, SpotChairPushBaselineConfig),
    "spot_rack_push_baseline": (SpotRackPushBaseline, SpotRackPushBaselineConfig),
    "spot_cone_push_baseline": (SpotConePushBaseline, SpotConePushBaselineConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    """Registers a new task."""
    _registered_tasks[name] = (task_type, task_config_type)


__all__ = [
    "get_registered_tasks",
    "register_task",
    "Task",
    "TaskConfig",
    "CaltechLeapCube",
    "CaltechLeapCubeConfig",
    "Cartpole",
    "CartpoleConfig",
    "CylinderPush",
    "CylinderPushConfig",
    "FR3Pick",
    "FR3PickConfig",
    "LeapCube",
    "LeapCubeConfig",
    "LeapCubeDown",
    "LeapCubeDownConfig",
    "SpotBase",
    "SpotBaseConfig",
    "SpotBoxPush",
    "SpotBoxPushConfig",
    "SpotCrowdBarrier",
    "SpotCrowdBarrierConfig",
    "SpotYellowChair",
    "SpotYellowChairConfig",
    "SpotYellowChairRamp",
    "SpotYellowChairRampConfig",
    "SpotTrafficCone",
    "SpotTrafficConeConfig",
    "SpotBarbell",
    "SpotBarbellConfig",
    "SpotTableDrag",
    "SpotTableDragConfig",
    "SpotTirePush",
    "SpotTirePushConfig",
    "SpotChairPush",
    "SpotChairPushConfig",
    "SpotRackPush",
    "SpotRackPushConfig",
    "SpotConePush",
    "SpotConePushConfig",
    # "SpotLocomotion",
    # "SpotLocomotionConfig",
    "SpotBoxPushBaseline",
    "SpotBoxPushBaselineConfig",
    "SpotTirePushBaseline",
    "SpotTirePushBaselineConfig",
    "SpotChairPushBaseline",
    "SpotChairPushBaselineConfig",
    "SpotRackPushBaseline",
    "SpotRackPushBaselineConfig",
    "SpotConePushBaseline",
    "SpotConePushBaselineConfig",
]
