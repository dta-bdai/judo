# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo import MODEL_PATH
from judo.controller.controller import Controller, ControllerConfig
from judo.controller.overrides import (
    set_default_caltech_leap_cube_overrides,
    set_default_cartpole_overrides,
    set_default_cylinder_push_overrides,
    set_default_fr3_pick_overrides,
    set_default_leap_cube_down_overrides,
    set_default_leap_cube_overrides,
    set_default_spot_yellow_chair_overrides,
    set_default_spot_yellow_chair_ramp_overrides,
    set_default_spot_traffic_cone_overrides,
    set_default_spot_box_overrides,
    set_default_spot_locomotion_overrides,
    set_default_spot_box_push_baseline_overrides,
    set_default_spot_chair_push_baseline_overrides,
    set_default_spot_tire_push_baseline_overrides,
    set_default_spot_rack_push_baseline_overrides,
    set_default_spot_cone_push_baseline_overrides,
    set_default_spot_rugged_box_push_overrides,
)
from judo.utils.assets import download_and_extract_meshes

download_and_extract_meshes(
    extract_root=str(MODEL_PATH),
    repo="bdaiinstitute/judo",
    asset_name="meshes.zip",
)

set_default_caltech_leap_cube_overrides()
set_default_cartpole_overrides()
set_default_cylinder_push_overrides()
set_default_fr3_pick_overrides()
set_default_leap_cube_overrides()
set_default_leap_cube_down_overrides()
set_default_spot_yellow_chair_overrides()
set_default_spot_yellow_chair_ramp_overrides()
set_default_spot_traffic_cone_overrides()
set_default_spot_box_overrides()
set_default_spot_locomotion_overrides()
set_default_spot_box_push_baseline_overrides()
set_default_spot_chair_push_baseline_overrides()
set_default_spot_tire_push_baseline_overrides()
set_default_spot_rack_push_baseline_overrides()
set_default_spot_cone_push_baseline_overrides()
set_default_spot_rugged_box_push_overrides()

__all__ = [
    "Controller",
    "ControllerConfig",
]
