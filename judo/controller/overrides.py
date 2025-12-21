# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.config import set_config_overrides
from judo.controller.controller import ControllerConfig

def set_default_cylinder_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cylinder push task."""
    set_config_overrides(
        "cylinder_push",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "zero",
        },
    )


def set_default_cartpole_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cartpole task."""
    set_config_overrides(
        "cartpole",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "zero",
        },
    )


def set_default_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube task."""
    set_config_overrides(
        "leap_cube",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_leap_cube_down_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube down task."""
    set_config_overrides(
        "leap_cube_down",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_caltech_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the caltech leap cube task."""
    set_config_overrides(
        "caltech_leap_cube",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_fr3_pick_overrides() -> None:
    """Sets the default task-specific controller config overrides for the fr3 pick task."""
    set_config_overrides(
        "fr3_pick",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "linear",
            "max_num_traces": 3,
            "control_freq": 20.0,
        },
    )


def set_default_spot_yellow_chair_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot locomotion task."""
    set_config_overrides(
        "spot_yellow_chair",
        ControllerConfig,
        {
            "horizon": 3.0,
        },
    )

def set_default_spot_yellow_chair_ramp_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot yellow chair ramp task."""
    set_config_overrides(
        "spot_yellow_chair_ramp",
        ControllerConfig,
        {
            "horizon": 3.0,
        },
    )

def set_default_spot_traffic_cone_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot traffic cone task."""
    set_config_overrides(
        "spot_traffic_cone",
        ControllerConfig,
        {
            "horizon": 2.0,
        },
    )

def set_default_spot_box_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot box task."""
    set_config_overrides(
        "spot_box",
        ControllerConfig,
        {
            "horizon": 1.0,
            "control_freq": 20.0,
        },
    )

def set_default_spot_locomotion_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot locomotion task."""
    set_config_overrides(
        "spot_locomotion",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "control_freq": 50.0,
        },
    )

def set_default_spot_box_push_baseline_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot box baseline task."""
    set_config_overrides(
        "spot_box_push_baseline",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "control_freq": 50.0,
        },
    )

def set_default_spot_chair_push_baseline_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot chair baseline task."""
    set_config_overrides(
        "spot_chair_push_baseline",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "control_freq": 50.0,
        },
    )

def set_default_spot_tire_push_baseline_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot tire baseline task."""
    set_config_overrides(
        "spot_tire_push_baseline",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "control_freq": 50.0,
        },
    )

def set_default_spot_rack_push_baseline_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot rack baseline task."""
    set_config_overrides(
        "spot_rack_push_baseline",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "control_freq": 50.0,
        },
    )

def set_default_spot_cone_push_baseline_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot cone baseline task."""
    set_config_overrides(
        "spot_cone_push_baseline",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "control_freq": 50.0,
        },
    )

def set_default_spot_rugged_box_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the spot rugged box push task."""
    set_config_overrides(
        "spot_rugged_box_push",
        ControllerConfig,
        {
            "horizon": 2.0,
            "spline_order": "cubic",
        },
    )