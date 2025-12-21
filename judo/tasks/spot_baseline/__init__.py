"""Simplified baseline Spot tasks using standard MuJoCo rollouts."""

from .spot_locomotion import SpotLocomotion, SpotLocomotionConfig
from .spot_box_push import SpotBoxPushBaseline, SpotBoxPushBaselineConfig
from .spot_tire_push import SpotTirePushBaseline, SpotTirePushBaselineConfig
from .spot_chair_push import SpotChairPushBaseline, SpotChairPushBaselineConfig
from .spot_rack_push import SpotRackPushBaseline, SpotRackPushBaselineConfig
from .spot_cone_push import SpotConePushBaseline, SpotConePushBaselineConfig

__all__ = [
    "SpotLocomotion",
    "SpotLocomotionConfig",
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
