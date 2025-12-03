"""Simplified baseline Spot tasks using standard MuJoCo rollouts."""

from .spot_locomotion import SpotLocomotion, SpotLocomotionConfig
from .spot_box_push import SpotBoxPushBaseline, SpotBoxPushBaselineConfig

__all__ = [
    "SpotLocomotion",
    "SpotLocomotionConfig",
    "SpotBoxPushBaseline",
    "SpotBoxPushBaselineConfig",
]
