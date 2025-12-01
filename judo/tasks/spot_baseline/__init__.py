"""Simplified baseline Spot tasks using standard MuJoCo rollouts."""

from .spot_locomotion import SpotLocomotion, SpotLocomotionConfig
from .spot_box import SpotBoxBaseline, SpotBoxBaselineConfig

__all__ = [
    "SpotLocomotion",
    "SpotLocomotionConfig",
    "SpotBoxBaseline",
    "SpotBoxBaselineConfig",
]
