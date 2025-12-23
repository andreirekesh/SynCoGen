"""Interpolators for flow matching."""

from syncogen.diffusion.interpolation.base import InterpolatorBase
from syncogen.diffusion.interpolation.linear import LinearInterpolator
from syncogen.diffusion.interpolation.geometric import GeometricInterpolator


__all__ = [
    "InterpolatorBase",
    "LinearInterpolator",
    "GeometricInterpolator",
]
