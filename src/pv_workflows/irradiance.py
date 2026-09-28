"""
Plane-of-array (POA) irradiance workflows.

For front or backside of POA:
  (1) Measured GHI -> Decomposed Components -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (2) Measured GHI, DNI, DHI (at least 2 of 3) -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (3) Measured POA Irradiance -> POA-Irradiance Components -> Effective POA Irradiance

TODO: How to support including heterogeneous output that implementations may produce?
"""

import typing

from pv_workflows.common import Angles, Timestamps
from pv_workflows.weather import DhiDniGhi, Irradiances, Pressures, Temperatures


class DecompositionWeather(typing.Protocol):
    """Weather info needed for GHI-decomposition calculations."""

    ghi: Irradiances
    dew_point_temperatures: Temperatures | None = None
    pressures: Pressures | None = None


class DecompositionSolarPosition(typing.Protocol):
    """Solar position info needed for GHI-decomposition calculations."""

    zenith: Angles


class SolarPositionTimestampsWeather(typing.Protocol):
    """Inputs needed for GHI-decomposition calculations."""

    solar_position: DecompositionSolarPosition
    timestamps: Timestamps
    weather: DecompositionWeather


class SupportsDecomposition(typing.Protocol):
    def __call__(
        self,
        *,
        solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    ) -> DhiDniGhi:
        """Compute decomposition of GHI into DHI and DNI."""
