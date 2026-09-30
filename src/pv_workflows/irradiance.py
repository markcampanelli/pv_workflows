"""
Plane-of-array (POA) irradiance workflows.

For front or backside of POA:
  (1) Measured GHI -> Decomposed Components -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (2) Measured GHI, DNI, DHI (at least 2 of 3) -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (3) Measured POA Irradiance -> POA-Irradiance Components -> Effective POA Irradiance

TODO: How to support including heterogeneous output that implementations may produce?
"""

import typing
from dataclasses import dataclass

from pv_workflows.common import Angles, Timestamps
from pv_workflows.location import SolarPosition
from pv_workflows.weather import DhiDniGhi, Irradiances, Pressures, Temperatures


class SupportsExtraterrestrialRadiation(typing.Protocol):
    """Interface for callables that compute extraterrestrial radiation."""

    def __call__(
        self,
        *,
        timestamps: Timestamps,
    ) -> Irradiances:
        """Compute extraterrestrial DNI from FIXME."""


class DecompositionWeather(typing.Protocol):
    """Weather info needed for GHI-decomposition calculations."""

    ghi: Irradiances
    dew_point_temperatures: Temperatures | None = None
    pressures: Pressures | None = None


class DecompositionSolarPosition(typing.Protocol):
    """Solar position info needed for GHI-decomposition calculations."""

    zenith: Angles


class SupportsDecomposition(typing.Protocol):
    """Interface for callables that compute decompositions."""

    def __call__(
        self,
        *,
        timestamps: Timestamps,
        weather: DecompositionWeather,
        solar_position: DecompositionSolarPosition,
    ) -> DhiDniGhi:
        """Compute decomposition of GHI into DHI and DNI."""


@dataclass(frozen=True)
class OrientationsPOA:
    """POA irradiances with units."""

    surface_tilt: Angles
    surface_azimuth: Angles

    # FIXME Validate broadcastability?


@dataclass(frozen=True)
class IrradiancesDiffuse:
    """POA diffues irradiances with units."""

    circumsolar: Irradiances
    isotropic: Irradiances
    horizon: Irradiances
    ground: Irradiances

    # FIXME Validate broadcastability?


@dataclass(frozen=True)
class IrradiancesPOA:
    """POA irradiances with units."""

    orientation: OrientationsPOA
    direct: Irradiances
    diffuse: IrradiancesDiffuse

    # FIXME Validate broadcastability?


class SupportsTransposition(typing.Protocol):
    """Interface for callables that compute transpositions."""

    def __call__(
        self,
        *,
        timestamps: Timestamps,
        solar_position: SolarPosition,
        dhi_dni_ghi: DhiDniGhi,
        **kwargs,
    ) -> IrradiancesPOA:
        """Compute transposition of two of three of GHI, DHI, and DNI into POA."""


# import pvlib
# pvlib.irradiance.perez(
#     surface_tilt, surface_azimuth, dhi, dni, dni_extra,
#     solar_zenith, solar_azimuth, airmass,
#     model='allsitescomposite1990', return_components=False,
# )
