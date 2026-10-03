"""Location (in space and time) workflows using pvlib."""

import inspect

import numpy
import pvlib

from pv_workflows import XP
from pv_workflows.atmosphere import Pressures
from pv_workflows.common import (
    AzimuthAngles,
    ElevationAngles,
    Temperatures,
    ZenithAngles,
)
from pv_workflows.location import (
    Altitude,
    Latitude,
    Longitude,
    SunPosition,
    Timestamps,
)

_GET_SOLAR_POSITION_SIG = inspect.signature(pvlib.solarposition.get_solarposition)


def compute_sun_position_nrel_numpy(
    *,
    timestamp: Timestamps,
    latitude: Latitude,
    longitude: Longitude,
    altitude: Altitude,
    pressure: Pressures | None,
    dry_bulb_temperature: Temperatures | None,
    **kwargs,
) -> SunPosition:
    """
    Compute position of Sun from location, time, and weather on Earth.

    Implements pv_workflows.location.SupportsComputeSunPosition.
    """

    if pressure is None:
        pressure = _GET_SOLAR_POSITION_SIG.parameters["pressure"].default
    else:
        pressure = numpy.asarray(pressure.value)

    if dry_bulb_temperature is None:
        temperature = _GET_SOLAR_POSITION_SIG.parameters["temperature"].default
    else:
        temperature = numpy.asarray(dry_bulb_temperature.value)

    result = pvlib.solarposition.get_solarposition(
        timestamp.sequence,
        latitude.value,
        longitude.value,
        altitude.value,
        pressure,
        "nrel_numpy",
        temperature,
        **kwargs,
    )

    return SunPosition(
        azimuth=AzimuthAngles(
            value=XP.asarray(result["azimuth"].to_numpy()), units="deg"
        ),
        zenith=ZenithAngles(value=XP.asarray(result["zenith"].to_numpy()), units="deg"),
        zenith_apparent=ZenithAngles(
            value=XP.asarray(result["apparent_zenith"].to_numpy()), units="deg"
        ),
        elevation=ElevationAngles(
            value=XP.asarray(result["elevation"].to_numpy()), units="deg"
        ),
        elevation_apparent=ElevationAngles(
            value=XP.asarray(result["apparent_elevation"].to_numpy()), units="deg"
        ),
    )
