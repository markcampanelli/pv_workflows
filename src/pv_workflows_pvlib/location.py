"""Location (in space and time) workflows using pvlib."""

import inspect

import numpy
import pvlib

from pv_workflows import XP
from pv_workflows.common import Angles, ArrayWithUnits, Temperatures
from pv_workflows.location import (
    Altitude,
    Latitude,
    Longitude,
    SupportsSunPositionResult,
    Timestamps,
)

_GET_SOLAR_POSITION_SIG = inspect.signature(pvlib.solarposition.get_solarposition)


def sun_position_nrel_numpy(
    *,
    timestamp: Timestamps,
    latitude: Latitude,
    longitude: Longitude,
    altitude: Altitude,
    dry_bulb_temperature: Temperatures | None,
    **kwargs,
) -> SupportsSunPositionResult:
    """
    Compute position of Sun from location, time, and weather on Earth.

    Implements pv_workflows.location.SupportsSunPosition.
    """

    pressure = _GET_SOLAR_POSITION_SIG.parameters["pressure"].default

    if dry_bulb_temperature is None:
        temperature = _GET_SOLAR_POSITION_SIG.parameters["temperature"].default
    else:
        temperature = numpy.asarray(dry_bulb_temperature.array)

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

    return SupportsSunPositionResult(
        sun_azimuth=Angles(array=XP.asarray(result["azimuth"].to_numpy()), units="deg"),
        sun_zenith=Angles(array=XP.asarray(result["zenith"].to_numpy()), units="deg"),
        sun_zenith_apparent=Angles(
            array=XP.asarray(result["apparent_zenith"].to_numpy()), units="deg"
        ),
        sun_elevation=Angles(
            array=XP.asarray(result["elevation"].to_numpy()), units="deg"
        ),
        sun_elevation_apparent=Angles(
            array=XP.asarray(result["apparent_elevation"].to_numpy()), units="deg"
        ),
        equation_of_time=ArrayWithUnits(
            array=XP.asarray(result["equation_of_time"].to_numpy()), units="min"
        ),
    )
