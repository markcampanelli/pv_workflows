"""Location (in space and time) workflows using pvlib."""

import inspect

import numpy
import pvlib

from pv_workflows import XP
from pv_workflows.common import Angles
from pv_workflows.location import (
    EquationOfTimes,
    Location,
    SolarPosition,
    SolarPositionWeather,
    Timestamps,
)

_PVLIB_GET_SOLAR_POSITION_SIG = inspect.signature(pvlib.solarposition.get_solarposition)


def solar_position(
    *,
    timestamps: Timestamps,
    location: Location,
    weather: SolarPositionWeather | None,
    method: str = _PVLIB_GET_SOLAR_POSITION_SIG.parameters["method"].default,
    **kwargs,
) -> SolarPosition:
    """
    Compute position of Sun from location, time, and weather on Earth.

    Implements pv_workflows.location.SupportsSolarPosition.
    """

    if weather is None:
        temperature = _PVLIB_GET_SOLAR_POSITION_SIG.parameters["temperature"].default
        pressure = _PVLIB_GET_SOLAR_POSITION_SIG.parameters["pressure"].default
    else:
        temperature = numpy.asarray(weather.dry_bulb_temperatures.array)
        pressure = numpy.asarray(weather.pressures.array)

    result = pvlib.solarposition.get_solarposition(
        timestamps.sequence,
        location.latitude.value,
        location.longitude.value,
        location.altitude.value,
        pressure=pressure,
        method=method,
        temperature=temperature,
        **kwargs,
    )

    return SolarPosition(
        azimuths=Angles(array=XP.asarray(result["azimuth"].to_numpy()), units="deg"),
        zeniths=Angles(array=XP.asarray(result["zenith"].to_numpy()), units="deg"),
        apparent_zeniths=Angles(
            array=XP.asarray(result["apparent_zenith"].to_numpy()), units="deg"
        ),
        elevations=Angles(
            array=XP.asarray(result["elevation"].to_numpy()), units="deg"
        ),
        apparent_elevations=Angles(
            array=XP.asarray(result["apparent_elevation"].to_numpy()), units="deg"
        ),
        equation_of_times=EquationOfTimes(
            array=XP.asarray(result["equation_of_time"].to_numpy()), units="min"
        ),
    )
