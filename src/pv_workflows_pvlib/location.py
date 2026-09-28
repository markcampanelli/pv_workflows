"""Location (in space and time) workflows using pvlib."""

import inspect

import array_api_compat
import numpy
import pvlib

from pv_workflows.common import Angles
from pv_workflows.location import (
    EquationOfTimes,
    LocationTimestamps,
    LocationTimestampsWeather,
    SolarPosition,
    SolarPositionLocationTimestamps,
    SolarPositionLocationTimestampsWeather,
)

_PVLIB_GET_SOLAR_POSITION_SIG = inspect.signature(pvlib.solarposition.get_solarposition)


def solar_position_from_location_timestamps(
    *,
    location_timestamps: LocationTimestamps,
    method: str = _PVLIB_GET_SOLAR_POSITION_SIG.parameters["method"].default,
    **kwargs,
) -> SolarPositionLocationTimestamps:
    """
    Compute position of Sun from location and time on Earth.

    pv_workflows.location.SupportsSolarPositionFromLocationTimestamps
    """

    result = pvlib.solarposition.get_solarposition(
        location_timestamps.timestamps.sequence,
        location_timestamps.location.latitude.value,
        location_timestamps.location.longitude.value,
        location_timestamps.location.altitude.value,
        _PVLIB_GET_SOLAR_POSITION_SIG.parameters["pressure"].default,
        method,
        _PVLIB_GET_SOLAR_POSITION_SIG.parameters["temperature"].default,
        **kwargs,
    )

    # Note that Python Array API namespace cannot be inferred from arguments.
    return SolarPositionLocationTimestamps(
        solar_position=SolarPosition(
            azimuths=Angles(array=result["azimuth"].to_numpy(), units="deg"),
            zeniths=Angles(array=result["zenith"].to_numpy(), units="deg"),
            apparent_zeniths=Angles(
                array=result["apparent_zenith"].to_numpy(), units="deg"
            ),
            elevations=Angles(array=result["elevation"].to_numpy(), units="deg"),
            apparent_elevations=Angles(
                array=result["apparent_elevation"].to_numpy(), units="deg"
            ),
            equation_of_times=EquationOfTimes(
                array=result["equation_of_time"].to_numpy(), units="min"
            ),
        ),
        location=location_timestamps.location,
        timestamps=location_timestamps.timestamps,
    )


def solar_position_from_location_timestamps_weather(
    *,
    location_timestamps_weather: LocationTimestampsWeather,
    method: str = _PVLIB_GET_SOLAR_POSITION_SIG.parameters["method"].default,
    **kwargs,
) -> SolarPositionLocationTimestampsWeather:
    """
    Compute position of Sun from location, time, and weather on Earth.

    pv_workflows.location.SupportsSolarPositionFromLocationTimestampsWeather
    """

    result = pvlib.solarposition.get_solarposition(
        location_timestamps_weather.timestamps.sequence,
        location_timestamps_weather.location.latitude.value,
        location_timestamps_weather.location.longitude.value,
        location_timestamps_weather.location.altitude.value,
        pressure=numpy.asarray(location_timestamps_weather.weather.pressures.array),
        method=method,
        temperature=numpy.asarray(
            location_timestamps_weather.weather.dry_bulb_temperatures.array
        ),
        **kwargs,
    )

    xp = array_api_compat.array_namespace(
        location_timestamps_weather.weather.pressures.array
    )

    return SolarPositionLocationTimestampsWeather(
        solar_position=SolarPosition(
            azimuths=Angles(
                array=xp.asarray(result["azimuth"].to_numpy()), units="deg"
            ),
            zeniths=Angles(array=xp.asarray(result["zenith"].to_numpy()), units="deg"),
            apparent_zeniths=Angles(
                array=xp.asarray(result["apparent_zenith"].to_numpy()), units="deg"
            ),
            elevations=Angles(
                array=xp.asarray(result["elevation"].to_numpy()), units="deg"
            ),
            apparent_elevations=Angles(
                array=xp.asarray(result["apparent_elevation"].to_numpy()), units="deg"
            ),
            equation_of_times=EquationOfTimes(
                array=xp.asarray(result["equation_of_time"].to_numpy()), units="min"
            ),
        ),
        location=location_timestamps_weather.location,
        timestamps=location_timestamps_weather.timestamps,
        weather=location_timestamps_weather.weather,
    )
