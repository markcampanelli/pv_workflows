"""Location (in space and time) workflows."""

from dataclasses import dataclass
import inspect
import zoneinfo
import typing

from array_api.latest import Array
import array_api_compat
import numpy
import pvlib

from pv_workflows.common import Angles, ArrayWithUnits, Timestamps, ValueWithUnits
from pv_workflows.weather import Pressures, Temperatures


@dataclass(frozen=True)
class Latitude(ValueWithUnits):
    """Latitude on Earth."""

    units: typing.Literal["deg", "°"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.value > 90:
            raise ValueError("Latitude value cannot be greater than 90 deg.")

        if self.value < -90:
            raise ValueError("Latitude value cannot be less than -90 deg.")

        if self.units not in ("deg", "°"):
            raise ValueError("Latitude units must be deg or °.")


@dataclass(frozen=True)
class Longitude(ValueWithUnits):
    """Longitude on Earth."""

    units: typing.Literal["deg", "°"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.value > 180:
            raise ValueError("Longitude value cannot be greater than 180 deg.")

        if self.value < -180:
            raise ValueError("Longitude value cannot be less than -180 deg.")

        if self.units not in ("deg", "°"):
            raise ValueError("Longitude units must be deg or °.")


@dataclass(frozen=True)
class Altitude(ValueWithUnits):
    """Altitude on Earth relative to mean sea level."""

    units: typing.Literal["m"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("m",):
            raise ValueError("altitude units must be m.")


@dataclass(frozen=True)
class Location:
    """Location on Earth."""

    latitude: Latitude
    longitude: Longitude
    altitude: Altitude
    tzinfo: zoneinfo.ZoneInfo


@dataclass(frozen=True)
class LocationTimestamps:
    """Location on Earth specified with a timestamp sequence."""

    location: Location
    timestamps: Timestamps

    def __post_init__(self) -> None:
        """Validation."""

        if self.location.tzinfo != self.timestamps.tzinfo:
            raise ValueError("Timezone of timestamps does not match location.")


@dataclass(frozen=True)
class EquationOfTimes(ArrayWithUnits):
    """An equation of time (in minutes) with units."""

    array: Array
    units: typing.Literal["min"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("min",):
            raise ValueError("equation of time array units must be min.")


@dataclass(frozen=True)
class SolarPosition:
    """Position of Sun."""

    azimuths: Angles
    zeniths: Angles
    apparent_zeniths: Angles
    elevations: Angles
    apparent_elevations: Angles
    equation_of_times: EquationOfTimes


@dataclass(frozen=True)
class SolarPositionLocationTimestamps(LocationTimestamps):
    """Position of Sun with location and time on Earth."""

    solar_position: SolarPosition


class SupportsSolarPositionFromLocationTimestamps(typing.Protocol):
    def __call__(
        self, *, location_timestamps: LocationTimestamps, **_
    ) -> SolarPositionLocationTimestamps:
        """Compute position of Sun from location and time on Earth."""


_PVLIB_GET_SOLAR_POSITION_SIG = inspect.signature(pvlib.solarposition.get_solarposition)


def pvlib_solar_position_from_location_timestamps(
    *,
    location_timestamps: LocationTimestamps,
    method: str = _PVLIB_GET_SOLAR_POSITION_SIG.parameters["method"].default,
    **kwargs,
) -> SolarPositionLocationTimestamps:
    """Compute position of Sun from location and time on Earth."""

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


class SolarPositionWeather(typing.Protocol):
    """Interface for weather needed for solar-position calculations."""

    dry_bulb_temperatures: Temperatures
    pressures: Pressures


@dataclass(frozen=True)
class LocationTimestampsWeather(LocationTimestamps):
    """Location on Earth specified with a timestamp sequence and weather."""

    weather: SolarPositionWeather


@dataclass(frozen=True)
class SolarPositionLocationTimestampsWeather(LocationTimestampsWeather):
    """Position of Sun with location, time, and weather on Earth."""

    solar_position: SolarPosition


class SupportsSolarPositionFromLocationTimestampsWeather(typing.Protocol):
    def __call__(
        self,
        *,
        location_timestamps_weather: LocationTimestampsWeather,
        **_,
    ) -> SolarPositionLocationTimestampsWeather:
        """Compute position of Sun from location, time, and weather on Earth."""


def pvlib_solar_position_from_location_timestamps_weather(
    *,
    location_timestamps_weather: LocationTimestampsWeather,
    method: str = _PVLIB_GET_SOLAR_POSITION_SIG.parameters["method"].default,
    **kwargs,
) -> SolarPositionLocationTimestampsWeather:
    """Compute position of Sun from location, time, and weather on Earth."""

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
