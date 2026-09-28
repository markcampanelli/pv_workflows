"""Location (in space and time) workflows."""

import typing
import zoneinfo
from dataclasses import dataclass

from array_api.latest import Array

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
