"""Location (in space and time) workflows."""

import typing
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


class SolarPositionWeather(typing.Protocol):
    """Interface for weather needed for solar-position calculations."""

    dry_bulb_temperatures: Temperatures
    pressures: Pressures


class SupportsSolarPosition(typing.Protocol):
    def __call__(
        self,
        *,
        timestamps: Timestamps,
        location: Location,
        weather: SolarPositionWeather | None,
        **_,
    ) -> SolarPosition:
        """
        Compute position of Sun from time, location, and (optionally) weather on Earth.
        """
