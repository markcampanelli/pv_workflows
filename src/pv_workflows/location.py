"""Location (in space and time) workflows."""

import typing
from dataclasses import dataclass

from pv_workflows.common import Angles, Timestamps, ValueWithUnits


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


class SupportsSunPositionResult(typing.TypedDict):
    """Position of Sun from time and location on Earth."""

    sun_azimuth: Angles
    sun_zenith: Angles
    sun_zenith_apparent: Angles
    sun_elevation: Angles
    sun_elevation_apparent: Angles


class SupportsSunPosition(typing.Protocol):
    def __call__(
        self,
        *,
        timestamp: Timestamps,
        latitude: Latitude,
        longitude: Longitude,
        altitude: Altitude,
        **_: typing.Any,
    ) -> SupportsSunPositionResult:
        """Compute position of Sun from time and location on Earth."""
