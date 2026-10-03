"""Location (in space and time) workflows."""

import typing
from dataclasses import dataclass

from pv_workflows.common import (
    AzimuthAngles,
    ElevationAngles,
    ScalarWithUnits,
    Timestamps,
    ZenithAngles,
)


@dataclass(frozen=True)
class Latitude(ScalarWithUnits):
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
class Longitude(ScalarWithUnits):
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
class Altitude(ScalarWithUnits):
    """Altitude on Earth relative to mean sea level."""

    units: typing.Literal["m"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("m",):
            raise ValueError("altitude units must be m.")


@dataclass(frozen=True)
class SunPosition:
    """Position of Sun from time and location on Earth."""

    azimuth: AzimuthAngles
    zenith: ZenithAngles
    zenith_apparent: ZenithAngles
    elevation: ElevationAngles
    elevation_apparent: ElevationAngles


class SupportsComputeSunPosition(typing.Protocol):
    def __call__(
        self,
        *,
        timestamp: Timestamps,
        latitude: Latitude,
        longitude: Longitude,
        altitude: Altitude,
        **_: typing.Any,
    ) -> SunPosition:
        """Compute position of Sun from time and location on Earth."""
