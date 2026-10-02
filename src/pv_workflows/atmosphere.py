"""Atmosphere workflows."""

import typing
from dataclasses import dataclass

from pv_workflows import XP
from pv_workflows.common import Angles, ArrayWithUnits, Height, ValueWithUnits


@dataclass(frozen=True)
class Pressures(ArrayWithUnits):
    """Pressure array with units."""

    units: typing.Literal["Pa"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("Pa"):
            raise ValueError("Pressures units are not Pa.")


@dataclass(frozen=True)
class WindSpeeds(ArrayWithUnits):
    """Wind speed array with units and at specified height."""

    units: typing.Literal["m s-1"]
    height: Height

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("m s-1",):
            raise ValueError("WindSpeeds units are not m s-1.")

        if XP.any(self.array < 0):
            raise ValueError("WindSpeeds are not all non-negative.")


@dataclass(frozen=True)
class Irradiance(ValueWithUnits):
    """Irradiance value with units."""

    units: typing.Literal["W m-2"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("W m-2",):
            raise ValueError("Irradiance units are not W m-2.")

        if self.value < 0:
            raise ValueError("Irradiance is not non-negative.")


@dataclass(frozen=True)
class Irradiances(ArrayWithUnits):
    """Irradiance array with units."""

    units: typing.Literal["W m-2"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("W m-2",):
            raise ValueError("Irradiances units must be W m-2.")

        if XP.any(self.array < 0):
            raise ValueError("Irradiances are not all non-negative.")


@dataclass(frozen=True)
class AirMasses(ArrayWithUnits):
    """Air mass array with units."""

    units: typing.Literal[""]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("",):
            raise ValueError("Air masses are not unitless.")

        if XP.any(self.array < 1):
            raise ValueError("Air masses are not all greater than or equal to one.")


@dataclass(frozen=True)
class Albedos(ArrayWithUnits):
    """Albedo array with units."""

    units: typing.Literal[""]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("",):
            raise ValueError("Albedos are not unitless.")

        if XP.any(self.array < 0) or XP.any(self.array > 1):
            raise ValueError("Albedos are not all between zero and one, inclusive.")


# FIXME Need to adopt a convention for air mass when zenith is greater than 90 degrees.


class SupportsRelativeAirMassResult(typing.TypedDict):
    """Relative air mass at sea level."""

    relative_air_mass: AirMasses


class SupportsRelativeAirMassZenith(typing.Protocol):
    """
    Interface for callables that compute relative air mass at sea level from (true)
    zenith of Sum.
    """

    def __call__(
        self, *, sun_zenith: Angles, **_: typing.Any
    ) -> SupportsRelativeAirMassResult:
        """Compute relative air mass at sea level from (true) zenith of Sun."""


class SupportsRelativeAirMassZenithApparent(typing.Protocol):
    """
    Interface for callables that compute relative air mass at sea level from apparent
    zenith of Sun.
    """

    def __call__(
        self, *, sun_zenith_apparent: Angles, **_: typing.Any
    ) -> SupportsRelativeAirMassResult:
        """Compute relative air mass at sea level from apparent zenith of Sun."""
