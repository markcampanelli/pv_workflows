"""Atmosphere workflows."""

import typing
from dataclasses import dataclass

from pv_workflows import XP
from pv_workflows.common import (
    ArrayUnitless,
    ArrayWithUnits,
    Height,
    ScalarWithUnits,
    ZenithAngles,
)


@dataclass(frozen=True)
class Pressures(ArrayWithUnits):
    """Pressure array with units."""

    units: typing.Literal["Pa"]

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units not in ("Pa"):
            raise ValueError("Pressure units are not Pa.")


@dataclass(frozen=True)
class WindSpeeds(ArrayWithUnits):
    """Wind speed array with units and at specified height."""

    units: typing.Literal["m s-1"]
    height: Height

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units not in ("m s-1",):
            raise ValueError("Wind speed units are not m s-1.")

        if XP.any(self.value < 0):
            raise ValueError("Wind speeds are not all non-negative.")


@dataclass(frozen=True)
class Irradiance(ScalarWithUnits):
    """Irradiance value with units."""

    units: typing.Literal["W m-2"]

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

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

        super().__post_init__()

        if self.units not in ("W m-2",):
            raise ValueError("Irradiance units must be W m-2.")

        if XP.any(self.value < 0):
            raise ValueError("Irradiances are not all non-negative.")


@dataclass(frozen=True)
class Albedos(ArrayUnitless):
    """Albedo array (unitless, not percentage)."""

    def __post_init__(self) -> None:
        """Validation."""

        if XP.any(self.value < 0) or XP.any(self.value > 1):
            raise ValueError("Albedos are not all between zero and one, inclusive.")


@dataclass(frozen=True)
class AbsoluteAirMasses(ArrayUnitless):
    """Absolute air-mass array (unitless)."""

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if XP.any(self.value <= 0):
            raise ValueError("Absolute air masses are not all greater than zero.")


@dataclass(frozen=True)
class RelativeAirMasses(ArrayUnitless):
    """Relative air-mass array (unitless)."""

    def __post_init__(self) -> None:
        """Validation."""

        if XP.any(self.value < 1):
            raise ValueError(
                "Relative air masses are not all greater than or equal to one."
            )


# FIXME Need to adopt a convention for air mass when zenith is greater than 90 degrees.


class SupportsComputeRelativeAirMassFromSunZenith(typing.Protocol):
    """
    Interface for callables that compute relative air mass from (true) zenith of Sum.
    """

    def __call__(
        self, *, sun_zenith: ZenithAngles, **_: typing.Any
    ) -> RelativeAirMasses:
        """Compute relative air mass (at sea level) from (true) zenith of Sun."""


class SupportsComputeRelativeAirMassFromSunZenithApparent(typing.Protocol):
    """
    Interface for callables that compute relative air mass from apparent zenith of Sun.
    """

    def __call__(
        self, *, sun_zenith_apparent: ZenithAngles, **_: typing.Any
    ) -> RelativeAirMasses:
        """Compute relative air mass (at sea level) from apparent zenith of Sun."""
