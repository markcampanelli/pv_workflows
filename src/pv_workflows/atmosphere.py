"""Atmosphere workflows."""

import typing
from dataclasses import dataclass
from functools import cached_property

import scipy.constants

from pv_workflows import XP
from pv_workflows.common import Angle, Array, ArrayWithUnits, Height, Timestamp


@dataclass(frozen=True)
class Pressure(ArrayWithUnits):
    """Pressure array with units."""

    units: typing.Literal["Pa"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("Pa"):
            raise ValueError("Pressure units must be Pa.")


@dataclass(frozen=True)
class Temperature(ArrayWithUnits):
    """Temperature array with units."""

    units: typing.Literal["K", "degC", "°C"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("K", "degC", "°C"):
            raise ValueError("Temperature units must be K, degC, or °C.")

        if XP.any(self.array_K <= 0):
            raise ValueError(
                "Temperature cannot be less than or equal to absolute zero."
            )

    @cached_property
    def array_degC(self) -> Array:
        """Temperature array with degrees Celsius units."""

        if self.units in ("degC", "°C"):
            return self.array

        return scipy.constants.convert_temperature(self.array, "Kelvin", "Celsius")

    @cached_property
    def array_K(self) -> Array:
        """Temperature array with Kelvin units."""

        if self.units in ("K",):
            return self.array

        return scipy.constants.convert_temperature(self.array, "Celsius", "Kelvin")


@dataclass(frozen=True)
class WindSpeed(ArrayWithUnits):
    """Wind speed array with units and at specified height."""

    units: typing.Literal["m s-1"]
    height: Height

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("m s-1",):
            raise ValueError("WindSpeed units must be m s-1.")

        if XP.any(self.array < 0):
            raise ValueError("WindSpeed must not be negative.")


@dataclass(frozen=True)
class Irradiance(ArrayWithUnits):
    """Irradiance array with units."""

    units: typing.Literal["W m-2"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("W m-2",):
            raise ValueError("Irradiance units must be W m-2.")

        if XP.any(self.array < 0):
            raise ValueError("Irradiance must not be negative.")


@dataclass(frozen=True)
class AirMass(ArrayWithUnits):
    """Air mass array with units."""

    units: typing.Literal[""]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("",):
            raise ValueError("Air mass must be unitless.")

        if XP.any(self.array < 1):
            raise ValueError("Air mass must not be less than one.")


@dataclass(frozen=True)
class Albedo(ArrayWithUnits):
    """Albedo array with units."""

    units: typing.Literal[""]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("",):
            raise ValueError("Albedo must be unitless.")

        if XP.any(self.array < 0) or XP.any(self.array > 1):
            raise ValueError("Albedo must be between zero and one.")


# FIXME Need to adopt a convention for air mass when zenith is greater than 90 degrees.


class SupportsAirMassRelativeZenithApparent(typing.Protocol):
    """
    Interface for callables that compute relative air mass at sea level from apparent
    zenith of Sun.
    """

    def __call__(self, *, sun_zenith_apparent: Angle, **_: typing.Any) -> AirMass:
        """Compute relative air mass at sea level from apparent zenith of Sun."""


class SupportsAirMassRelativeZenith(typing.Protocol):
    """
    Interface for callables that compute relative air mass at sea level from (true)
    zenith of Sum.
    """

    def __call__(self, *, sun_zenith: Angle, **_: typing.Any) -> Array:
        """Compute relative air mass at sea level from (true) zenith of Sun."""


# FIXME To be defined and implemented.
class SupportsExtraterrestrialDni(typing.Protocol):
    """Interface for callables that compute extraterrestrial DNI."""

    def __call__(self, *, timestamp: Timestamp) -> Irradiance:
        """Compute extraterrestrial DNI from FIXME."""
