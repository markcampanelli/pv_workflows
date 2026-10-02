"""Common items."""

import datetime
import math
import typing
import zoneinfo
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import scipy.constants
from array_api.latest import Array

from pv_workflows import XP

_ABS_ZERO_DEGC = scipy.constants.convert_temperature(0, "Kelvin", "Celsius")


@dataclass(frozen=True)
class ValueWithUnits:
    """A value (aka. number or scalar) with units."""

    value: float | int
    units: str


@dataclass(frozen=True)
class ValueUnitless(ValueWithUnits):
    """A value (aka. number or scalar) without units."""

    units: str = typing.Literal[""]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units != "":
            raise ValueError("Units specified for unitless value.")


@dataclass(frozen=True)
class ArrayWithUnits:
    """A Python-Array-API array with units."""

    array: Array
    units: str


@dataclass(frozen=True)
class Timestamps:
    """Timestamp sequence with timezone."""

    sequence: Sequence[datetime.datetime]

    def __post_init__(self) -> None:
        """Validation."""

        if len(self.sequence) == 0:
            raise ValueError("Timestamps sequence is empty.")

        tzinfos = {timestamp.tzinfo for timestamp in self.sequence}

        if len(tzinfos) > 1:
            raise ValueError("Timestamps sequence has multiple timezones.")

        if tzinfos.pop() is None:
            raise ValueError("Timestamps sequence has naive datetimes.")

    @cached_property
    def tzinfo(self) -> zoneinfo.ZoneInfo:
        """Return timezone of timestamp sequence."""

        return self.sequence[0].tzinfo


@dataclass(frozen=True)
class Angle(ValueWithUnits):
    """Angle value with units."""

    units: typing.Literal["rad", "deg", "°"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("rad", "deg", "°"):
            raise ValueError("Angle units are not rad, deg, or °.")

    @cached_property
    def to_deg(self) -> typing.Self:
        """Angle value with degree units."""

        if self.units in ("deg", "°"):
            return self

        return Angle(value=180.0 / math.pi * self.value, units="deg")

    @cached_property
    def to_rad(self) -> typing.Self:
        """Angle value with radian units."""

        if self.units in ("rad",):
            return self

        return Angle(value=math.pi / 180.0 * self.value, units="rad")


@dataclass(frozen=True)
class AngleCosine(ValueUnitless):
    """Unitless cosine of angle value."""

    def __post_init__(self):
        """Validation."""

        if (self.value < -1) or (self.value > 1):
            raise ValueError(
                "AngleCosine is not between negative one and one, inclusive."
            )


@dataclass(frozen=True)
class Angles(ArrayWithUnits):
    """Angle array with units."""

    units: typing.Literal["rad", "deg", "°"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("rad", "deg", "°"):
            raise ValueError("Angles units are not rad, deg, or °.")

    @cached_property
    def to_deg(self) -> typing.Self:
        """Angle array in degrees."""

        if self.units in ("deg", "°"):
            return self

        return Angles(array=180.0 / math.pi * self.array, units="deg")

    @cached_property
    def to_rad(self) -> typing.Self:
        """Angle array in radians."""

        if self.units in ("rad",):
            return self

        return Angles(array=math.pi / 180.0 * self.array, units="rad")


@dataclass(frozen=True)
class Height(ValueWithUnits):
    """Height value (non-negative) with units."""

    units: typing.Literal["m"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("m",):
            raise ValueError("Height units are not m.")

        if self.value < 0:
            raise ValueError("Height is negative.")


@dataclass(frozen=True)
class Temperature(ValueWithUnits):
    """Temperature value with units."""

    units: typing.Literal["K", "degC", "°C"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("K", "degC", "°C"):
            raise ValueError("Temperature units are not K, degC, or °C.")

        if XP.any(self.to_degC.value <= _ABS_ZERO_DEGC):
            raise ValueError("Temperature is not greater than absolute zero.")

    @cached_property
    def to_degC(self) -> typing.Self:
        """Temperature value with degrees Celsius units."""

        if self.units in ("degC", "°C"):
            return self

        return Temperature(
            value=float(
                scipy.constants.convert_temperature(self.value, "Kelvin", "Celsius")
            ),
            units="degC",
        )

    @cached_property
    def to_K(self) -> typing.Self:
        """Temperature value with Kelvin units."""

        if self.units in ("K",):
            return self

        return Temperature(
            value=float(
                scipy.constants.convert_temperature(self.value, "Celsius", "Kelvin")
            ),
            units="K",
        )


@dataclass(frozen=True)
class Temperatures(ArrayWithUnits):
    """Temperature array with units."""

    units: typing.Literal["K", "degC", "°C"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("K", "degC", "°C"):
            raise ValueError("Temperatures units are not K, degC, or °C.")

        if XP.any(self.to_degC.array <= _ABS_ZERO_DEGC):
            raise ValueError("Temperatures not all greater than absolute zero.")

    @cached_property
    def to_degC(self) -> typing.Self:
        """Temperature array with degrees Celsius units."""

        if self.units in ("degC", "°C"):
            return self

        return Temperatures(
            array=XP.asarray(
                scipy.constants.convert_temperature(self.array, "Kelvin", "Celsius")
            ),
            units="degC",
        )

    @cached_property
    def to_K(self) -> typing.Self:
        """Temperature array with Kelvin units."""

        if self.units in ("K",):
            return self

        return Temperatures(
            array=XP.asarray(
                scipy.constants.convert_temperature(self.array, "Celsius", "Kelvin")
            ),
            units="K",
        )


@dataclass(frozen=True)
class Absorption(ValueUnitless):
    """Unitless absorption value."""

    units: typing.Literal[""]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("",):
            raise ValueError("Absorption is not unitless.")

        if (self.value < 0) or (self.value > 1):
            raise ValueError("Absorption is not between zero and one, inclusive.")


@dataclass(frozen=True)
class Efficiency(ValueWithUnits):
    """Efficiency value with (optional) units."""

    units: typing.Literal["", "pc", "%"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("", "pc", "%"):
            raise ValueError("Efficiency is not unitless or units are not pc or %.")

        if (self.to_frac.value < 0) or (self.to_frac.value > 1):
            raise ValueError("Efficiency is not between zero and one, inclusive.")

    # FIXME Conversions.
