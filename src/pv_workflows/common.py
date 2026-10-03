"""Common items."""

import collections.abc
import datetime
import math
import typing
import zoneinfo
from dataclasses import dataclass, field
from functools import cached_property

import scipy.constants
from array_api.latest import Array

from pv_workflows import XP

_ABS_ZERO_DEGC = scipy.constants.convert_temperature(0, "Kelvin", "Celsius")


@dataclass(frozen=True)
class ScalarWithUnits:
    """A numeric scalar with units."""

    value: float | int
    units: str

    def __post_init__(self) -> None:
        """Validation."""

        if self.units == "":
            raise ValueError("Empty units specified.")


@dataclass(frozen=True)
class ScalarPercentage:
    """A numeric value with percentage units."""

    value: float | int
    units: str = field(default="%", init=False)

    @cached_property
    def to_fraction(self) -> ScalarUnitless:
        """Percentage value to unitless value."""

        return ScalarUnitless(value=self.value / 100)


@dataclass(frozen=True)
class ScalarUnitless:
    """A numeric value without units."""

    value: float | int
    units: str = field(default="", init=False)

    @cached_property
    def to_percent(self) -> ScalarPercentage:
        """Unitless value to percentage value."""

        return ScalarPercentage(value=100 * self.value)


@dataclass(frozen=True)
class ArrayWithUnits:
    """A numeric Python-Array-API array with units."""

    value: Array
    units: str

    def __post_init__(self) -> None:
        """Validation."""

        if self.units == "":
            raise ValueError("Empty units specified.")


@dataclass(frozen=True)
class ArrayPercentage:
    """A numeric Python-Array-API array with units."""

    value: Array
    units: str = field(default="%", init=False)

    @cached_property
    def to_fraction(self) -> ArrayUnitless:
        """Percentage value to unitless value."""

        return ArrayUnitless(value=self.value / 100)


@dataclass(frozen=True)
class ArrayUnitless:
    """A numeric Python-Array-API array without units."""

    value: Array
    units: str = field(default="", init=False)

    @cached_property
    def to_percent(self) -> ArrayPercentage:
        """Unitless array to percentage value."""

        return ArrayPercentage(value=100 * self.value)


@dataclass(frozen=True)
class Timestamps:
    """Timestamp sequence with timezone."""

    sequence: collections.abc.Sequence[datetime.datetime]

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
class Angle(ScalarWithUnits):
    """Angle value with units."""

    units: typing.Literal["rad", "deg", "°"]

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units not in ("rad", "deg", "°"):
            raise ValueError("Angle units are not rad, deg, or °.")

    # TODO Add a modulo 360 deg function.

    @cached_property
    def to_deg(self) -> typing.Self:
        """Angle value with degree units."""

        if self.units in ("deg", "°"):
            return self

        return self.__class__(value=180.0 / math.pi * self.value, units="deg")

    @cached_property
    def to_rad(self) -> typing.Self:
        """Angle value with radian units."""

        if self.units in ("rad",):
            return self

        return self.__class__(value=math.pi / 180.0 * self.value, units="rad")


@dataclass(frozen=True)
class Cosine(ScalarUnitless):
    """Unitless cosine of angle value."""

    def __post_init__(self):
        """Validation."""

        super().__post_init__()

        if (self.value < -1) or (self.value > 1):
            raise ValueError("Cosine is not between negative one and one, inclusive.")


@dataclass(frozen=True)
class Angles(ArrayWithUnits):
    """Angle array with units."""

    units: typing.Literal["rad", "deg", "°"]

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units not in ("rad", "deg", "°"):
            raise ValueError("Angles units are not rad, deg, or °.")

    # TODO Add a modulo 360 deg function.

    @cached_property
    def to_deg(self) -> typing.Self:
        """Angle array in degrees."""

        if self.units in ("deg", "°"):
            return self

        return self.__class__(value=180.0 / math.pi * self.value, units="deg")

    @cached_property
    def to_rad(self) -> typing.Self:
        """Angle array in radians."""

        if self.units in ("rad",):
            return self

        return self.__class__(value=math.pi / 180.0 * self.value, units="rad")


@dataclass(frozen=True)
class AzimuthAngles(Angles):
    """Azimuth-angle array with units."""

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units in ("rad"):
            if XP.any(self.to_deg.value < 0) or XP.any(
                self.to_deg.value >= 2 * math.pi
            ):
                raise ValueError(
                    "Azimuth angles are not all between zero, inclusive, and 2*pi "
                    "radians, exclusive."
                )
        else:
            if XP.any(self.value < 0) or XP.any(self.value >= 360):
                raise ValueError(
                    "Azimuth angles are not all between zero, inclusive, and 360 "
                    "degrees, exclusive."
                )


@dataclass(frozen=True)
class ZenithAngles(Angles):
    """Zenith-angle array with units."""

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units in ("rad"):
            if XP.any(self.to_deg.value < 0) or XP.any(self.to_deg.value > math.pi):
                raise ValueError(
                    "Zenith angles are not all between zero and pi radians, inclusive."
                )
        else:
            if XP.any(self.value < 0) or XP.any(self.value > 180):
                raise ValueError(
                    "Zenith angles are not all between zero and 180 degrees, inclusive."
                )


@dataclass(frozen=True)
class ElevationAngles(Angles):
    """Elevation-angle array with units."""

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units in ("rad"):
            if XP.any(self.to_deg.value < -math.pi) or XP.any(
                self.to_deg.value > math.pi
            ):
                raise ValueError(
                    "Elevation angles are not all between zero and pi radians, "
                    "inclusive."
                )
        else:
            if XP.any(self.value < -90) or XP.any(self.value > 90):
                raise ValueError(
                    "Elevation angles are not all between zero and 180 degrees, "
                    "inclusive."
                )


@dataclass(frozen=True)
class TiltAngles(Angles):
    """Tilt-angle array with units."""

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units in ("rad"):
            if XP.any(self.to_deg.value < 0) or XP.any(self.to_deg.value > math.pi):
                raise ValueError(
                    "Tilt angles are not all between zero and pi radians, inclusive."
                )
        else:
            if XP.any(self.value < 0) or XP.any(self.value > 180):
                raise ValueError(
                    "Tilt angles are not all between zero and 180 degrees, inclusive."
                )


@dataclass(frozen=True)
class Cosines(ArrayUnitless):
    """Unitless cosines of angles array."""

    def __post_init__(self):
        """Validation."""

        if XP.any(self.value < -1) or XP.any(self.value > 1):
            raise ValueError(
                "Cosines are not all between negative one and one, inclusive."
            )


@dataclass(frozen=True)
class Height(ScalarWithUnits):
    """Height value (non-negative) with units."""

    units: typing.Literal["m"]

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units not in ("m",):
            raise ValueError("Height units are not m.")

        if self.value < 0:
            raise ValueError("Height is negative.")


@dataclass(frozen=True)
class Temperature(ScalarWithUnits):
    """Temperature value with units."""

    units: typing.Literal["K", "degC", "°C"]

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units not in ("K", "degC", "°C"):
            raise ValueError("Temperature units are not K, degC, or °C.")

        if XP.any(self.to_degC.value <= _ABS_ZERO_DEGC):
            raise ValueError("Temperature is not greater than absolute zero.")

    @cached_property
    def to_degC(self) -> typing.Self:
        """Temperature value with degrees Celsius units."""

        if self.units in ("degC", "°C"):
            return self

        return self.__class__(
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

        return self.__class__(
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

        super().__post_init__()

        if self.units not in ("K", "degC", "°C"):
            raise ValueError("Temperature units are not K, degC, or °C.")

        if XP.any(self.to_degC.value <= _ABS_ZERO_DEGC):
            raise ValueError("Temperatures not all greater than absolute zero.")

    @cached_property
    def to_degC(self) -> typing.Self:
        """Temperature array with degrees Celsius units."""

        if self.units in ("degC", "°C"):
            return self

        return self.__class__(
            value=XP.asarray(
                scipy.constants.convert_temperature(self.value, "Kelvin", "Celsius")
            ),
            units="degC",
        )

    @cached_property
    def to_K(self) -> typing.Self:
        """Temperature array with Kelvin units."""

        if self.units in ("K",):
            return self

        return self.__class__(
            value=XP.asarray(
                scipy.constants.convert_temperature(self.value, "Celsius", "Kelvin")
            ),
            units="K",
        )


@dataclass(frozen=True)
class Absorption(ScalarUnitless):
    """Absorption value (unitless, not percentage)."""

    def __post_init__(self):
        """Validation."""

        super().__post_init__()

        if (self.value < 0) or (self.value > 1):
            raise ValueError("Absorption is not between zero and one, inclusive.")


@dataclass(frozen=True)
class Efficiency(ScalarUnitless):
    """Efficiency value (unitless, not percentage)."""

    def __post_init__(self):
        """Validation."""

        super().__post_init__()

        if (self.value < 0) or (self.value > 1):
            raise ValueError("Efficiency is not between zero and one, inclusive.")


@dataclass(frozen=True)
class Losses(ArrayUnitless):
    """A bounded loss array (unitless)."""

    def __post_init__(self) -> None:
        """Validation."""

        if XP.any(self.value > 1):
            raise ValueError("Losses are not all less than or equal to one.")
