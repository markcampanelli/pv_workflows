"""Common items."""

import datetime
import math
import typing
import zoneinfo
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from array_api.latest import Array


@dataclass(frozen=True)
class ValueWithUnits:
    """A value (aka. number or scalar) with units."""

    value: float | int
    units: str


@dataclass(frozen=True)
class ArrayWithUnits:
    """A Python-Array-API array with units."""

    array: Array
    units: str


@dataclass(frozen=True)
class Timestamps:
    """Timestamps with timezone."""

    sequence: Sequence[datetime.datetime]

    def __post_init__(self) -> None:
        """Validation."""

        if len(self.sequence) == 0:
            raise ValueError("Timestamps sequence is empty.")

        tzinfos = {timestamp.tzinfo for timestamp in self.sequence}

        if len(tzinfos) > 1:
            raise ValueError("Multiple timezones specified in timestamp sequence.")

        if tzinfos.pop() is None:
            raise ValueError("Naive datetimes not permitted in timestamp sequence.")

    @cached_property
    def tzinfo(self) -> zoneinfo.ZoneInfo:
        """Return timezone of timestamp sequence."""

        return self.sequence[0].tzinfo


@dataclass(frozen=True)
class Angles(ArrayWithUnits):
    """Angles with units."""

    units: typing.Literal["rad", "deg", "°"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("rad", "deg", "°"):
            raise ValueError("angle units must be rad, deg, or °.")

    @cached_property
    def array_deg(self) -> Array:
        """Angles with degree units."""

        if self.units in ("deg", "°"):
            return self.array

        return 180.0 / math.pi * self.array

    @cached_property
    def array_rad(self) -> Array:
        """Angles with radian units."""

        if self.units in ("rad",):
            return self.array

        return math.pi / 180.0 * self.array


@dataclass(frozen=True)
class Height(ValueWithUnits):
    """Height (non-negative) with units."""

    units: typing.Literal["m"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("m",):
            raise ValueError("Height units must be m.")

        if self.value < 0:
            raise ValueError("Height must not be negative.")
