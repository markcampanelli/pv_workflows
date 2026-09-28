"""Weather workflows."""

from dataclasses import dataclass
from functools import cached_property
import typing

from array_api.latest import Array
import array_api_compat
import scipy.constants

from pv_workflows.common import ArrayWithUnits, Height


@dataclass(frozen=True)
class Pressures(ArrayWithUnits):
    """Pressures with units."""

    units: typing.Literal["Pa"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("Pa"):
            raise ValueError("Pressures units must be Pa.")


@dataclass(frozen=True)
class Temperatures(ArrayWithUnits):
    """Temperatures with units."""

    units: typing.Literal["K", "degC", "°C"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("K", "degC", "°C"):
            raise ValueError("Temperatures units must be K, degC, or °C.")

        xp = array_api_compat.array_namespace(self.array)

        if xp.any(self.array_K <= 0):
            raise ValueError(
                "Temperatures cannot be less than or equal to absolute zero."
            )

    @cached_property
    def array_degC(self) -> Array:
        """Temperatures with degrees Celsius units."""

        if self.units in ("degC", "°C"):
            return self.array

        return scipy.constants.convert_temperature(self.array, "Kelvin", "Celsius")

    @cached_property
    def array_K(self) -> Array:
        """Temperatures with Kelvin units."""

        if self.units in ("K",):
            return self.array

        return scipy.constants.convert_temperature(self.array, "Celsius", "Kelvin")


@dataclass(frozen=True)
class WindSpeeds(ArrayWithUnits):
    """Wind speeds with units and at specified height."""

    units: typing.Literal["m s-1"]
    height: Height

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("m s-1",):
            raise ValueError("WindSpeeds units must be m s-1.")

        xp = array_api_compat.array_namespace(self.array)

        if xp.any(self.array < 0):
            raise ValueError("WindSpeeds must not be negative.")


@dataclass(frozen=True)
class WeatherNoIrradiance:
    """Weather."""

    dew_point_temperatures: Temperatures | None
    dry_bulb_temperatures: Temperatures
    pressures: Pressures
    wind_speeds: WindSpeeds

    # FIXME Validate broadcastability?


@dataclass(frozen=True)
class Irradiances(ArrayWithUnits):
    """Irradiances with units."""

    units: typing.Literal["W m-2"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("W m-2",):
            raise ValueError("Irradiances units must be W m-2.")

        xp = array_api_compat.array_namespace(self.array)

        if xp.any(self.array < 0):
            raise ValueError("Irradiances must not be negative.")


@dataclass(frozen=True)
class WeatherGhi(WeatherNoIrradiance):
    """Weather with GHI irradiance."""

    ghi: Irradiances

    # FIXME Validate broadcastability?


@dataclass(frozen=True)
class DhiDniGhi:
    """
    DHI, DNI, and GHI irradiances with units.

    DHI: Diffuse-horizontal irradiance.
    DNI: Direct-normal irradiance.
    GHI: Global-horizontal irradiance.
    """

    dhi: Irradiances
    dni: Irradiances
    ghi: Irradiances

    # FIXME Validate broadcastability?


@dataclass(frozen=True)
class WeatherDhiDniGhi(WeatherNoIrradiance, DhiDniGhi):
    """Weather with DHI, DNI, and GHI irradiances."""

    # FIXME Validate broadcastability?
