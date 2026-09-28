"""
Plane-of-array (POA) irradiance workflows.

For front or backside of POA:
  (1) Measured GHI -> Decomposed Components -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (2) Measured GHI, DNI, DHI (at least 2 of 3) -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (3) Measured POA Irradiance -> POA-Irradiance Components -> Effective POA Irradiance

TODO: How to support including heterogeneous output that implementations may produce?
"""

from dataclasses import dataclass
import inspect
import typing

from array_api.latest import Array
import array_api_compat
import numpy
import pandas
import pvlib

from pv_workflows.common import Angles, ArrayWithUnits, Timestamps
from pv_workflows.weather import DhiDniGhi, Irradiances, Pressures, Temperatures


class DecompositionWeather(typing.Protocol):
    """Weather info needed for GHI-decomposition calculations."""

    ghi: Irradiances
    dew_point_temperatures: Temperatures | None = None
    pressures: Pressures | None = None


class DecompositionSolarPosition(typing.Protocol):
    """Solar position info needed for GHI-decomposition calculations."""

    zenith: Angles


class SolarPositionTimestampsWeather(typing.Protocol):
    """Inputs needed for GHI-decomposition calculations."""

    solar_position: DecompositionSolarPosition
    timestamps: Timestamps
    weather: DecompositionWeather


class SupportsDecomposition(typing.Protocol):
    def __call__(
        self,
        *,
        solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    ) -> DhiDniGhi:
        """Compute decomposition of GHI into DHI and DNI."""


_PVLIB_DIRINT_SIG = inspect.signature(pvlib.irradiance.dirint)


def pvlib_dirint_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_DIRINT_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_DIRINT_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """Compute decomposition of GHI into DHI and DNI using pvlib's DIRINT model."""

    if len(solar_position_timestamps_weather.timestamps.sequence) < 2:
        raise ValueError("pvlib's DIRINT requires at least two timesteps.")

    use_delta_kt_prime = True  # Requires all array inputs be pandas.Series.
    times = pandas.to_datetime(solar_position_timestamps_weather.timestamps.sequence)
    ghi = pandas.Series(
        numpy.asarray(solar_position_timestamps_weather.weather.ghi.array),
        index=times,
    )
    solar_zenith = pandas.Series(
        numpy.asarray(
            solar_position_timestamps_weather.solar_position.zeniths.array_deg
        ),
        index=times,
    )

    if solar_position_timestamps_weather.weather.pressures is None:
        pressure = _PVLIB_DIRINT_SIG.parameters["pressure"].default
    else:
        pressure = pandas.Series(
            numpy.asarray(solar_position_timestamps_weather.weather.pressures.array),
            index=times,
        )

    if solar_position_timestamps_weather.weather.dew_point_temperatures is None:
        temp_dew = _PVLIB_DIRINT_SIG.parameters["pressure"].default
    else:
        temp_dew = pandas.Series(
            numpy.asarray(
                solar_position_timestamps_weather.weather.dew_point_temperatures.array_degC
            ),
            index=times,
        )

    result = pvlib.irradiance.dirint(
        ghi,
        solar_zenith,
        times,
        pressure,
        use_delta_kt_prime,
        temp_dew,
        min_cos_zenith,
        max_zenith,
    )

    print("")
    print(result)
    print("")

    xp = array_api_compat.array_namespace(
        solar_position_timestamps_weather.weather.ghi.array
    )
    dni = xp.asarray(result["dni"].to_numpy())
    dhi = solar_position_timestamps_weather.weather.ghi.array - dni * xp.cos(
        solar_position_timestamps_weather.solar_position.zeniths.array_rad
    )

    return DhiDniGhi(
        dhi=Irradiances(array=dhi, units="W m-2"),
        dni=Irradiances(array=dni, units="W m-2"),
        ghi=solar_position_timestamps_weather.weather.ghi,
    )


_PVLIB_DISC_SIG = inspect.signature(pvlib.irradiance.disc)


def pvlib_disc_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_DISC_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_DISC_SIG.parameters["max_zenith"].default,
    max_airmass: float = _PVLIB_DISC_SIG.parameters["max_airmass"].default,
) -> DhiDniGhi:
    """Compute decomposition of GHI into DHI and DNI using pvlib's DISC model."""

    if solar_position_timestamps_weather.weather.pressures is None:
        pressue = None
    else:
        pressue = numpy.asarray(
            solar_position_timestamps_weather.weather.pressures.array
        )

    result = pvlib.irradiance.disc(
        numpy.asarray(solar_position_timestamps_weather.weather.ghi.array),
        numpy.asarray(
            solar_position_timestamps_weather.solar_position.zeniths.array_deg
        ),
        pandas.to_datetime(solar_position_timestamps_weather.timestamps.sequence),
        pressue,
        min_cos_zenith,
        max_zenith,
        max_airmass,
    )

    xp = array_api_compat.array_namespace(
        solar_position_timestamps_weather.weather.ghi.array
    )
    dni = xp.asarray(result["dni"].to_numpy())
    dhi = solar_position_timestamps_weather.weather.ghi.array - dni * xp.cos(
        solar_position_timestamps_weather.solar_position.zeniths.array_rad
    )

    return DhiDniGhi(
        dhi=Irradiances(array=dhi, units="W m-2"),
        dni=Irradiances(array=dni, units="W m-2"),
        ghi=solar_position_timestamps_weather.weather.ghi,
    )


_PVLIB_ERBS_SIG = inspect.signature(pvlib.irradiance.erbs)


def pvlib_erbs_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_ERBS_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_ERBS_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """Compute decomposition of GHI into DHI and DNI using pvlib's Erbs model."""

    result = pvlib.irradiance.erbs(
        numpy.asarray(solar_position_timestamps_weather.weather.ghi.array),
        numpy.asarray(
            solar_position_timestamps_weather.solar_position.zeniths.array_deg
        ),
        pandas.to_datetime(solar_position_timestamps_weather.timestamps.sequence),
        min_cos_zenith,
        max_zenith,
    )

    xp = array_api_compat.array_namespace(
        solar_position_timestamps_weather.weather.ghi.array
    )

    return DhiDniGhi(
        dhi=Irradiances(array=xp.asarray(result["dhi"].to_numpy()), units="W m-2"),
        dni=Irradiances(array=xp.asarray(result["dni"].to_numpy()), units="W m-2"),
        ghi=solar_position_timestamps_weather.weather.ghi,
    )


_PVLIB_ERBS_DRIESSE_SIG = inspect.signature(pvlib.irradiance.erbs_driesse)


def pvlib_erbs_driesse_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_ERBS_DRIESSE_SIG.parameters[
        "min_cos_zenith"
    ].default,
    max_zenith: float = _PVLIB_ERBS_DRIESSE_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs-Driesse model.
    """

    result = pvlib.irradiance.erbs_driesse(
        numpy.asarray(solar_position_timestamps_weather.weather.ghi.array),
        numpy.asarray(
            solar_position_timestamps_weather.solar_position.zeniths.array_deg
        ),
        pandas.to_datetime(solar_position_timestamps_weather.timestamps.sequence),
        None,
        min_cos_zenith,
        max_zenith,
    )

    xp = array_api_compat.array_namespace(
        solar_position_timestamps_weather.weather.ghi.array
    )

    return DhiDniGhi(
        dhi=Irradiances(array=xp.asarray(result["dhi"].to_numpy()), units="W m-2"),
        dni=Irradiances(array=xp.asarray(result["dni"].to_numpy()), units="W m-2"),
        ghi=solar_position_timestamps_weather.weather.ghi,
    )
