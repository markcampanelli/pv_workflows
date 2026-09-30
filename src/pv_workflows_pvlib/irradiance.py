"""Plane-of-array (POA) irradiance workflows using pvlib."""

import inspect

import numpy
import pandas
import pvlib

from pv_workflows import XP
from pv_workflows.common import Timestamps
from pv_workflows.irradiance import DecompositionWeather
from pv_workflows.location import SolarPosition
from pv_workflows.weather import DhiDniGhi, Irradiances

_PVLIB_DIRINT_SIG = inspect.signature(pvlib.irradiance.dirint)


def dirint_decomposition(
    *,
    timestamps: Timestamps,
    weather: DecompositionWeather,
    solar_position: SolarPosition,
    min_cos_zenith: float = _PVLIB_DIRINT_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_DIRINT_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's DIRINT model.

    Implements pv_workflows.irradiance.SupportsDecomposition.
    """

    if len(timestamps.sequence) < 2:
        raise ValueError("pvlib's DIRINT requires at least two timesteps.")

    # use_delta_kt_prime = True requires all array inputs be pandas.Series.
    times = pandas.to_datetime(timestamps.sequence)

    if weather.pressures is None:
        pressure = _PVLIB_DIRINT_SIG.parameters["pressure"].default
    else:
        pressure = pandas.Series(numpy.asarray(weather.pressures.array), index=times)

    if weather.dew_point_temperatures is None:
        temp_dew = _PVLIB_DIRINT_SIG.parameters["pressure"].default
    else:
        temp_dew = pandas.Series(
            numpy.asarray(weather.dew_point_temperatures.array_degC), index=times
        )

    result = pvlib.irradiance.dirint(
        pandas.Series(numpy.asarray(weather.ghi.array), index=times),
        pandas.Series(numpy.asarray(solar_position.zeniths.array_deg), index=times),
        times,
        pressure,
        True,
        temp_dew,
        min_cos_zenith,
        max_zenith,
    )

    dni = XP.asarray(result["dni"].to_numpy())
    dhi = weather.ghi.array - dni * XP.cos(solar_position.zeniths.array_rad)

    return DhiDniGhi(
        dhi=Irradiances(array=dhi, units="W m-2"),
        dni=Irradiances(array=dni, units="W m-2"),
        ghi=weather.ghi,
    )


_PVLIB_DISC_SIG = inspect.signature(pvlib.irradiance.disc)


def disc_decomposition(
    *,
    timestamps: Timestamps,
    weather: DecompositionWeather,
    solar_position: SolarPosition,
    min_cos_zenith: float = _PVLIB_DISC_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_DISC_SIG.parameters["max_zenith"].default,
    max_airmass: float = _PVLIB_DISC_SIG.parameters["max_airmass"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's DISC model.

    Implements pv_workflows.irradiance.SupportsDecomposition.
    """

    if weather.pressures is None:
        pressue = None
    else:
        pressue = numpy.asarray(weather.pressures.array)

    result = pvlib.irradiance.disc(
        numpy.asarray(weather.ghi.array),
        numpy.asarray(solar_position.zeniths.array_deg),
        pandas.to_datetime(timestamps.sequence),
        pressue,
        min_cos_zenith,
        max_zenith,
        max_airmass,
    )

    dni = XP.asarray(result["dni"].to_numpy())
    dhi = weather.ghi.array - dni * XP.cos(solar_position.zeniths.array_rad)

    return DhiDniGhi(
        dhi=Irradiances(array=dhi, units="W m-2"),
        dni=Irradiances(array=dni, units="W m-2"),
        ghi=weather.ghi,
    )


_PVLIB_ERBS_SIG = inspect.signature(pvlib.irradiance.erbs)


def erbs_decomposition(
    *,
    timestamps: Timestamps,
    weather: DecompositionWeather,
    solar_position: SolarPosition,
    min_cos_zenith: float = _PVLIB_ERBS_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_ERBS_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs model.

    Implements pv_workflows.irradiance.SupportsDecomposition.
    """

    result = pvlib.irradiance.erbs(
        numpy.asarray(weather.ghi.array),
        numpy.asarray(solar_position.zeniths.array_deg),
        pandas.to_datetime(timestamps.sequence),
        min_cos_zenith,
        max_zenith,
    )

    return DhiDniGhi(
        dhi=Irradiances(array=XP.asarray(result["dhi"].to_numpy()), units="W m-2"),
        dni=Irradiances(array=XP.asarray(result["dni"].to_numpy()), units="W m-2"),
        ghi=weather.ghi,
    )


_PVLIB_ERBS_DRIESSE_SIG = inspect.signature(pvlib.irradiance.erbs_driesse)


def erbs_driesse_decomposition(
    *,
    timestamps: Timestamps,
    weather: DecompositionWeather,
    solar_position: SolarPosition,
    min_cos_zenith: float = _PVLIB_ERBS_DRIESSE_SIG.parameters[
        "min_cos_zenith"
    ].default,
    max_zenith: float = _PVLIB_ERBS_DRIESSE_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs-Driesse model.

    Implements pv_workflows.irradiance.SupportsDecomposition.
    """

    result = pvlib.irradiance.erbs_driesse(
        numpy.asarray(weather.ghi.array),
        numpy.asarray(solar_position.zeniths.array_deg),
        pandas.to_datetime(timestamps.sequence),
        None,  # FUTURE Accomodate this optimization where datetime_or_doy is None.
        min_cos_zenith,
        max_zenith,
    )

    return DhiDniGhi(
        dhi=Irradiances(array=XP.asarray(result["dhi"].to_numpy()), units="W m-2"),
        dni=Irradiances(array=XP.asarray(result["dni"].to_numpy()), units="W m-2"),
        ghi=weather.ghi,
    )
