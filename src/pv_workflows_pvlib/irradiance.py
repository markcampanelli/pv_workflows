"""Plane-of-array (POA) irradiance workflows using pvlib."""

import inspect

import array_api_compat
import numpy
import pandas
import pvlib

from pv_workflows.irradiance import SolarPositionTimestampsWeather
from pv_workflows.weather import DhiDniGhi, Irradiances

_PVLIB_DIRINT_SIG = inspect.signature(pvlib.irradiance.dirint)


def dirint_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_DIRINT_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_DIRINT_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's DIRINT model.

    pv_workflows.irradiance.SupportsDecomposition
    """

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


def disc_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_DISC_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_DISC_SIG.parameters["max_zenith"].default,
    max_airmass: float = _PVLIB_DISC_SIG.parameters["max_airmass"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's DISC model.

    pv_workflows.irradiance.SupportsDecomposition
    """

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


def erbs_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_ERBS_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _PVLIB_ERBS_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs model.

    pv_workflows.irradiance.SupportsDecomposition
    """

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


def erbs_driesse_decomposition_from_solar_position_timestamps_weather(
    *,
    solar_position_timestamps_weather: SolarPositionTimestampsWeather,
    min_cos_zenith: float = _PVLIB_ERBS_DRIESSE_SIG.parameters[
        "min_cos_zenith"
    ].default,
    max_zenith: float = _PVLIB_ERBS_DRIESSE_SIG.parameters["max_zenith"].default,
) -> DhiDniGhi:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs-Driesse model.

    pv_workflows.irradiance.SupportsDecomposition
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
