"""Plane-of-array (POA) irradiance workflows using pvlib."""

import inspect
import typing

import numpy
import pandas
import pvlib

from pv_workflows import XP
from pv_workflows.atmosphere import AirMass, Albedo, Irradiance
from pv_workflows.common import Angle, ArrayWithUnits, Timestamp
from pv_workflows.irradiance import (
    SupportsDecompositionResult,
    SupportsPoaComponentsResult,
)

# _DIRINT_SIG = inspect.signature(pvlib.irradiance.dirint)


# def dirint_decomposition(
#     *,
#     timestamps: Timestamps,
#     weather: WeatherDecomposition,
#     solar_position: SolarPosition,
#     min_cos_zenith: float = _DIRINT_SIG.parameters["min_cos_zenith"].default,
#     max_zenith: float = _DIRINT_SIG.parameters["max_zenith"].default,
# ) -> DhiDniGhi:
#     """
#     Compute decomposition of GHI into DHI and DNI using pvlib's DIRINT model.

#     Implements pv_workflows.irradiance.SupportsDecomposition.
#     """

#     if len(timestamps.sequence) < 2:
#         raise ValueError("pvlib's DIRINT requires at least two timesteps.")

#     # use_delta_kt_prime = True requires all array inputs be pandas.Series.
#     times = pandas.to_datetime(timestamps.sequence)

#     if weather.pressures is None:
#         pressure = _DIRINT_SIG.parameters["pressure"].default
#     else:
#         pressure = pandas.Series(numpy.asarray(weather.pressures.array), index=times)

#     if weather.dew_point_temperatures is None:
#         temp_dew = _DIRINT_SIG.parameters["pressure"].default
#     else:
#         temp_dew = pandas.Series(
#             numpy.asarray(weather.dew_point_temperatures.array_degC), index=times
#         )

#     result = pvlib.irradiance.dirint(
#         pandas.Series(numpy.asarray(weather.ghi.array), index=times),
#         pandas.Series(numpy.asarray(solar_position.zeniths.array_deg), index=times),
#         times,
#         pressure,
#         True,
#         temp_dew,
#         min_cos_zenith,
#         max_zenith,
#     )

#     dni = XP.asarray(result["dni"].to_numpy())
#     dhi = weather.ghi.array - dni * XP.cos(solar_position.zeniths.array_rad)

#     return DhiDniGhi(
#         dhi=Irradiances(array=dhi, units="W m-2"),
#         dni=Irradiances(array=dni, units="W m-2"),
#         ghi=weather.ghi,
#     )


# _DISC_SIG = inspect.signature(pvlib.irradiance.disc)


# def disc_decomposition(
#     *,
#     timestamps: Timestamps,
#     weather: WeatherDecomposition,
#     solar_position: SolarPosition,
#     min_cos_zenith: float = _DISC_SIG.parameters["min_cos_zenith"].default,
#     max_zenith: float = _DISC_SIG.parameters["max_zenith"].default,
#     max_airmass: float = _DISC_SIG.parameters["max_airmass"].default,
# ) -> DhiDniGhi:
#     """
#     Compute decomposition of GHI into DHI and DNI using pvlib's DISC model.

#     Implements pv_workflows.irradiance.SupportsDecomposition.
#     """

#     if weather.pressures is None:
#         pressue = None
#     else:
#         pressue = numpy.asarray(weather.pressures.array)

#     result = pvlib.irradiance.disc(
#         numpy.asarray(weather.ghi.array),
#         numpy.asarray(solar_position.zeniths.array_deg),
#         pandas.to_datetime(timestamps.sequence),
#         pressue,
#         min_cos_zenith,
#         max_zenith,
#         max_airmass,
#     )

#     dni = XP.asarray(result["dni"].to_numpy())
#     dhi = weather.ghi.array - dni * XP.cos(solar_position.zeniths.array_rad)

#     return DhiDniGhi(
#         dhi=Irradiances(array=dhi, units="W m-2"),
#         dni=Irradiances(array=dni, units="W m-2"),
#         ghi=weather.ghi,
#     )


_ERBS_SIG = inspect.signature(pvlib.irradiance.erbs)


def erbs_decomposition(
    *,
    timestamps: Timestamp,
    ground_ghi: Irradiance,
    sun_zenith: Angle,
    min_cos_zenith: float = _ERBS_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _ERBS_SIG.parameters["max_zenith"].default,
    **_: typing.Any,
) -> SupportsDecompositionResult:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs model.

    Implements pv_workflows.irradiance.SupportsDecompositionZenith.
    """

    result = pvlib.irradiance.erbs(
        numpy.asarray(ground_ghi.array),
        numpy.asarray(sun_zenith.array_deg),
        pandas.to_datetime(timestamps.sequence),
        min_cos_zenith,
        max_zenith,
    )

    return SupportsDecompositionResult(
        ground_dhi=Irradiance(
            array=XP.asarray(result["dhi"].to_numpy()), units="W m-2"
        ),
        ground_dni=Irradiance(
            array=XP.asarray(result["dni"].to_numpy()), units="W m-2"
        ),
        kt=ArrayWithUnits(array=XP.asarray(result["kt"].to_numpy()), units=""),
    )


_ERBS_DRIESSE_SIG = inspect.signature(pvlib.irradiance.erbs_driesse)


def erbs_driesse_decomposition(
    *,
    timestamps: Timestamp,
    ground_ghi: Irradiance,
    sun_zenith: Angle,
    min_cos_zenith: float = _ERBS_DRIESSE_SIG.parameters["min_cos_zenith"].default,
    max_zenith: float = _ERBS_DRIESSE_SIG.parameters["max_zenith"].default,
    **_: typing.Any,
) -> SupportsDecompositionResult:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs model.

    Implements pv_workflows.irradiance.SupportsDecompositionZenith.
    """

    result = pvlib.irradiance.erbs_driesse(
        numpy.asarray(ground_ghi.array),
        numpy.asarray(sun_zenith.array_deg),
        pandas.to_datetime(timestamps.sequence),
        None,  # FUTURE Accomodate this optimization where datetime_or_doy is None.
        min_cos_zenith,
        max_zenith,
    )

    return SupportsDecompositionResult(
        ground_dhi=Irradiance(
            array=XP.asarray(result["dhi"].to_numpy()), units="W m-2"
        ),
        ground_dni=Irradiance(
            array=XP.asarray(result["dni"].to_numpy()), units="W m-2"
        ),
        kt=ArrayWithUnits(array=XP.asarray(result["kt"].to_numpy()), units=""),
    )


def perez_poa_components_allsitescomposite1990(
    *,
    poa_tilt: Angle,
    poa_azimuth: Angle,
    sun_zenith_apparent: Angle,
    sun_azimuth: Angle,
    ground_dhi: Irradiance,
    ground_dni: Irradiance,
    extraterrestrial_dni: Irradiance,
    air_mass_relative: AirMass,
    ground_ghi: Irradiance,
    ground_albedo: Albedo,
    **_: typing.Any,
) -> SupportsPoaComponentsResult:
    """
    Compute POA-irradance components using pvlib.irradiance's aoi_projection,
    perez allsitescomposite1990, and get_ground_diffuse models.

    Implements pv_workflows.irradiance.SupportsPoaComponents.
    """

    poa_direct = ground_dni.array * pvlib.irradiance.aoi_projection(
        numpy.asarray(poa_tilt.array_deg),
        numpy.asarray(poa_azimuth.array_deg),
        numpy.asarray(sun_zenith_apparent.array_deg),
        numpy.asarray(sun_azimuth.array_deg),
    )

    result = pvlib.irradiance.perez(
        numpy.asarray(poa_tilt.array_deg),
        numpy.asarray(poa_azimuth.array_deg),
        numpy.asarray(ground_dhi.array),
        numpy.asarray(ground_dni.array),
        numpy.asarray(extraterrestrial_dni.array),
        numpy.asarray(sun_zenith_apparent.array_deg),
        numpy.asarray(sun_azimuth.array_deg),
        numpy.asarray(air_mass_relative.array),
        model="allsitescomposite1990",
        return_components=True,
    )

    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
        numpy.asarray(poa_tilt.array_deg),
        numpy.asarray(ground_ghi.array),
        numpy.asarray(ground_albedo.array),
    )

    units = "W m-2"
    return SupportsPoaComponentsResult(
        poa_direct=Irradiance(array=poa_direct, units=units),
        poa_circumsolar=Irradiance(
            array=XP.asarray(result["poa_circumsolar"]), units=units
        ),
        poa_isotropic=Irradiance(
            array=XP.asarray(result["poa_isotropic"]), units=units
        ),
        poa_horizon=Irradiance(array=XP.asarray(result["poa_horizon"]), units=units),
        poa_ground=Irradiance(array=XP.asarray(poa_ground_diffuse), units=units),
    )


def perez_driesse_poa_components(
    *,
    poa_tilt: Angle,
    poa_azimuth: Angle,
    sun_zenith_apparent: Angle,
    sun_azimuth: Angle,
    ground_dhi: Irradiance,
    ground_dni: Irradiance,
    extraterrestrial_dni: Irradiance,
    air_mass_relative: AirMass,
    ground_ghi: Irradiance,
    ground_albedo: Albedo,
    **_: typing.Any,
) -> SupportsPoaComponentsResult:
    """
    Compute POA-irradance components using pvlib.irradiance's aoi_projection,
    perez-driesse, and get_ground_diffuse models.

    Implements pv_workflows.irradiance.SupportsPoaComponents.
    """

    poa_direct = ground_dni.array * pvlib.irradiance.aoi_projection(
        numpy.asarray(poa_tilt.array_deg),
        numpy.asarray(poa_azimuth.array_deg),
        numpy.asarray(sun_zenith_apparent.array_deg),
        numpy.asarray(sun_azimuth.array_deg),
    )

    result = pvlib.irradiance.perez_driesse(
        numpy.asarray(poa_tilt.array_deg),
        numpy.asarray(poa_azimuth.array_deg),
        numpy.asarray(ground_dhi.array),
        numpy.asarray(ground_dni.array),
        numpy.asarray(extraterrestrial_dni.array),
        numpy.asarray(sun_zenith_apparent.array_deg),
        numpy.asarray(sun_azimuth.array_deg),
        numpy.asarray(air_mass_relative.array),
        return_components=True,
    )

    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
        numpy.asarray(poa_tilt.array_deg),
        numpy.asarray(ground_ghi.array),
        numpy.asarray(ground_albedo.array),
    )

    units = "W m-2"
    return SupportsPoaComponentsResult(
        poa_direct=Irradiance(array=poa_direct, units=units),
        poa_circumsolar=Irradiance(
            array=XP.asarray(result["poa_circumsolar"]), units=units
        ),
        poa_isotropic=Irradiance(
            array=XP.asarray(result["poa_isotropic"]), units=units
        ),
        poa_horizon=Irradiance(array=XP.asarray(result["poa_horizon"]), units=units),
        poa_ground=Irradiance(array=XP.asarray(poa_ground_diffuse), units=units),
    )
