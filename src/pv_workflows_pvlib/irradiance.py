"""Plane-of-array (POA) irradiance workflows using pvlib."""

import inspect
import typing

import numpy
import pandas
import pvlib

from pv_workflows import XP
from pv_workflows.atmosphere import AirMasses, Albedos, Irradiance, Irradiances
from pv_workflows.common import Angle, AngleCosine, Angles, ArrayWithUnits, Timestamps
from pv_workflows.irradiance import (
    SupportsDecompositionResult,
    SupportsExtraterrestrialDniResult,
    SupportsPoaIrradianceComponentsResult,
)

_GET_EXTRA_RADIATION_SIG = inspect.signature(pvlib.irradiance.get_extra_radiation)


def extraterrestrial_dni_spencer(
    *, timestamp: Timestamps, solar_constant: Irradiance | None, **_: typing.Any
) -> SupportsExtraterrestrialDniResult:
    """
    Compute extraterrestrial DNI from time and (optional) solar constant.

    Implements pv_workflows.irradiance.SupportsExtraterrestrialDni.
    """

    if solar_constant is None:
        solar_constant = _GET_EXTRA_RADIATION_SIG.parameters["solar_constant"].default
    else:
        solar_constant = solar_constant.value

    result = pvlib.irradiance.get_extra_radiation(
        pandas.to_datetime(timestamp.sequence), solar_constant, "spencer"
    )

    return SupportsExtraterrestrialDniResult(
        extraterrestrial_dni=Irradiances(
            array=XP.asarray(result.to_numpy()), units="W m-2"
        )
    )


# _DIRINT_SIG = inspect.signature(pvlib.irradiance.dirint)

# FIXME
# def dirint_decomposition(
#     *,
#     timestamp: Timestamps,
#     weather: WeatherDecomposition,
#     solar_position: SolarPosition,
#     min_cos_zenith: float = _DIRINT_SIG.parameters["min_cos_zenith"].default,
#     max_zenith: float = _DIRINT_SIG.parameters["max_zenith"].default,
# ) -> DhiDniGhi:
#     """
#     Compute decomposition of GHI into DHI and DNI using pvlib's DIRINT model.

#     Implements pv_workflows.irradiance.SupportsDecomposition.
#     """

#     if len(timestamp.sequence) < 2:
#         raise ValueError("pvlib's DIRINT requires at least two timesteps.")

#     # use_delta_kt_prime = True requires all array inputs be pandas.Series.
#     times = pandas.to_datetime(timestamp.sequence)

#     if weather.pressures is None:
#         pressure = _DIRINT_SIG.parameters["pressure"].default
#     else:
#         pressure = pandas.Series(numpy.asarray(weather.pressures.array), index=times)

#     if weather.dew_point_temperatures is None:
#         temp_dew = _DIRINT_SIG.parameters["pressure"].default
#     else:
#         temp_dew = pandas.Series(
#             numpy.asarray(weather.dew_point_temperatures.to_degC.array), index=times
#         )

#     result = pvlib.irradiance.dirint(
#         pandas.Series(numpy.asarray(weather.ghi.array), index=times),
#         pandas.Series(numpy.asarray(solar_position.zeniths.to_deg.array), index=times),
#         times,
#         pressure,
#         True,
#         temp_dew,
#         min_cos_zenith,
#         max_zenith,
#     )

#     dni = XP.asarray(result["dni"].to_numpy())
#     dhi = weather.ghi.array - dni * XP.cos(solar_position.zeniths.to_rad.array)

#     return DhiDniGhi(
#         dhi=Irradiances(array=dhi, units="W m-2"),
#         dni=Irradiances(array=dni, units="W m-2"),
#         ghi=weather.ghi,
#     )


# _DISC_SIG = inspect.signature(pvlib.irradiance.disc)

# FIXME
# def disc_decomposition(
#     *,
#     timestamp: Timestamps,
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
#         numpy.asarray(solar_position.zeniths.to_deg.array),
#         pandas.to_datetime(timestamp.sequence),
#         pressue,
#         min_cos_zenith,
#         max_zenith,
#         max_airmass,
#     )

#     dni = XP.asarray(result["dni"].to_numpy())
#     dhi = weather.ghi.array - dni * XP.cos(solar_position.zeniths.to_rad.array)

#     return DhiDniGhi(
#         dhi=Irradiances(array=dhi, units="W m-2"),
#         dni=Irradiances(array=dni, units="W m-2"),
#         ghi=weather.ghi,
#     )


_ERBS_SIG = inspect.signature(pvlib.irradiance.erbs)


def erbs_decomposition(
    *,
    timestamp: Timestamps,
    ground_ghi: Irradiances,
    sun_zenith: Angles,
    min_cos_zenith: AngleCosine | None = None,
    max_zenith: Angle | None = None,
    **_: typing.Any,
) -> SupportsDecompositionResult:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs model.

    Implements pv_workflows.irradiance.SupportsDecompositionZenith.
    """

    if min_cos_zenith is None:
        min_cos_zenith = _ERBS_SIG.parameters["min_cos_zenith"].default
    else:
        min_cos_zenith = min_cos_zenith.to_deg.value

    if max_zenith is None:
        max_zenith = _ERBS_SIG.parameters["max_zenith"].default
    else:
        max_zenith = max_zenith.to_deg.value

    result = pvlib.irradiance.erbs(
        numpy.asarray(ground_ghi.array),
        numpy.asarray(sun_zenith.to_deg.array),
        pandas.to_datetime(timestamp.sequence),
        min_cos_zenith,
        max_zenith,
    )

    return SupportsDecompositionResult(
        ground_dhi=Irradiances(
            array=XP.asarray(result["dhi"].to_numpy()), units="W m-2"
        ),
        ground_dni=Irradiances(
            array=XP.asarray(result["dni"].to_numpy()), units="W m-2"
        ),
        kt=ArrayWithUnits(array=XP.asarray(result["kt"].to_numpy()), units=""),
    )


_ERBS_DRIESSE_SIG = inspect.signature(pvlib.irradiance.erbs_driesse)


def erbs_driesse_decomposition(
    *,
    timestamp: Timestamps,
    ground_ghi: Irradiances,
    sun_zenith: Angles,
    extraterrestrial_dni: Irradiances | None = None,
    min_cos_zenith: AngleCosine | None = None,
    max_zenith: Angle | None = None,
    **_: typing.Any,
) -> SupportsDecompositionResult:
    """
    Compute decomposition of GHI into DHI and DNI using pvlib's Erbs-Driesse model.

    Implements pv_workflows.irradiance.SupportsDecompositionZenith.
    """

    if extraterrestrial_dni is None:
        datetime_or_doy = pandas.to_datetime(timestamp.sequence)
        dni_extra = None
    else:
        datetime_or_doy = None
        dni_extra = XP.toarray(extraterrestrial_dni.array)

    if min_cos_zenith is None:
        min_cos_zenith = _ERBS_DRIESSE_SIG.parameters["min_cos_zenith"].default
    else:
        min_cos_zenith = min_cos_zenith.to_deg.value

    if max_zenith is None:
        max_zenith = _ERBS_DRIESSE_SIG.parameters["max_zenith"].default
    else:
        max_zenith = max_zenith.to_deg.value

    result = pvlib.irradiance.erbs_driesse(
        numpy.asarray(ground_ghi.array),
        numpy.asarray(sun_zenith.to_deg.array),
        datetime_or_doy,
        dni_extra,
        min_cos_zenith,
        max_zenith,
    )

    return SupportsDecompositionResult(
        ground_dhi=Irradiances(
            array=XP.asarray(result["dhi"].to_numpy()), units="W m-2"
        ),
        ground_dni=Irradiances(
            array=XP.asarray(result["dni"].to_numpy()), units="W m-2"
        ),
        kt=ArrayWithUnits(array=XP.asarray(result["kt"].to_numpy()), units=""),
    )


def perez_poa_irradiance_components_allsitescomposite1990(
    *,
    poa_tilt: Angles,
    poa_azimuth: Angles,
    sun_zenith_apparent: Angles,
    sun_azimuth: Angles,
    ground_dhi: Irradiances,
    ground_dni: Irradiances,
    extraterrestrial_dni: Irradiances,
    relative_air_mass: AirMasses,
    ground_ghi: Irradiances,
    ground_albedo: Albedos,
    **_: typing.Any,
) -> SupportsPoaIrradianceComponentsResult:
    """
    Compute POA-irradance components using pvlib.irradiance's aoi_projection,
    perez allsitescomposite1990, and get_ground_diffuse models.

    Implements pv_workflows.irradiance.SupportsPoaIrradianceComponents.
    """

    poa_direct = ground_dni.array * pvlib.irradiance.aoi_projection(
        numpy.asarray(poa_tilt.to_deg.array),
        numpy.asarray(poa_azimuth.to_deg.array),
        numpy.asarray(sun_zenith_apparent.to_deg.array),
        numpy.asarray(sun_azimuth.to_deg.array),
    )

    result = pvlib.irradiance.perez(
        numpy.asarray(poa_tilt.to_deg.array),
        numpy.asarray(poa_azimuth.to_deg.array),
        numpy.asarray(ground_dhi.array),
        numpy.asarray(ground_dni.array),
        numpy.asarray(extraterrestrial_dni.array),
        numpy.asarray(sun_zenith_apparent.to_deg.array),
        numpy.asarray(sun_azimuth.to_deg.array),
        numpy.asarray(relative_air_mass.array),
        model="allsitescomposite1990",
        return_components=True,
    )

    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
        numpy.asarray(poa_tilt.to_deg.array),
        numpy.asarray(ground_ghi.array),
        numpy.asarray(ground_albedo.array),
    )

    units = "W m-2"
    return SupportsPoaIrradianceComponentsResult(
        poa_direct=Irradiances(array=poa_direct, units=units),
        poa_circumsolar=Irradiances(
            array=XP.asarray(result["poa_circumsolar"]), units=units
        ),
        poa_isotropic=Irradiances(
            array=XP.asarray(result["poa_isotropic"]), units=units
        ),
        poa_horizon=Irradiances(array=XP.asarray(result["poa_horizon"]), units=units),
        poa_ground=Irradiances(array=XP.asarray(poa_ground_diffuse), units=units),
    )


def perez_driesse_poa_irradiance_components(
    *,
    poa_tilt: Angles,
    poa_azimuth: Angles,
    sun_zenith_apparent: Angles,
    sun_azimuth: Angles,
    ground_dhi: Irradiances,
    ground_dni: Irradiances,
    extraterrestrial_dni: Irradiances,
    relative_air_mass: AirMasses,
    ground_ghi: Irradiances,
    ground_albedo: Albedos,
    **_: typing.Any,
) -> SupportsPoaIrradianceComponentsResult:
    """
    Compute POA-irradance components using pvlib.irradiance's aoi_projection,
    perez-driesse, and get_ground_diffuse models.

    Implements pv_workflows.irradiance.SupportsPoaIrradianceComponents.
    """

    poa_direct = ground_dni.array * pvlib.irradiance.aoi_projection(
        numpy.asarray(poa_tilt.to_deg.array),
        numpy.asarray(poa_azimuth.to_deg.array),
        numpy.asarray(sun_zenith_apparent.to_deg.array),
        numpy.asarray(sun_azimuth.to_deg.array),
    )

    result = pvlib.irradiance.perez_driesse(
        numpy.asarray(poa_tilt.to_deg.array),
        numpy.asarray(poa_azimuth.to_deg.array),
        numpy.asarray(ground_dhi.array),
        numpy.asarray(ground_dni.array),
        numpy.asarray(extraterrestrial_dni.array),
        numpy.asarray(sun_zenith_apparent.to_deg.array),
        numpy.asarray(sun_azimuth.to_deg.array),
        numpy.asarray(relative_air_mass.array),
        return_components=True,
    )

    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
        numpy.asarray(poa_tilt.to_deg.array),
        numpy.asarray(ground_ghi.array),
        numpy.asarray(ground_albedo.array),
    )

    units = "W m-2"
    return SupportsPoaIrradianceComponentsResult(
        poa_direct=Irradiances(array=poa_direct, units=units),
        poa_circumsolar=Irradiances(
            array=XP.asarray(result["poa_circumsolar"]), units=units
        ),
        poa_isotropic=Irradiances(
            array=XP.asarray(result["poa_isotropic"]), units=units
        ),
        poa_horizon=Irradiances(array=XP.asarray(result["poa_horizon"]), units=units),
        poa_ground=Irradiances(array=XP.asarray(poa_ground_diffuse), units=units),
    )
