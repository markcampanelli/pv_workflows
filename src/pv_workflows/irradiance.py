"""Plane-of-array (POA) irradiance workflows."""

import typing

from pv_workflows.atmosphere import AirMass, Albedo, Irradiance
from pv_workflows.common import Angle, Timestamp


class SupportsDecompositionResult(typing.TypedDict):
    """Interface for result from callables that compute decompositions."""

    ground_dhi: Irradiance
    ground_dni: Irradiance


class SupportsDecompositionZenith(typing.Protocol):
    """
    Interface for callables that compute decompositions using (true) zenith of the Sun.
    """

    # FIXME Input angle can be validated for proper ranges.
    def __call__(
        self,
        *,
        timestamps: Timestamp,
        ground_ghi: Irradiance,
        sun_zenith: Angle,
        **_: typing.Any,
    ) -> SupportsDecompositionResult:
        """
        Compute decomposition of GHI into DHI and DNI using (true) zenith of the Sun.
        """


class SupportsPoaComponentsResult(typing.TypedDict):
    """
    Interface for result of computing POA-irradance components from transposition of DHI
    and DNI and ground diffuse from GHI and albedo.
    """

    poa_direct: Irradiance
    poa_circumsolar: Irradiance
    poa_isotropic: Irradiance
    poa_horizon: Irradiance
    poa_ground: Irradiance


class SupportsPoaComponents(typing.Protocol):
    """Interface for callables that compute POA-irradiance components."""

    # FIXME Input angles can be validated for proper ranges.
    # FIXME Algorithms may return negative DHI, which is currently invalid irradiance.
    # FIMXE How does this generalize for back-side irradiance?
    def __call__(
        self,
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
        """Compute POA-irradance components."""
