"""Plane-of-array (POA) irradiance workflows."""

import typing

from pv_workflows.atmosphere import AirMasses, Albedos, Irradiances
from pv_workflows.common import Angles, Timestamps


class SupportsExtraterrestrialDniResult(typing.TypedDict):
    """Extraterrestrial DNI."""

    extrarerrestrial_dni: Irradiances


class SupportsExtraterrestrialDni(typing.Protocol):
    """Interface for callables that compute extraterrestrial DNI."""

    def __call__(
        self,
        *,
        timestamp: Timestamps,
        solar_constant: Irradiances | None,
        **_: typing.Any,
    ) -> Irradiances:
        """Compute extraterrestrial DNI from time and (optional) solar constant."""


class SupportsDecompositionResult(typing.TypedDict):
    """Interface for result from callables that compute decompositions."""

    ground_dhi: Irradiances
    ground_dni: Irradiances


class SupportsDecompositionZenith(typing.Protocol):
    """
    Interface for callables that compute decompositions using (true) zenith of the Sun.
    """

    # FIXME Input angle should be validated for proper ranges.
    def __call__(
        self,
        *,
        timestamp: Timestamps,
        ground_ghi: Irradiances,
        sun_zenith: Angles,
        **_: typing.Any,
    ) -> SupportsDecompositionResult:
        """
        Compute decomposition of GHI into DHI and DNI using (true) zenith of the Sun.
        """


class SupportsPoaIrradianceComponentsResult(typing.TypedDict):
    """
    Interface for result of computing POA-irradance components from transposition of DHI
    and DNI and ground diffuse from GHI and albedo.
    """

    poa_direct: Irradiances
    poa_circumsolar: Irradiances
    poa_isotropic: Irradiances
    poa_horizon: Irradiances
    poa_ground: Irradiances


class SupportsPoaIrradianceComponents(typing.Protocol):
    """Interface for callables that compute POA-irradiance components."""

    # FIXME Input angles should be validated for proper ranges.
    # FIXME Algorithms may return negative DHI, which is currently invalid irradiance.
    # FIXME How does this generalize for back-side irradiance?
    def __call__(
        self,
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
        """Compute POA-irradance components."""
