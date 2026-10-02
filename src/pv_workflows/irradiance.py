"""Plane-of-array (POA) irradiance workflows."""

import typing
from dataclasses import dataclass, fields
from functools import cached_property

from pv_workflows.atmosphere import Albedos, Irradiances, RelativeAirMasses
from pv_workflows.common import Angles, Timestamps


class SupportsComputeExtraterrestrialDni(typing.Protocol):
    """Interface for callables that compute extraterrestrial DNI."""

    def __call__(
        self,
        *,
        timestamp: Timestamps,
        solar_constant: Irradiances | None,
        **_: typing.Any,
    ) -> Irradiances:
        """Compute extraterrestrial DNI from time and (optional) solar constant."""


@dataclass(frozen=True)
class GhiDecomposition:
    """Interface for result of computing GHI-decomposition."""

    dhi: Irradiances
    dni: Irradiances


class SupportsDecomposeGhiFromSunZenith(typing.Protocol):
    """
    Interface for callables that compute decompositions using (true) zenith of the Sun.
    """

    # FIXME Input angle should be validated for proper ranges.
    def __call__(
        self,
        *,
        timestamp: Timestamps,
        ghi: Irradiances,
        sun_zenith: Angles,
        **_: typing.Any,
    ) -> GhiDecomposition:
        """
        Compute decomposition of GHI into DHI and DNI using (true) zenith of the Sun.
        """


@dataclass(frozen=True)
class PoaIrradianceComponents:
    """Interface for result of computing POA-irradance components."""

    direct: Irradiances
    circumsolar: Irradiances
    isotropic: Irradiances
    horizon: Irradiances
    ground: Irradiances

    @cached_property
    def total(self) -> Irradiances:
        """Compute sum of POA irradiance components."""

        return Irradiances(
            array=sum(
                self.__getattribute__(field.name).array for field in fields(self)
            ),
            units="W m-2",
        )


class SupportsComputePoaIrradianceComponents(typing.Protocol):
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
        dhi: Irradiances,
        dni: Irradiances,
        extraterrestrial_dni: Irradiances,
        relative_air_mass: RelativeAirMasses,
        ghi: Irradiances,
        albedo: Albedos,
        **_: typing.Any,
    ) -> PoaIrradianceComponents:
        """Compute POA-irradance components."""


# TODO Add SupportsComponentizePoaIrradiance.
