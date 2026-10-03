"""
Plane-of-array (POA) irradiance workflows.

FIXME How to extend for backside irradiance for bifacials?
"""

import collections.abc
import math
import typing
from dataclasses import dataclass
from functools import cached_property, reduce

import scipy.interpolate

from pv_workflows import XP
from pv_workflows.atmosphere import (
    Albedos,
    Irradiances,
    IrradiancesNonPhysical,
    RelativeAirMasses,
)
from pv_workflows.common import (
    Angles,
    ArrayUnitless,
    AzimuthAngles,
    Losses,
    TiltAngles,
    Timestamps,
    ZenithAngles,
)


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

    def __call__(
        self,
        *,
        timestamp: Timestamps,
        ghi: Irradiances,
        sun_zenith: ZenithAngles,
        **_: typing.Any,
    ) -> GhiDecomposition:
        """
        Compute decomposition of GHI into DHI and DNI using (true) zenith of the Sun.
        """


@dataclass(frozen=True)
class IrradianceComponents:
    """Interface for result of computing irradance components."""

    direct: Irradiances
    circumsolar: Irradiances
    isotropic: Irradiances
    horizon: IrradiancesNonPhysical
    ground: Irradiances

    @cached_property
    def total(self) -> Irradiances:
        """Compute sum of POA irradiance components."""

        return Irradiances(
            value=sum(
                (
                    self.direct.value,
                    self.circumsolar.value,
                    self.isotropic.value,
                    self.horizon.value,
                    self.ground.value,
                )
            ),
            units="W m-2",
        )


@dataclass(frozen=True)
class PoaIrradianceComponents(IrradianceComponents):
    """Interface for result of computing POA-irradance components."""


class SupportsComputePoaIrradianceComponents(typing.Protocol):
    """Interface for callables that compute POA-irradiance components."""

    def __call__(
        self,
        *,
        poa_tilt: TiltAngles,
        poa_azimuth: AzimuthAngles,
        sun_zenith_apparent: ZenithAngles,
        sun_azimuth: AzimuthAngles,
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


@dataclass(frozen=True)
class IncidentAngles(Angles):
    """Incident-angle array with units."""

    def __post_init__(self) -> None:
        """Validation."""

        super().__post_init__()

        if self.units in ("rad"):
            if XP.any(self.to_deg.value < 0) or XP.any(self.to_deg.value > math.pi):
                raise ValueError(
                    "Incident angles are not all between zero and pi radians, "
                    "inclusive."
                )
        else:
            if XP.any(self.value < 0) or XP.any(self.value > 180):
                raise ValueError(
                    "Incident angles are not all between zero and 180 degrees, "
                    "inclusive."
                )


@dataclass(frozen=True)
class IncidentAngleModifiers(ArrayUnitless):
    """Incident-angle modifier array (unitless between zero and one)."""

    def __post_init__(self):
        """Validation."""

        if XP.any(self.value < 0) or XP.any(self.value > 1):
            raise ValueError(
                "Incident-angle modifiers are not all between zero and one, inclusive."
            )


class SupportsComputeIamFromIncidentAngle(typing.Protocol):
    """
    Interface for callables that compute incident-angle modifier (IAM) from
    plane-of-array (POA) incident angle.
    """

    def __call__(
        self,
        *,
        incident_angle: IncidentAngles,
        **_: typing.Any,
    ) -> IncidentAngleModifiers:
        """
        Compute incident-angle modifier (IAM) from plane of array (POA) incident angle.
        """


def construct_compute_iam_from_incident_angle_pchip(
    *,
    incident_angle: IncidentAngles,
    incident_angle_modifier: IncidentAngleModifiers,
) -> SupportsComputeIamFromIncidentAngle:
    """
    Construct IAM function using PCHIP interpolation with zero extrapolation.

    Returns a callable that implements pv_workflows.irradiance.SupportsComputeIamFromIncidentAngle.
    """

    pchip_interpolator = scipy.interpolate.PchipInterpolator(
        incident_angle.value, incident_angle_modifier.value, extrapolate=False
    )

    def compute_iam(*, incident_angle: IncidentAngles) -> IncidentAngleModifiers:
        """
        The callable to be returned.

        Implements pv_workflows.irradiance.SupportsComputeIam.
        """

        incident_angle_modifier = pchip_interpolator(incident_angle.value)
        incident_angle_modifier[90 < incident_angle.to_deg.value] = 0.0

        return IncidentAngleModifiers(value=incident_angle_modifier)

    return compute_iam


class SupportsComputePoaComponentIamFromIncidentAngle(typing.Protocol):
    """
    Interface for callables that compute plane-of-array (POA) incident-angle modifier
    from POA incident angle, such as for direct or circumsolar (concentrated at center
    of sun disk) components.
    """

    def __call__(
        self,
        *,
        compute_iam: SupportsComputeIamFromIncidentAngle,
        incident_angle: IncidentAngles,
        **_: typing.Any,
    ) -> IncidentAngleModifiers:
        """
        Compute component's plane-of-array (POA) incident-angle modifier (IAM) from
        incident angle.
        """


def compute_poa_iam_direct_circumsolar(
    *,
    compute_iam: SupportsComputeIamFromIncidentAngle,
    incident_angle: IncidentAngles,
    **_: typing.Any,
) -> IncidentAngleModifiers:
    """
    Compute direct or (sun disk concentrated) circumsolar component's plane-of-array
    (POA) incident-angle modifier (IAM) from POA incident angle.

    Implements pv_workflows.irradiance.SupportsComputePoaComponentIamFromIncidentAngle.
    """

    return compute_iam(incident_angle=incident_angle)


class SupportsComputeIamFromTiltAngle(typing.Protocol):
    """
    Interface for callables that compute incident-angle modifier (IAM) from
    plane-of-array (POA) tilt angle.
    """

    def __call__(
        self,
        *,
        compute_iam: SupportsComputeIamFromIncidentAngle,
        tilt_angle: TiltAngles,
        **_: typing.Any,
    ) -> IncidentAngleModifiers:
        """
        Compute incident-angle modifiers (IAMs) from plane of array (POA) incident
        angles.
        """


class SupportsComputePoaComponentIamFromTiltAngle(typing.Protocol):
    """
    Interface for callables that compute plane-of-array (POA) incident-angle modifier
    from POA tilt angle, such as for isotropic, horizon, or ground components.
    """

    def __call__(
        self,
        *,
        compute_iam: SupportsComputeIamFromTiltAngle,
        tilt_angle: TiltAngles,
        **_: typing.Any,
    ) -> IncidentAngleModifiers:
        """
        Compute component's plane-of-array (POA) incident-angle modifier (IAM) from
        POA tilt angle.
        """


@dataclass(frozen=True)
class PoaIamComponents:
    """Incident-angle modifier (IAM) components at plane of array (POA)."""

    direct: IncidentAngleModifiers
    circumsolar: IncidentAngleModifiers
    isotropic: IncidentAngleModifiers
    horizon: IncidentAngleModifiers
    ground: IncidentAngleModifiers


@dataclass(frozen=True)
class EffectivePoaIrradianceComponents(IrradianceComponents):
    """Interface for result of computing effective POA-irradance components."""


class SupportsComputeEffectivePoaIrradiance(typing.Protocol):
    """Interface for callables that compute effective POA irradiance."""

    def __call__(
        self,
        *,
        poa_irradiance_components: PoaIrradianceComponents,
        poa_iam_components: PoaIamComponents,
        losses: collections.abc.Sequence[ArrayUnitless],
        **_: typing.Any,
    ) -> EffectivePoaIrradianceComponents:
        """
        Compute effective POA irradance from incident-angle modifiers and losses.

        Losses can be considered as Isc losses. Note the sign convention for losses.
        """


def compute_effective_poa_irradiance_components(
    *,
    poa_irradiance_components: PoaIrradianceComponents,
    poa_iam_components: PoaIamComponents,
    losses: collections.abc.Sequence[Losses] | None,
) -> EffectivePoaIrradianceComponents:
    """
    Compute effective POA irradance from incident-angle modifiers and losses.

    Losses can be considered as Isc losses. Note the sign convention for losses.

    Implements pv_workflows.irradiance.SupportsComputeEffectivePoaIrradiance.
    """

    if losses:
        derate = 1 - reduce(XP.multiply, (loss.value for loss in losses))
    else:
        derate = 1

    units = "W m-2"
    return EffectivePoaIrradianceComponents(
        direct=Irradiances(
            value=derate
            * poa_iam_components.direct.value
            * poa_irradiance_components.direct.value,
            units=units,
        ),
        circumsolar=Irradiances(
            value=derate
            * poa_iam_components.circumsolar.value
            * poa_irradiance_components.circumsolar.value,
            units=units,
        ),
        isotropic=Irradiances(
            value=derate
            * poa_iam_components.isotropic.value
            * poa_irradiance_components.isotropic.value,
            units=units,
        ),
        horizon=IrradiancesNonPhysical(
            value=derate
            * poa_iam_components.horizon.value
            * poa_irradiance_components.horizon.value,
            units=units,
        ),
        ground=Irradiances(
            value=derate
            * poa_iam_components.ground.value
            * poa_irradiance_components.ground.value,
            units=units,
        ),
    )
