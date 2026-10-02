"""Device temperature workflows."""

import typing
from dataclasses import dataclass

from pv_workflows.atmosphere import WindSpeeds
from pv_workflows.common import Temperatures, ValueWithUnits
from pv_workflows.irradiance import Irradiances


@dataclass(frozen=True)
class Uc(ValueWithUnits):
    """Thermal-conduction coefficient value (positive) with units."""

    units: typing.Literal["W m-2 degC-1"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("W m-2 degC-1",):
            raise ValueError("Uc units are not W m-2 degC-1.")

        if self.value <= 0:
            raise ValueError("Uc is non-positive.")


@dataclass(frozen=True)
class Uv(ValueWithUnits):
    """Thermal-convection coefficient value (non-negative) with units."""

    units: typing.Literal["W s m-3 degC-1"]

    def __post_init__(self) -> None:
        """Validation."""

        if self.units not in ("W s m-3 degC-1",):
            raise ValueError("Uv units are not W s m-3 degC-1.")

        if self.value < 0:
            raise ValueError("Uv is negative.")


class SupportsComputeCellTemperatureFromHeatBalance(typing.Protocol):
    """
    Interface for callables that compute cell temperature using heat balance equation.
    """

    def __call__(
        self,
        *,
        dry_bulb_temperature: Temperatures,
        poa_irradiance: Irradiances,
        wind_speed: WindSpeeds,
        thermal_conduction_coefficient: Uc,
        thermal_convection_coefficient: Uv,
        **_: typing.Any,
    ) -> Temperatures:
        """Compute cell temperature using heat balance equation."""


def cell_heat_balance(
    *,
    dry_bulb_temperature: Temperatures,
    poa_irradiance: Irradiances,
    wind_speed: WindSpeeds,
    thermal_conduction_coefficient: Uc,
    thermal_convection_coefficient: Uv,
    **_: typing.Any,
) -> Temperatures:
    """
    Compute cell temperature using heat balance equation.

    A streamlined version of the PVsyst heat-balance model, where the factors involving
    PVsyst's absorption-coefficient parameter (e.g., 0.9) and device-efficiency
    parameter (e.g., 0.15) have been absorbed into the Uc and Uv coefficents.

    Implements pc_workflows.temperature.SupportsComputeCellTemperatureFromHeatBalance.
    """

    return Temperatures(
        array=dry_bulb_temperature.array
        + poa_irradiance.array
        / (
            thermal_conduction_coefficient.value
            + thermal_convection_coefficient.value * wind_speed.array
        ),
        units="degC",
    )
