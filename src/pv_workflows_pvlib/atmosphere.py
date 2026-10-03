"""Atmosphere workflows using pvlib."""

import typing

import numpy
import pvlib

from pv_workflows import XP
from pv_workflows.atmosphere import RelativeAirMasses
from pv_workflows.common import ZenithAngles


def compute_relative_air_mass_kastenyoung1989(
    *, sun_zenith_apparent: ZenithAngles, **_: typing.Any
) -> RelativeAirMasses:
    """
    Compute relative air mass at sea level from apparent zenith of Sun using pvlib's
    kastenyoung1989 implementation.

    Implements pv_workflows.atmosphere.SupportsComputeRelativeAirMassFromSunZenithApparent.
    """

    return RelativeAirMasses(
        value=XP.asarray(
            pvlib.atmosphere.get_relative_airmass(
                numpy.asarray(sun_zenith_apparent.to_deg.value), model="kastenyoung1989"
            )
        )
    )


def compute_relative_air_mass_young1994(
    *, sun_zenith: ZenithAngles, **_: typing.Any
) -> RelativeAirMasses:
    """
    Compute relative air mass at sea level from (true) zenith of Sun using pvlib's
    young1994 implementation.

    Implements pv_workflows.atmosphere.SupportsComputeRelativeAirMassFromSunZenith.
    """

    return RelativeAirMasses(
        value=XP.asarray(
            pvlib.atmosphere.get_relative_airmass(
                numpy.asarray(sun_zenith.to_deg.value), model="young1994"
            )
        )
    )
