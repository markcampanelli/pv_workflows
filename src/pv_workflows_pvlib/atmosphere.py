"""Atmosphere workflows using pvlib."""

import typing

import numpy
import pvlib

from pv_workflows import XP
from pv_workflows.atmosphere import AirMass
from pv_workflows.common import Angle


def air_mass_relative_kastenyoung1989(
    *, sun_zenith_apparent: Angle, **_: typing.Any
) -> AirMass:
    """
    Compute relative air mass at sea level from apparent zenith of Sun using pvlib's
    kastenyoung1989 implementation.

    Implements pv_workflows.atmosphere.SupportsAirMassRelativeZenithApparent.
    """

    result = pvlib.atmosphere.get_relative_airmass(
        numpy.asarray(sun_zenith_apparent.array_deg), model="kastenyoung1989"
    )

    return AirMass(array=XP.asarray(result), units="")


def air_mass_relative_young1994(*, sun_zenith: Angle, **_: typing.Any) -> AirMass:
    """
    Compute relative air mass at sea level from (true) zenith of Sun using pvlib's
    young1994 implementation.

    Implements pv_workflows.atmosphere.SupportsAirMassRelativeZenith.
    """

    result = pvlib.atmosphere.get_relative_airmass(
        numpy.asarray(sun_zenith.array_deg), model="young1994"
    )

    return AirMass(array=XP.asarray(result), units="")
