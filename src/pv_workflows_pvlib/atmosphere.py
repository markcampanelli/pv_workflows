"""Atmosphere workflows using pvlib."""

import typing

import numpy
import pvlib

from pv_workflows import XP
from pv_workflows.atmosphere import RelativeAirMasses
from pv_workflows.common import Angles


def relative_air_mass_kastenyoung1989(
    *, sun_zenith_apparent: Angles, **_: typing.Any
) -> RelativeAirMasses:
    """
    Compute relative air mass at sea level from apparent zenith of Sun using pvlib's
    kastenyoung1989 implementation.

    Implements pv_workflows.atmosphere.SupportsRelativeAirMassZenithApparent.
    """

    return RelativeAirMasses(
        array=XP.asarray(
            pvlib.atmosphere.get_relative_airmass(
                numpy.asarray(sun_zenith_apparent.to_deg.array), model="kastenyoung1989"
            )
        )
    )


def relative_air_mass_young1994(
    *, sun_zenith: Angles, **_: typing.Any
) -> RelativeAirMasses:
    """
    Compute relative air mass at sea level from (true) zenith of Sun using pvlib's
    young1994 implementation.

    Implements pv_workflows.atmosphere.SupportsRelativeAirMassZenith.
    """

    return RelativeAirMasses(
        array=XP.asarray(
            pvlib.atmosphere.get_relative_airmass(
                numpy.asarray(sun_zenith.to_deg.array), model="young1994"
            )
        )
    )
