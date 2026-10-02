"""
pv-workflows: Extensible computational workflows for photovoltaic (PV) energy simulation

TODO: Formalize this as a collection of computational DAGs in proper documentation.
For front or backside of POA:
  (1) Measured GHI -> Decomposed Components -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (2) Measured GHI, DNI, DHI (at least 2 of 3) -> Transposed Components -> POA-Irradiance Components -> Effective POA Irradiance
  (3) Measured POA Irradiance -> POA-Irradiance Components -> Effective POA Irradiance
"""

import importlib.metadata
import typing

import array_api_compat
import numpy

# Default Python-Array-API implementation is numpy.
XP = array_api_compat.array_namespace(numpy.array(()))


def version() -> str:
    """Return current package version."""

    return importlib.metadata.version("pv-workflows")


def customize_array_api_compat(*xs: typing.Any):
    """Customize default Python Array APIimplementation."""

    global XP
    XP = array_api_compat.array_namespace(xs)
