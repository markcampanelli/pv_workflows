"""pv-workflows"""

import importlib.metadata
import typing

import array_api_compat
import numpy

# Default Python-Array-API implementation is numpy, but using compatible mode.
XP = array_api_compat.array_namespace(numpy.array(()), use_compat=True)


def version() -> str:
    """Return current package version."""

    return importlib.metadata.version("pv-workflows")


def customize_array_api_compat(*xs: typing.Any):
    """Customize default Python Array APIimplementation."""

    global XP
    XP = array_api_compat.array_namespace(xs)
