"""pv-workflows"""

import importlib.metadata


def version() -> str:
    """Return current package version."""

    return importlib.metadata.version("pv-workflows")
