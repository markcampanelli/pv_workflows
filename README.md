# pv-workflows

Extensible computational workflows for photovoltaic (PV) energy simulation

GOAL: Create an alternative to `pvlib`'s `ModelChain` computational workflow that
accomodates component functions beyond those provided by `pvlib-python`. The approach
aims to be top down (workflow focused), rather than bottom up (algorithm focused).


## Getting Started

This is very much in the experimental phase. Hopefully it bears fruit.

If you're feeling brave, then try to install the package and from the repo root execute
```terminal
uv run examples/getting_started.py
```

## Development

Uses `uv` with the `setuptools` build backend. 

- `uv run ruff format .`
- `uv run ruff check .`

For everyone's sanity, the `pvlib` version is pinned.
