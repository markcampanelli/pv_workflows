# pv-workflows

Extensible computational workflows for photovoltaic (PV) energy simulation

## Goal

Create an alternative to `pvlib`'s
[`ModelChain`](https://github.com/pvlib/pvlib-python/blob/main/pvlib/modelchain.py)
computational workflow that also accomodates plugging in component-model functions
beyond those provided by `pvlib-python`. The approach here aims to be top down
(workflows focused), rather than bottom up (algorithm focused).

### Design Principles

To acheive the above goal, our approach uses the transformation of data as value
objects (validated, frozen dataclasses) by functions (not class methods!) that satisfy
well defined `typing.Protocol`s. Please be patient as we work out how to best combine
the value-object pattern with Python's `typing.Protocol`, as interfaces in Python are
not conventionally emphasized as much as they are in other languages. 

All interfaces are typed, and the code is also array-implementation agnostic via the
Python Array API. In particular, `pandas` and its alternatives are avoided, and a rather
simple, yet portable, approach is taken to timestamp sequences. Also, true data
immutability can be elusive in Python, but we will try our best. We shall see if
performance suffers...

For some background on these ideas, see
- https://youtu.be/CWYwz3iV1g0?si=wff1cOgKQfQv9VyN
- https://youtu.be/kDDCKwP7QgQ?si=hGj9B9wuZxWC8Q7q
- https://data-apis.org/array-api/latest/

Also, for everyone's sanity, the `pvlib` version is pinned in the reference usage.

## Getting Started

This is very much in the experimental phase. Hopefully it bears fruit.

If you're feeling brave, then try to install the package and from the repo root execute
```terminal
uv run examples/getting_started.py
```

The core data structures and interfaces are defined in the `src/pv_workflows` package.

The reference usage example, i.e., a `pvlib-python` interface to `pv_workflows`, is
defined in the `src/pv_workflows_pvlib` package. Usage is demonstrated in
`examples/getting_started.py`.

## Development

Uses `uv` with the `setuptools` build backend. 

- `uv run ruff format .`
- `uv run ruff check .`
