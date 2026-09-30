# pv-workflows

Extensible computational workflows for photovoltaic (PV) energy simulation

## Goal

Create an alternative to `pvlib`'s
[`ModelChain`](https://github.com/pvlib/pvlib-python/blob/main/pvlib/modelchain.py)
computational workflow that also accommodates plugging in component-model functions
beyond those provided by `pvlib-python`. The approach here aims to be top down
(workflows focused), rather than bottom up (algorithm focused).

Currently, a secondary design goal is to readily accomodate parallized computations
using an API such as Dask's `delayed` decorator, which creates a directed acyclic graph
(DAG) for computational workflows.

### Design Principles

To acheive the above goal, our approach uses the transformation of data as value
objects (validated, frozen dataclasses that include units) by functions (not class
methods!) that satisfy well defined `typing.Protocol`s. Please be patient as we work out
how to best combine the value-object pattern with Python's `typing.Protocol`, as
interfaces in Python are not conventionally emphasized as much as they are in other
languages. Composition of value objects is greatly preferred over class inheritance.

The code is also array-implementation agnostic via the Python Array API. In particular,
`pandas` and its alternatives are avoided, and a rather simple, yet portable, approach
is taken to timestamp sequences, which are not (yet?) supported by the Python Array API.
All interfaces have type hints, but hinting array shapes (and validating broadcast
compatability) is still TBD. Also, true data immutability can be elusive in Python, but
we will try our best. We shall see if/where performance suffers...

For some background on these ideas, see
- https://youtu.be/CWYwz3iV1g0?si=wff1cOgKQfQv9VyN
- https://youtu.be/kDDCKwP7QgQ?si=hGj9B9wuZxWC8Q7q
- https://data-apis.org/array-api/latest/

The core data structures and interfaces are defined in the `src/pv_workflows` package.

The reference usage example, i.e., a `pvlib-python` interface to `pv_workflows`, is
defined in the `src/pv_workflows_pvlib` package and demonstrated in
`examples/getting_started.py`.

Lastly, for everyone's sanity, the `pvlib` version is pinned. It is not this project's
intention to chase versioning issues.

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
