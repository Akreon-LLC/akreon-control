# Control bindings

Build/install and notebook commands live in the [README](../README.md#python-bindings-and-notebooks).
The Python distribution is `akreon-control`; import it as
`akreon_native`. The C ABI target is `Akreon::ControlC`.

## Boundary

`c/` maps validated runtime matrix views into C++ template instances and returns
caller-owned gains/diagnostics through the C ABI. `python/` handles NumPy inputs,
Python configuration/result objects, and plotting helpers. Neither layer owns
optimization algorithms.
CMake options default off; Python and nanobind are required only for Python builds.

## Compiled shapes

The current bindings support continuous-time float64 models with dimensions
selected in `c/src/controls/common/supported_dimensions.hpp`.

| Family | Compiled dimensions |
| --- | --- |
| State feedback/observer | 4 states; 1 control / 2 measurements |
| H2 feedback | `(states, disturbances, controls, performance) = (4, 1, 1, 5)` |
| H-infinity feedback | `(4, 1, 1, 5)`, `(4, 1, 1, 4)`, `(7, 1, 1, 4)`, `(9, 2, 1, 5)` |
| H2/H-infinity observer | `(states, disturbances, performance, measurements) = (4, 3, 4, 2)` |
| Polytopic families | Two vertices with the corresponding 4-state CartPole dimensions above |

The augmented H-infinity shapes support the quarter-car dynamic-weight and
mixed-sensitivity designs. Exact tuples avoid compiling unused Cartesian products.
Unsupported shapes raise `ValueError` in Python or return an unsupported-dimension
status through C. Extending the shape list requires rebuilding the binary.

## Results

Results contain `K` or `L`, solver status/events, objectives or gamma as applicable,
and canonical/numerical diagnostics. Inspect the result status before using a gain.
Plotting dependencies are optional and imported through `akreon_native.plot`.
