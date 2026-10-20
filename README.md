# akreon-control

Header-only C++26 control and state estimation, built on [akreon-linalg](https://github.com/Akreon-LLC/akreon-linalg) and [akreon-convex](https://github.com/Akreon-LLC/akreon-convex).

[Documentation](https://akreon-llc.github.io/akreon-control/) · [Architecture](docs/ARCHITECTURE.md) · [Support and roadmap](docs/ROADMAP.md) · [Native examples](examples/) · [Python bindings](bindings/README.md) · [Releases](https://github.com/Akreon-LLC/akreon-control/releases)

[![DOI](https://zenodo.org/badge/1424743167.svg)](https://doi.org/10.5281/zenodo.23485956)

## Capabilities

- Fixed-size continuous/discrete state-space models, role-tagged plant ports, and zero-order-hold discretization.
- State-feedback and Luenberger observer design, H₂/H∞ synthesis, and polytopic formulations.
- Convex problem construction, controller/observer gain recovery, and numerical diagnostics.
- Optional Python/C bindings for selected compiled dimensions.

MPC, MHE, and CLF-CBF-QP are demonstrated as native examples; reusable formulation APIs remain planned. See the [roadmap](docs/ROADMAP.md) for support boundaries.

## Native control & estimation examples

| Example | Demonstrates |
|---|---|
| [H₂ state feedback](examples/consumer/main.cpp) | Plant construction, SDP synthesis, and gain recovery |
| [MPC](examples/mpc/README.md) | Constrained double-integrator control with terminal feedback and warm starts |
| [MHE](examples/mhe/README.md) | Position/velocity estimation with an arrival prior and shifted warm starts |
| [CLF-CBF-QP](examples/clf_cbf_qp/README.md) | Adaptive cruise control with tracking, headway, and actuator constraints |

### Build and run native examples

```sh
cmake -S . -B build/examples -DCMAKE_BUILD_TYPE=Release \
  -DAKREON_CONTROL_BUILD_EXAMPLES=ON \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_FUNCTIONAL_TESTS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=/absolute/path/akreon-convex
cmake --build build/examples --parallel 2
ctest --test-dir build/examples --output-on-failure
```

Example READMEs describe their models, checks, and assumptions. The CLF-CBF-QP simulation does not establish continuous-time safety.

## Python bindings and notebooks

Requires Python 3.12+. Install `akreon-control` and import `akreon_native`. See the [binding contract](bindings/README.md) for supported dimensions and result semantics.

From a local repository checkout:

```sh
python3.12 -m venv .venv

.venv/bin/python -m pip install --upgrade pip

.venv/bin/python -m pip install ".[examples,test]" \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=/absolute/path/akreon-convex
```

Run the binding tests:

```sh
.venv/bin/python -m pytest bindings/python/tests
```

Launch the notebooks:

```sh
.venv/bin/python -m ipykernel install \
  --prefix .venv \
  --name akreon-control \
  --display-name "Akreon Control"

.venv/bin/jupyter lab examples
```

Open the [CartPole](examples/cartpole/cartpole.ipynb) or [quarter-car wheel](examples/quarter_car_wheel/quarter_car_wheel.ipynb) notebook using the Akreon Control kernel.

The optional C ABI can also be built independently with `AKREON_CONTROL_BUILD_C_BINDINGS=ON`.

## Dependencies

Requires CMake 3.30+ and a C++26 compiler with constexpr math support. Linalg and Convex resolve from local `FETCHCONTENT_SOURCE_DIR_AKREON_*` overrides, installed packages through `CMAKE_PREFIX_PATH`, then FetchContent (default `v0.1.0`). Override revisions with `AKREON_LINALG_GIT_TAG` and `AKREON_CONVEX_GIT_TAG`.

## Build and test

For a standard C++ build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The top-level project enables its unit and functional tests by default. They can be disabled using:

```sh
-DBUILD_UNIT_TESTS=OFF -DBUILD_FUNCTIONAL_TESTS=OFF
```

Tests use Catch2 3. Optional Python/C ABI bindings and native MPC/MHE/CLF-CBF-QP examples are controlled by separate CMake options.

The production C++ library has no Python dependency.

## Use in another CMake project

The public target is `Akreon::Control`, exported by the `AkreonControl` package.

```cmake
find_package(AkreonControl 0.1 CONFIG REQUIRED)

add_executable(my_controller main.cpp)
target_link_libraries(my_controller PRIVATE Akreon::Control)
```

Include the public API using:

```cpp
#include <akreon/control/control.hpp>
```

The CMake target propagates its public include paths, C++ standard requirement, and Linalg/Convex dependencies.

To install Akreon Control:

```sh
cmake --install build --prefix /absolute/path/akreon-install
```

Ensure Akreon Linalg and Akreon Convex are also installed and discoverable through the same prefix or `CMAKE_PREFIX_PATH`.

The [consumer example](examples/consumer) demonstrates installed-package consumption and H₂ state-feedback synthesis.

## Documentation and verification

See [architecture](docs/ARCHITECTURE.md), [support and roadmap](docs/ROADMAP.md), [contributing](CONTRIBUTING.md), and [security](SECURITY.md).

Generate API documentation with Doxygen and Graphviz installed:

```sh
doxygen Doxyfile
```

Open `docs/doxygen/html/index.html`. Tests cover implemented cases, not every dimension or conditioning regime.

## Citation and license

For research citation, use [CITATION.cff](CITATION.cff). Licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
