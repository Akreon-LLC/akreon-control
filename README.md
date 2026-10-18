# akreon-control

A general-purpose library for optimal and robust control and state estimation,
with model predictive control (MPC), moving-horizon estimation (MHE), and control
barrier functions (CBF) on the roadmap.

Header-only C++26 library. Public target: `Akreon::Control`; installed package:
`AkreonControl`.
Control and estimation APIs use the `control` namespace.
Public headers live under `include/akreon/control/`. Only canonical `akreon/...` include paths are supported.

## Compatibility

See the [support roadmap](docs/ROADMAP.md) for implemented capabilities and
planned work.

## Build and test

Use CMake 3.30+ (for the exported `cxx_std_26` requirement) and a C++26-capable compiler with constexpr math support.
From this checkout:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=/absolute/path/akreon-convex
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

`host-default` and `host-functional-only` presets are also available. Pass the
same local-source overrides when configuring presets. Tests default on only for
the top-level project. Disable tests using `-DBUILD_UNIT_TESTS=OFF
-DBUILD_FUNCTIONAL_TESTS=OFF`.

## Dependencies

The library depends on `akreon-linalg` (`Akreon::Linalg`) and `akreon-convex` (`Akreon::Convex`).
Each dependency resolves in this order:

1. An explicit local checkout via `FETCHCONTENT_SOURCE_DIR_AKREON_LINALG`, `FETCHCONTENT_SOURCE_DIR_AKREON_CONVEX`.
2. An installed CMake package found through `CMAKE_PREFIX_PATH`.
3. FetchContent from the corresponding `Akreon-LLC/akreon-<name>` Git repository,
   pinned to the planned `v0.1.0` release.

The build command above uses local checkouts. No sibling-directory layout or
Git submodules are required. Until release tags exist, use local checkouts or
installed packages. For private Git access or another compatible release, set
`AKREON_LINALG_GIT_REPOSITORY` / `AKREON_LINALG_GIT_TAG`, `AKREON_CONVEX_GIT_REPOSITORY` / `AKREON_CONVEX_GIT_TAG`.

Unit tests use installed Catch2 3 or fetch v3.8.1. For offline testing, pass
`-DFETCHCONTENT_SOURCE_DIR_CATCH2=/absolute/path/to/Catch2`.

## Install and consume

The [consumer example](examples/consumer) shows how a separate CMake project
uses the installed library. Ensure `akreon-linalg` and `akreon-convex` are installed
into the same prefix (or include their prefixes in `CMAKE_PREFIX_PATH`). After
building above, install this package, then configure, build, and test the example:

```sh
cmake --install build --prefix /absolute/path/akreon-install
cmake -S examples/consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH=/absolute/path/akreon-install
cmake --build build/consumer
ctest --test-dir build/consumer --output-on-failure
```

The example's [CMakeLists.txt](examples/consumer/CMakeLists.txt) finds the package
and links its `consumer` executable:

```cmake
find_package(AkreonControl 0.1 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Akreon::Control)
```

Its [main.cpp](examples/consumer/main.cpp) includes `<akreon/control/control.hpp>`
and synthesizes H2 state feedback for a two-state plant, recovers the gain, and
checks the gain, closed-loop poles, and objective against the analytic solution.
Follow the same pattern in your own project, setting `CMAKE_PREFIX_PATH` to the
installation prefix. `Akreon::Linalg` and `Akreon::Convex` propagate through
`Akreon::Control`, along with include directories and the C++ standard requirement.
Installed packages resolve their dependencies without fetching source.

## Python bindings and notebooks

Python support is optional; C++ builds do not require Python or nanobind. From
this repository, use Python 3.12 or newer and install the package and example/test
dependencies. Replace the dependency paths with your local checkouts:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ".[examples,test]" \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg \
  --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=/absolute/path/akreon-convex
.venv/bin/python -m pytest bindings/python/tests
.venv/bin/python -m ipykernel install --prefix .venv --name akreon-control --display-name "Akreon Control"
.venv/bin/jupyter lab examples
```

Open [CartPole](examples/cartpole/cartpole.ipynb) or
[quarter-car wheel](examples/quarter_car_wheel/quarter_car_wheel.ipynb) with the
`Akreon Control` kernel. They call `akreon_native.controls` directly and use
`akreon_native.plot` for diagnostics.
Generalized feedback calls specify `exogenous_dim`; observers specify
`performance_dim` to describe the input/output port split.

For an optional C ABI/Python CMake build and native tests:

```sh
cmake -S . -B build/python \
  -DAKREON_CONTROL_BUILD_PYTHON=ON \
  -DAKREON_CONTROL_BUILD_BINDING_TESTS=ON \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_FUNCTIONAL_TESTS=OFF \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=/absolute/path/akreon-convex
cmake --build build/python --parallel 2
ctest --test-dir build/python --output-on-failure
```

Use `AKREON_CONTROL_BUILD_C_BINDINGS=ON` alone for the C ABI without Python.
The [binding contract](bindings/README.md) describes compiled dimensions and results.

## Documentation

The [architecture guide](docs/ARCHITECTURE.md) is the Doxygen landing page.
Contributor workflow is in [CONTRIBUTING.md](CONTRIBUTING.md); test coverage is
in [tests/README.md](tests/README.md).
API and algorithm documentation is generated from the header comments. To build
both together, install Doxygen and Graphviz, then run from this repository:

```sh
doxygen Doxyfile
```

Open `docs/doxygen/html/index.html`. The same generated HTML directory can be
published to GitHub Pages; no separate copy of the architecture guide is needed.

## Citation

If you use this software in research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

## License and notices

Licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for attribution
information.
