# Tests

Catch2 unit tests cover plant port selection, zero-order-hold discretization,
state feedback and observers, and nominal and polytopic H2/H∞ synthesis.

Use the [build and test commands](../README.md#build-and-test) from the repository root.

Run a single suite with `ctest --test-dir build -R state_feedback_test --output-on-failure`.
See the [root README](../README.md#dependencies) for compiler and dependency requirements.

The [native examples](../README.md#native-control--estimation-examples) register
their built-in checks with CTest when `AKREON_CONTROL_BUILD_EXAMPLES=ON`.
Optional binding tests are described in the [binding guide](../bindings/README.md).
