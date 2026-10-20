# Akreon Control

Optimal and robust control and state estimation in C++26.

## Start here

- [Build, install, and examples](https://github.com/Akreon-LLC/akreon-control#readme)
- [Architecture](ARCHITECTURE.md): source organization and design.
- [Support and roadmap](ROADMAP.md): implemented capabilities and limitations.
- \ref control "API reference": types and functions.

## Workflow

| Area | Purpose |
|---|---|
| Model | Continuous/discrete state-space systems, role-tagged plant ports, and ZOH discretization. |
| Design | State feedback, observers, H₂/H∞ synthesis, and polytopic formulations. |
| Recover | Controller and observer gains with numerical diagnostics. |

## C++ integration

```cpp
#include <akreon/control/control.hpp>
```

Link the `Akreon::Control` CMake target. The library is header-only; dimensions and
mathematical contracts are represented in C++ types.

MPC, MHE, and CLF-CBF-QP are native examples; reusable formulation APIs remain planned. Python/C bindings support selected compiled dimensions.
