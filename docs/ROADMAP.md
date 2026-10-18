# Support and Roadmap {#control_support_roadmap}

For source organization and synthesis flow, see the [architecture guide](ARCHITECTURE.md).

## Current scope

State-space/plant models and the available feedback/observer formulations have
unit tests. These include H2/H-infinity and
polytopic cases; each formulation's assumptions are documented in its header.
This is not a claim of support for every control design problem.

Numerical and optimization capabilities depend on Akreon Linalg and Akreon Convex.
Their unsupported targets or cones do not become supported through Control.
A solved synthesis problem alone does not establish closed-loop performance.

## Remaining work

- Model predictive control (MPC): constrained finite-horizon formulations and receding-horizon execution.
- Moving-horizon estimation (MHE): constrained estimation, arrival costs, and window updates.
- Control barrier functions (CBF): safety-filter formulations and infeasibility handling.
- More C++ and Python examples with feasibility and closed-loop validation.
- Numerical hardening: gain-recovery conditioning diagnostics, scaling sensitivity,
  and formulation-specific regression tests.

These items carry no delivery commitment. Solver/KKT regularization belongs to
Convex; control-specific formulations and gain recovery belong here.
