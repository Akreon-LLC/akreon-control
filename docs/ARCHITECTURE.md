# Akreon Control Architecture

Akreon Control owns control and estimation models, formulations, and execution.
This guide describes their source organization and data flow.
For supported scope and future work, see the [support roadmap](ROADMAP.md).

## Source organization

Paths below are relative to `include/akreon/control/`.

| Path | Responsibility |
| --- | --- |
| `control_traits.hpp` | Time domains, model concepts, state/input/output types, and port roles |
| `model/state_space.hpp` | Continuous/discrete state-space representations |
| `model/plant_model.hpp` | State-space models with typed input and output port layouts |
| `feedback/` | State-feedback synthesis, H2/H-infinity and polytopic formulations, and gain recovery |
| `estimation/` | Luenberger, H2/H-infinity, and polytopic observer formulations |
| `control.hpp` | Public umbrella header |
| `bindings/` (repository root) | Optional C ABI and Python adapters |

Public control APIs use `control`. Numerical types come from `linalg`, while
optimization expressions, canonicalization, and solver results come from Convex.

## Modeling and synthesis flow

1. Express a continuous or discrete model with `StateSpace`. Dimensions and
   scalar types are part of the matrix types.
2. Wrap it in `PlantModel` when synthesis requires role-tagged input/output
   ports, such as control, measurement, disturbance, or performance channels.
3. Select a feedback or observer formulation. The formulation assembles its
   optimization variables, objective, and constraints using Convex APIs.
4. Canonicalize and solve through Akreon Convex, inspecting its result and
   numerical status.
5. Recover controller or observer gains from the solved variables. Gain recovery
   belongs here because its interpretation depends on the control formulation.

Polytopic formulations describe synthesis across a collection of plant models.
They share the modeling and solver layers rather than implementing a separate
optimization stack. Supported families and their assumptions are documented in
the individual feedback and estimation headers.

## Dependency boundary

Akreon Linalg owns matrices, spaces, factorizations, and generic numerical kernels.
Akreon Convex owns DCP composition, cones, canonicalization, and solver iteration.
Control owns plant semantics, optimal and robust control formulations, estimation,
and gain recovery. Future predictive-control, horizon-estimation, and barrier
formulations belong here; their implementation status lives in the roadmap.
There is no dependency on akreon-dynamics. Simulation results can be used to
construct model matrices without introducing a library dependency in either direction.

## API reference

Use Doxygen's namespace, class, and file indexes for model contracts, formulation
assumptions, and gain-recovery algorithms documented in the headers.

## Language boundary

Optional bindings adapt runtime arrays to the compile-time C++ APIs. The
binding contract is maintained in `bindings/README.md`; build and notebook usage
is in the repository README. The numerical algorithms stay in the C++ libraries.
