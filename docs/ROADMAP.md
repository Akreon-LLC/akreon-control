# Support and Roadmap {#control_support_roadmap}

## Current scope

State-space/plant models, ZOH discretization, H2/H-infinity feedback and
observers, and polytopic formulations have unit coverage. Native examples cover
[CLF-CBF-QP ACC](../examples/clf_cbf_qp/README.md),
[condensed MPC](../examples/mpc/README.md), and
[position-only MHE](../examples/mhe/README.md), each with simulation checks.
See the [architecture guide](ARCHITECTURE.md) for source organization.

## Planned work

- **MPC:** general horizon assembly, terminal sets, and warm-start policies.
- **MHE:** general constrained estimation, arrival costs, and window updates.
- **CLF-CBF-QP:** general safety filters and barrier synthesis, outlined below.
- **Examples:** more C++ and Python applications with closed-loop validation.
- **Numerics:** gain-recovery conditioning, scaling sensitivity, and regression tests.

Control owns control formulations; Convex owns optimization modeling and solver
numerics. Capabilities depend on the underlying Linalg and Convex support.

## CLF-CBF-QP and barrier synthesis

Build from the [ACC example](../examples/clf_cbf_qp/README.md):

1. Add a second example with multiple barriers or control inputs.
2. Extract reusable online QP assembly accepting CLF/CBF values and Lie
   derivatives, decay functions, input bounds, cost weights, and slack policy.
   Start with control-affine systems and relative-degree-one barriers; extend
   to higher relative degree, uncertainty, and sampled-data formulations.
3. Add polynomial/SOS modeling to **Convex**: polynomial bases, coefficient
   matching, Gram matrices, and semidefinite constraints.
4. Develop offline barrier verification and synthesis in **Control**, given
   dynamics, input limits, a certification domain, and safety specifications.
   Explore alternating solves for coupled barrier/policy searches.
5. Export synthesized `h` and its derivatives through the same online interface
   as hand-designed barriers, retaining the certificate's domain and assumptions.

References: [Ames et al., CBF theory and applications](https://arxiv.org/abs/1903.11199)
and [Clark, CBF verification and synthesis](https://arxiv.org/abs/2104.14001).
