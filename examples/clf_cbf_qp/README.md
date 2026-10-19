# Adaptive cruise control with a CLF-CBF-QP

Native C++ adaptive cruise control inspired by
[Ames et al. (2017)](https://arxiv.org/abs/1609.06408), using net acceleration
and a time-headway zeroing barrier. `Akreon::Convex` solves the QP at each step.

## Model

Ego speed `v` and lead speed `v_l` are in m/s; bumper gap `D` is in metres.
The input `a` is net acceleration in m/s², assuming an ideal actuator:

```text
v_dot   = a
v_l_dot = a_l(t)
D_dot   = v_l - v
```

The ego targets 25 m/s. The lead starts at 20 m/s, brakes at -2 m/s² from
10–14 s, then accelerates at 1 m/s² from 25–33 s.

## CLF-CBF-QP

For speed error `e = v - v_des`, define

```text
V = e²/2                         V_dot = e*a
h = D - D_0 - T*v                h_dot = v_l - v - T*a
```

Use standstill gap `D_0 = 5 m` and headway `T = 1.8 s`. Solve:

```text
minimize_(a, delta)  1/2 (a - a_nom)² + rho/2 delta²
subject to          e*a + c*V <= delta                        (soft CLF)
                    v_l - v - T*a + gamma*(h - epsilon) >= 0  (hard CBF)
                    -3 <= a <= 2                             (actuator)
                    delta >= 0

a_nom = -k*e
c = 1/s; gamma = 0.5/s; k = 0.5/s; rho = 0.01 s²/m²
```

Slack `delta` [m²/s³] allows speed tracking to yield to following distance.
The CBF uses a 0.1 m reserve `epsilon` to offset sampling effects in this
scenario. Diagnostics report physical margin `h` and the reserved CBF residual.

### Native formulation

[acc.hpp](acc.hpp) assembles `[a, delta]` with five linear inequalities:

```cpp
convex::make_problem(x)
    .in_set(convex::Polyhedron(A, b, x))
    .minimize(convex::make_quad_form(P, q, r, x));
```

`P = diag(1, rho)`, `q = [-a_nom, 0]`, and `r = a_nom²/2`.
Constraint rows are scaled before solving with `PrimalDualIPM`: 128 iterations,
`Delta = 10`, and `1e-5` primal, dual, and gap tolerances. Commands require
`OPTIMAL` and physical residuals within `1e-3`; failed solves throw.
Incompatible CBF and actuator bounds raise `InfeasibleQP`.

The first solve is cold; later solves reuse the full primal-dual iterate stored
in `main`. Before reuse, cone variables are blended 1% toward their interior
seeds to avoid boundary stalls. The default run takes 16 iterations initially;
warm solves average about 7.81 iterations, with a maximum of 9.

## Run

From `akreon-control`, with a C++26 toolchain:

```sh
cmake -S . -B build/examples -DCMAKE_BUILD_TYPE=Release \
  -DAKREON_CONTROL_BUILD_EXAMPLES=ON \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_FUNCTIONAL_TESTS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=../akreon-linalg \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=../akreon-convex
cmake --build build/examples --parallel 2
ctest --test-dir build/examples --output-on-failure
build/examples/examples/clf_cbf_qp/clf_cbf_qp_acc
```

Adjust dependency paths for your checkouts. Edit the controller configuration,
initial state, and lead-vehicle profile in [main.cpp](main.cpp). The example runs
45 seconds at 20 ms and prints final speeds, minimum headway margin, and maximum
CLF slack, plus cold and warm iteration counts. CTest runs the same executable, checking each solve and the sampled
headway margin. The default minimum margin is approximately 0.0935 m.

## Assumptions

The model omits drag, actuator lag, sensing error, and delay. The headway barrier
does not account for stopping-distance viability; `h >= 0` alone does not ensure
feasibility under input limits. Accelerations are held between samples and
integrated exactly. The empirical reserve and sampled checks do not establish
continuous-time safety between updates.

## Next steps

See the [roadmap](../../docs/ROADMAP.md#clf-cbf-qp-and-barrier-synthesis) for
general QP assembly and polynomial/SOS verification and synthesis of `h`.
