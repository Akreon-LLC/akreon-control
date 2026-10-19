# Double-integrator MPC

A constrained linear MPC example following the finite-horizon formulation in
[Borrelli, Bemporad, and Morari](https://sites.google.com/berkeley.edu/mpc-lab/mpc-course-material).
Native H2 synthesis supplies the terminal feedback; `Akreon::Convex` solves the
condensed QP at each update.

## Model and horizon

For position and velocity `x = [p, v]`, with acceleration input `u`, the exact
held-input discrete model, computed by `control::c2d_zoh`, is

```text
x_next = A*x + B*u
A = [1, dt; 0, 1]      B = [dt²/2; dt]
```

[main.cpp](main.cpp) sets `dt = 0.2 s`, horizon `N = 20` (4 seconds), and initial
state `[4 m, 0 m/s]`. The controller regulates to the origin with `|u| <= 1 m/s²`
and `|v| <= 2 m/s`.

At each update, solve

```text
minimize     sum(i=0,...,N-1) [x_i' Q x_i + R*u_i²] + x_N' P x_N
subject to   x_0 = measured state
             x_(i+1) = A*x_i + B*u_i
             |u_i| <= 1                  i = 0,...,N-1
             |v_i| <= 2                  i = 1,...,N

Q = diag(1, 0.1)      R = 0.1
```

Apply only `u_0`, advance the plant, and solve again. The example runs 60 updates
(12 seconds). The first QP is cold; subsequent solves shift the previous input
plan and append bounded terminal feedback `K*x_N`. Cost lifts and interior
primal/dual cone seeds are rebuilt for the shifted plan.

## Native formulation

[mpc.hpp](mpc.hpp) builds `X = Sx*x + Su*U`, where `X = [x_1,...,x_N]` and
`U = [u_0,...,u_(N-1)]`. Substituting this prediction into the objective and
velocity constraints gives a QP over the 20 inputs:

```text
W = blockdiag(Q, ..., Q, P)
H = 2*(Su' W Su + R*I)
q = 2*Su' W Sx*x
```

Input and predicted-velocity bounds form `G*U <= b(x)`. The native model is
`Polyhedron(G, b, U)` with `make_quad_form(H, q, U)`; state-only cost terms are
omitted. The objective is scaled by 0.01 for conic conditioning. Solver settings
allow 128 iterations with `Delta = 10` and default `1e-5` tolerances.

### H2 terminal cost

The current H2 API is continuous-time. We synthesize `u = K*x` for the continuous
double integrator with performance output `[sqrt(Q)*x; sqrt(R)*u]`, then evaluate
that gain on the discrete plant:

```text
Ac = A + B*K
P = Ac' P Ac + Q + K' R K
```

This discrete Lyapunov equation is evaluated iteratively using Akreon Linalg,
with a checked convergence tolerance of `1e-10`. `P` is the sampled H2 policy's
infinite-horizon cost, rather than the synthesis Gramian or a discrete Riccati
solution. No terminal set is imposed.

## Run

From `akreon-control` with a C++26 toolchain:

```sh
cmake -S . -B build/examples -DCMAKE_BUILD_TYPE=Release \
  -DAKREON_CONTROL_BUILD_EXAMPLES=ON \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_FUNCTIONAL_TESTS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=../akreon-linalg \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=../akreon-convex
cmake --build build/examples --parallel 2
build/examples/examples/mpc/mpc_double_integrator
ctest --test-dir build/examples --output-on-failure
```

Edit the configuration, initial state, and horizon in `main.cpp`. The executable
prints `K`, `P`, final state, peak input/velocity, and iteration counts. It checks
solver success, predicted bounds, condensed predictions against direct plant
propagation, and final regulation. The default run reaches position and velocity
below `1e-5`. The first solve takes 16 iterations; the 59 warm solves average
about 6.71 iterations (maximum 10).
