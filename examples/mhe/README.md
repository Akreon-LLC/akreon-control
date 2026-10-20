# Double-integrator MHE

Estimate position and velocity from noisy position measurements and known
acceleration inputs. The same ZOH double integrator as the [MPC example](../mpc/README.md)
is used, with a native QP over the past 10 intervals (11 measurements).

## Model and objective

```text
x_(i+1) = A*x_i + B*(u_i + d_i)    x = [position, velocity]
y_i     = [1, 0]*x_i + measurement noise
```

`control::c2d_zoh` supplies `A` and `B` at `dt = 0.2 s`. The unknown scalar `d_i`
represents an acceleration disturbance. For a window starting at sample `a`,
solve

```text
minimize    ||x_a - prior_a||²_Lambda
            + sum(i=a,...,k-1) d_i² / sigma_d²
            + sum(i=a,...,k) (y_i - position_i)² / sigma_y²
subject to  x_(i+1) = A*x_i + B*(u_i + d_i)
            |velocity_i| <= 2 m/s

Lambda = diag(1 / sigma_p², 1 / sigma_v²)
sigma_p = 1 m; sigma_v = 0.5 m/s
sigma_d = 0.05 m/s²; sigma_y = 0.1 m
```

Return the last estimated state `x_k`. The decision vector contains the two
initial-state corrections and ten disturbances; dynamics are condensed into
`X = E*z + f(prior, inputs)`. The objective is scaled by `sigma_y²` and assembled
with `make_quad_form`; velocity bounds use `Polyhedron`.

## Arrival prior and warm start

The arrival cost uses fixed weights. When the window advances, a fixed-gain
observer consumes only the departing position measurement, then predicts the
prior forward using its known input. Measurements still in the window are not
used in this update. This is an approximate arrival cost, not an exact
marginalized information matrix.

The first full window is solved cold. Later solves shift the estimated initial
state and disturbance sequence, append zero disturbance, and rebuild the conic
cost lifts and interior primal/dual seed. The native solver uses a 128-iteration
cap, `Delta = 10`, and default `1e-5` tolerances.

## Run

Build using the [native example commands](https://github.com/Akreon-LLC/akreon-control#build-and-run-native-examples), then run from `akreon-control`:

```sh
build/examples/examples/mhe/mhe_double_integrator
```

Edit the configuration, initial states, input, disturbance, and noise seed in
[main.cpp](main.cpp). The 24-second simulation uses uniform measurement noise
with standard deviation 0.1 m and a small sinusoidal acceleration disturbance.
The estimator receives only measurements and applied inputs; truth is used for
simulation and error reporting. Estimation begins after filling the first window.

The executable checks solver success, velocity bounds, condensed dynamics, and
estimation errors. Typical position/velocity RMSE is 0.069 m / 0.072 m/s, versus
0.101 m for raw position and 0.689 m/s for finite-difference velocity.
